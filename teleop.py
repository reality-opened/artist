"""Long-running supervised SO101 controller driven by a command file.

Append one JSON object per line to --commands; status is rewritten to --status.
Commands:
  {"joints": {"4": 3100}}          absolute raw-tick goals (ramped)
  {"delta": {"1": -40}}            relative to current goals
  {"xyz": [x, y, z], "down": true} IK for gripper_frame position (m, URDF base
                                   frame); "down" also points the jaws at the desk
  {"gripper": 1500}                raw gripper goal
  {"speed": 40}                    ramp rate, ticks per 50 ms cycle
  {"release": true}                torque off on every motor and exit
Adopts motors that are already holding (goal = present position if torque is
off). Faults hold position instead of releasing so the arm cannot drop; only
overheating (>= 55 C) or an explicit release turns torque off.
"""
import argparse
import json
import os
import time

import numpy as np

from joint_step import CALIBRATION
from kinematics import JOINTS, Kinematics, ticks_to_radians
from nudge import Servo

ARM_IDS = range(1, 6)


def tolerant_read(servo, address, size=1, tries=3):
    """Read a register; a status error byte (e.g. voltage warning) is returned, not raised."""
    fn = servo.packet.read2ByteTxRx if size == 2 else servo.packet.read1ByteTxRx
    for _ in range(tries):
        value, result, error = fn(servo.port, servo.id, address)
        if result == 0:
            return value, error
    raise RuntimeError(f"ID {servo.id} read {address}: communication={result}")


def tolerant_write(servo, address, value, size=2, tries=3):
    fn = servo.packet.write2ByteTxRx if size == 2 else servo.packet.write1ByteTxRx
    for _ in range(tries):
        result, error = fn(servo.port, servo.id, address, value)
        if result == 0:
            return error
    raise RuntimeError(f"ID {servo.id} write {address}: communication={result}")


def radians_to_ticks(q, calibration):
    return [int(round(q[i] * 4095 / (2 * np.pi) + (calibration[n]["range_min"] + calibration[n]["range_max"]) / 2))
            for i, n in enumerate(JOINTS)]


def _dls(kin, q, target, down, lo, hi, w=0.15):
    for _ in range(200):
        T = kin.fk(q)
        err = [target - T[:3, 3]]
        if down:
            err.append(w * (np.array([0, 0, -1.0]) - T[:3, 2]))
        err = np.concatenate(err)
        if np.linalg.norm(err) < 1e-4:
            break
        J = []
        for i in range(5):
            dq = np.zeros(5); dq[i] = 1e-5
            Tp, Tm = kin.fk(q + dq), kin.fk(q - dq)
            col = [(Tp[:3, 3] - Tm[:3, 3]) / 2e-5]
            if down:
                col.append(w * (Tp[:3, 2] - Tm[:3, 2]) / 2e-5)
            J.append(np.concatenate(col))
        J = np.array(J).T
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(err)), err), lo, hi)
    T = kin.fk(q)
    return q, np.linalg.norm(target - T[:3, 3])


def solve_ik(kin, calibration, start_ticks, xyz, down, limits):
    """Damped least squares on position (+ jaw axis pointing down), multi-seed."""
    q0 = ticks_to_radians(start_ticks, calibration).astype(float)
    tick_lo = ticks_to_radians([limits[i][0] + 2 for i in ARM_IDS], calibration)
    tick_hi = ticks_to_radians([limits[i][1] - 2 for i in ARM_IDS], calibration)
    lo = np.maximum([kin.joints[n]["limits"][0] for n in JOINTS], tick_lo)
    hi = np.minimum([kin.joints[n]["limits"][1] for n in JOINTS], tick_hi)
    target = np.asarray(xyz, float)
    seeds = [q0] + [np.clip(np.array([q0[0], a, b, c, q0[4]]), lo, hi)
                    for a in (-0.8, 0, 0.8) for b in (-0.8, 0, 0.8) for c in (0, 0.8, 1.4)]
    # Prefer jaws straight down; relax toward position-only when out of reach.
    for w in ((0.3, 0.1, 0.03) if down else (0,)):
        results = [_dls(kin, s.copy(), target, down, lo, hi, w) for s in seeds]
        good = [(np.abs(q - q0).sum(), q) for q, cost in results if cost < 0.003]
        if good:
            break
    if not good:
        q = min(results, key=lambda r: r[1])[0]
        T = kin.fk(q)
        raise ValueError(f"IK unreachable: best {T[:3, 3].round(3).tolist()} axis {T[:3, 2].round(2).tolist()}")
    q = min(good, key=lambda g: g[0])[1]
    T = kin.fk(q)
    ticks = radians_to_ticks(q, calibration)
    for i, t in zip(ARM_IDS, ticks):
        if not limits[i][0] <= t <= limits[i][1]:
            raise ValueError(f"IK joint {i} target {t} outside {limits[i]}")
    return ticks, T[:3, 3].tolist(), T[:3, 2].tolist()


def main():
    import scservo_sdk as sdk
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True)
    ap.add_argument("--commands", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--torque-limit", type=int, default=500)
    ap.add_argument("--gripper-torque", type=int, default=250)
    args = ap.parse_args()
    calibration = json.loads(CALIBRATION.read_text())
    kin = Kinematics()
    port = sdk.PortHandler(args.port)
    if not port.setBaudRate(1_000_000):
        raise RuntimeError("Cannot open controller")
    packet = sdk.PacketHandler(0)
    servos = {i: Servo(port, packet, i) for i in range(1, 7)}
    limits, goals = {}, {}
    for i, s in servos.items():
        low, high = s.read(9, 2), s.read(11, 2)
        limits[i] = (low + 8, high - 8)
        pos = s.read(56, 2)
        goals[i] = s.read(42, 2) if s.read(40) else pos
        if not s.read(40):
            s.write(42, pos, 2)
        s.write(48, args.gripper_torque if i == 6 else args.torque_limit, 2)
        s.write(41, 254)
        s.write(46, 0, 2)  # speed limited by our own ramp
        s.write(40, 1)
    targets = dict(goals)
    speed = 10
    hot = {}
    log = open(args.status + ".log", "a")
    offset = os.path.getsize(args.commands) if os.path.exists(args.commands) else 0
    last_result = "started"
    try:
        while True:
            if os.path.exists(args.commands) and os.path.getsize(args.commands) > offset:
                with open(args.commands) as f:
                    f.seek(offset)
                    lines = f.read()
                    offset = f.tell()
                for line in lines.splitlines():
                    if not line.strip():
                        continue
                    try:
                        c = json.loads(line)
                        if c.get("release"):
                            raise KeyboardInterrupt
                        if "speed" in c:
                            speed = int(np.clip(c["speed"], 2, 80))
                        new = dict(targets)
                        for k, v in c.get("joints", {}).items():
                            new[int(k)] = int(v)
                        for k, v in c.get("delta", {}).items():
                            new[int(k)] = targets[int(k)] + int(v)
                        if "gripper" in c:
                            new[6] = int(c["gripper"])
                        extra = {}
                        if "xyz" in c:
                            ticks, xyz, axis = solve_ik(kin, calibration, [targets[i] for i in ARM_IDS],
                                                        c["xyz"], c.get("down", False), limits)
                            new.update(dict(zip(ARM_IDS, ticks)))
                            extra = dict(model_xyz=xyz, jaw_axis=axis)
                        for i, t in new.items():
                            if not limits[i][0] <= t <= limits[i][1]:
                                raise ValueError(f"joint {i} target {t} outside {limits[i]}")
                        targets = new
                        last_result = dict(accepted=c, targets=targets, **extra)
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                        last_result = dict(rejected=line, error=str(exc))
                    log.write(json.dumps(dict(t=time.time(), result=last_result)) + "\n"); log.flush()
            for i in servos:
                step = int(np.clip(targets[i] - goals[i], -speed, speed))
                if step:
                    goals[i] += step
                    tolerant_write(servos[i], 42, goals[i])
            state = {}
            for i, s in servos.items():
                pos, err = tolerant_read(s, 56, 2)
                state[i] = dict(pos=pos, goal=goals[i], target=targets[i], load=tolerant_read(s, 60, 2)[0],
                                temp=tolerant_read(s, 63)[0], volt=tolerant_read(s, 62)[0] / 10, err=err)
                # A single garbled read (e.g. 247 C) must not drop the arm: require
                # three consecutive plausible over-temperature readings.
                hot[i] = hot.get(i, 0) + 1 if 55 <= state[i]["temp"] <= 100 else 0
                if hot[i] >= 3:
                    raise RuntimeError(f"Overheat on {i}")
            q = ticks_to_radians([state[i]["pos"] for i in ARM_IDS], calibration)
            T = kin.fk(q)
            status = dict(t=time.time(), motors=state, model_xyz=T[:3, 3].round(4).tolist(),
                          jaw_axis=T[:3, 2].round(3).tolist(), moving=any(goals[i] != targets[i] for i in servos),
                          last=last_result)
            tmp = args.status + ".tmp"
            with open(tmp, "w") as f:
                json.dump(status, f)
            # Windows refuses the replace while a reader has status open; skip a cycle.
            for _ in range(3):
                try:
                    os.replace(tmp, args.status)
                    break
                except PermissionError:
                    time.sleep(0.01)
            time.sleep(0.05)
    except KeyboardInterrupt:
        release = True
    except Exception as exc:
        # Communication errors: leave torque on so servos keep their last goal.
        log.write(json.dumps(dict(t=time.time(), event="error_holding", error=repr(exc))) + "\n")
        release = "Overheat" in str(exc)
    if release:
        for s in servos.values():
            for _ in range(3):
                try:
                    s.write(40, 0)
                    break
                except Exception:
                    pass
        log.write(json.dumps(dict(t=time.time(), event="released")) + "\n")
    port.closePort()


if __name__ == "__main__":
    main()
