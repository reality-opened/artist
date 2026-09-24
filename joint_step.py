"""Small supervised SO101 joint step with live calibration checks and logging.

Stages current goals before enabling selected motors. Releases selected motors
after each step; use only where the arm can remain supported when torque is off.
Unselected motors are observed for unexpected movement. Not a grasp controller.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from arm import capture_dir
from kinematics import JOINTS, Kinematics, ticks_to_radians
from nudge import Servo


CALIBRATION = Path("/Users/zhangbocheng/.cache/huggingface/lerobot/calibration/robots/so_follower/exp23_follower.json")


def make_plan(values, calibration, deltas, torque_limit):
    if not deltas or not 1 <= torque_limit <= 500:
        raise ValueError("Select joints and torque limit 1..500")
    for i, delta in deltas.items():
        cap = 114 if i == 3 else 24
        if i not in range(1, 6) or not 1 <= abs(delta) <= cap:
            raise ValueError("Only arm joints 1..5; elbow maximum 114 ticks, others 24")
    start = [values[i]["position"] for i in range(1, 6)]
    target = [start[i - 1] + deltas.get(i, 0) for i in range(1, 6)]
    for i, name in enumerate(JOINTS, 1):
        v, c = values[i], calibration[name]
        if c["id"] != i or v["model"] != 777 or v["mode"] != 0 or v["torque"] != 0:
            raise ValueError(f"Joint {i}: unexpected ID/model/mode/torque")
        if (v["low"], v["high"], v["homing_offset"]) != (c["range_min"], c["range_max"], c["homing_offset"]):
            raise ValueError(f"Joint {i}: calibration mismatch")
        if v["temperature"] >= 50 or not v["voltage_min"] <= v["voltage"] <= v["voltage_max"]:
            raise ValueError(f"Joint {i}: temperature or voltage out of bounds")
        if not v["low"] + 8 <= min(start[i-1], target[i-1]) <= max(start[i-1], target[i-1]) <= v["high"] - 8:
            raise ValueError(f"Joint {i}: encoder limit margin")
        if not 0 < v["max_torque_limit"] <= 1000:
            raise ValueError(f"Joint {i}: invalid torque limit")
    kin = Kinematics()
    poses = []
    for f in np.linspace(0, 1, 9):
        q = ticks_to_radians(np.array(start) * (1-f) + np.array(target) * f, calibration)
        for name, angle in zip(JOINTS, q):
            lo, hi = kin.joints[name]["limits"]
            if not lo <= angle <= hi:
                raise ValueError(f"{name}: URDF joint limit")
        xyz = kin.fk(q)[:3, 3]
        if not 0.05 <= xyz[0] <= 0.42 or abs(xyz[1]) > 0.30 or not 0.020 <= xyz[2] <= 0.35:
            raise ValueError("Model gripper position outside approach bounds")
        poses.append(xyz.tolist())
    return dict(start=start, target=target, deltas=deltas, torque_limit=torque_limit,
                model_start_xyz=poses[0], model_target_xyz=poses[-1])


def execute(servos, values, plan, record):
    selected = list(plan["deltas"])
    for i in range(1, 7):
        if servos[i].read(40) != 0 or abs(servos[i].read(56, 2) - values[i]["position"]) > 2:
            raise RuntimeError("Arm shifted after inspection")
    try:
        for i in selected:
            s = servos[i]
            s.write(48, min(plan["torque_limit"], values[i]["max_torque_limit"]), 2)
            s.write(41, 254)
            s.write(44, 0, 2)
            s.write(46, 60, 2)
            s.write(42, plan["start"][i-1], 2)
        for i in range(1, 7):
            if abs(servos[i].read(56, 2) - values[i]["position"]) > 2:
                raise RuntimeError("Arm shifted during staging")
        for i in selected:
            servos[i].write(40, 1)
            servos[i].write(55, 1)
        for i in selected:
            servos[i].write(42, plan["target"][i-1], 2)
        deadline, settled = time.monotonic() + 4, 0
        while time.monotonic() < deadline:
            frame = {}
            reached = True
            for i, s in servos.items():
                p = s.read(56, 2)
                v = dict(position=p)
                frame[i] = v
                if i in selected:
                    for name, address, size in [("voltage",62,1), ("temperature",63,1),
                            ("status",65,1), ("load_raw",60,2), ("current_raw",69,2),
                            ("torque",40,1), ("goal",42,2)]:
                        v[name] = s.read(address, size)
                    target, start = plan["target"][i-1], plan["start"][i-1]
                    if not min(start,target)-8 <= p <= max(start,target)+8:
                        raise RuntimeError(f"Joint {i} left bounded interval: {v}")
                    if v["temperature"] >= 50 or v["status"] or v["torque"] != 1 or v["goal"] != target:
                        raise RuntimeError(f"Joint {i} fault or changed command: {v}")
                    if not values[i]["voltage_min"] <= v["voltage"] <= values[i]["voltage_max"]:
                        raise RuntimeError(f"Joint {i} supply out of bounds: {v}")
                    reached &= abs(p - target) <= 4
                elif abs(p - values[i]["position"]) > 16:
                    raise RuntimeError(f"Unselected joint {i} shifted to {p}")
            record(dict(event="step", motors=frame, time_monotonic=time.monotonic()))
            settled = settled + 1 if reached else 0
            if settled >= 3:
                return
            time.sleep(0.05)
        raise RuntimeError("Selected joints did not reach targets within four seconds")
    finally:
        errors = []
        for i in selected:
            for attempt in range(3):
                try:
                    servos[i].write(40, 0)
                    record(dict(event="torque_released", id=i, position=servos[i].read(56, 2)))
                    break
                except Exception as exc:
                    if attempt == 2:
                        errors.append(f"{i}: {exc}")
        if errors:
            raise RuntimeError(f"TORQUE RELEASE UNCONFIRMED: {errors}")


def main():
    import scservo_sdk as sdk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--deltas", required=True, help="JSON object, e.g. {\"2\":24,\"3\":-24}")
    parser.add_argument("--torque-limit", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    deltas = {int(k): v for k, v in json.loads(args.deltas).items()}
    if any(type(v) is not int for v in deltas.values()):
        raise ValueError("Integer tick deltas required")
    calibration = json.loads(CALIBRATION.read_text())
    port = sdk.PortHandler(args.port)
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError("Cannot open controller")
        packet = sdk.PacketHandler(0)
        servos = {i: Servo(port, packet, i) for i in range(1, 7)}
        values = {}
        for i, s in servos.items():
            values[i] = s.inspect()
            raw = s.read(31, 2)
            values[i]["homing_offset"] = -(raw & 2047) if raw & 2048 else raw
        plan = make_plan(values, calibration, deltas, args.torque_limit)
        print(json.dumps(dict(plan=plan, execute=args.execute)), flush=True)
        if args.execute:
            path = capture_dir() / "joint_step.jsonl"
            print(f"Telemetry: {path}", flush=True)
            with path.open("x") as output:
                def record(item):
                    output.write(json.dumps(item) + "\n")
                    output.flush()
                record(dict(plan=plan, preflight=values))
                try:
                    execute(servos, values, plan, record)
                except BaseException as exc:
                    record(dict(result="failed", error=str(exc)))
                    raise
                record(dict(result="target_reached_and_torque_released"))
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    main()
