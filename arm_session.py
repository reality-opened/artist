"""Supervised JSON-line SO101 control session that holds between small moves.

No motion without --execute. Commands: {"status":true}, {"move":{"3":-57}},
{"park":true}, {"release":true}. Move deltas apply to measured positions.
All goals are staged before torque is enabled. Faults/EOF release torque.
Camera observation is the operator's responsibility between commands.
"""
import argparse
import json
import select
import sys
import time

import numpy as np

from arm import capture_dir
from joint_step import CALIBRATION
from kinematics import JOINTS, Kinematics, ticks_to_radians
from nudge import Servo


class Session:
    def __init__(self, servos, calibration, record, torque_limit=300):
        self.servos, self.calibration, self.record = servos, calibration, record
        self.kin = Kinematics()
        self.values = {}
        self.start = {}
        self.goals = {}
        self.enabled = False
        self.torque_limit = torque_limit
        if not 1 <= torque_limit <= 500:
            raise ValueError("Arm torque limit must be 1..500")
        if set(range(1, 6)) - set(servos):
            raise ValueError("Arm joints 1..5 must all be held")
        for i, name in enumerate(JOINTS + ["gripper"], 1):
            if i not in servos:
                continue
            s, c = servos[i], calibration[name]
            v = s.inspect()
            raw = s.read(31, 2)
            offset = -(raw & 2047) if raw & 2048 else raw
            if (c["id"], c["range_min"], c["range_max"], c["homing_offset"]) != (i, v["low"], v["high"], offset):
                raise ValueError(f"Joint {i}: calibration mismatch")
            if v["model"] != 777 or v["mode"] != 0 or v["torque"] != 0:
                raise ValueError(f"Joint {i}: unexpected model/mode/torque")
            if not v["low"] + 8 <= v["position"] <= v["high"] - 8:
                raise ValueError(f"Joint {i}: initial position lacks limit margin")
            if v["temperature"] >= 50 or not v["voltage_min"] <= v["voltage"] <= v["voltage_max"]:
                raise ValueError(f"Joint {i}: initial temperature/voltage")
            self.values[i], self.start[i] = v, v["position"]
        self.goals = self.start.copy()

    def enable(self):
        for i, s in self.servos.items():
            if s.read(40) or abs(s.read(56, 2) - self.start[i]) > 2:
                raise RuntimeError("Arm changed after preflight")
        # Mark before the first write so cleanup covers partial setup.
        self.enabled = True
        for i, s in self.servos.items():
            cap = 150 if i == 6 else self.torque_limit
            s.write(48, min(cap, self.values[i]["max_torque_limit"]), 2)
            s.write(41, 254)
            s.write(44, 0, 2)
            s.write(46, 60, 2)
            s.write(42, self.start[i], 2)
        for i, s in self.servos.items():
            if abs(s.read(56, 2) - self.start[i]) > 2:
                raise RuntimeError("Arm shifted during staging")
        for s in self.servos.values():
            s.write(40, 1)
            s.write(55, 1)
        self.record(dict(event="enabled", start=self.start, torque_limit=self.torque_limit))

    def observe(self):
        frame = {}
        for i, s in self.servos.items():
            v = {name: s.read(address, size) for name, address, size in [
                ("position",56,2), ("voltage",62,1), ("temperature",63,1),
                ("status",65,1), ("load_raw",60,2), ("current_raw",69,2),
                ("torque",40,1), ("goal",42,2)]}
            frame[i] = v
            if v["temperature"] >= 50 or v["status"] or v["torque"] != 1 or v["goal"] != self.goals[i]:
                raise RuntimeError(f"Joint {i}: fault or unexpected command: {v}")
            if not self.values[i]["voltage_min"] <= v["voltage"] <= self.values[i]["voltage_max"]:
                raise RuntimeError(f"Joint {i}: supply out of bounds: {v}")
            if not self.values[i]["low"] - 4 <= v["position"] <= self.values[i]["high"] + 4:
                raise RuntimeError(f"Joint {i}: position outside limits: {v}")
            if abs(v["position"] - self.goals[i]) > 160:
                raise RuntimeError(f"Joint {i}: excessive tracking error: {v}")
        self.record(dict(event="observation", time_monotonic=time.monotonic(), motors=frame))
        return frame

    def check_targets(self, positions, targets):
        for i, p in targets.items():
            v = self.values[i]
            if not v["low"] + 8 <= p <= v["high"] - 8:
                raise ValueError(f"Joint {i}: target violates limit margin")
        for f in np.linspace(0, 1, 9):
            ticks = [(1-f)*positions[i]+f*targets[i] for i in range(1,6)]
            q = ticks_to_radians(ticks, self.calibration)
            for name, angle in zip(JOINTS, q):
                lo, hi = self.kin.joints[name]["limits"]
                if not lo <= angle <= hi:
                    raise ValueError(f"{name}: model joint limit")
            xyz = self.kin.fk(q)[:3,3]
            if not 0.05 <= xyz[0] <= 0.42 or abs(xyz[1]) > 0.30 or not 0.020 <= xyz[2] <= 0.35:
                raise ValueError(f"Model gripper outside approach bounds: {xyz}")

    def move(self, deltas):
        frame = self.observe()
        positions = {i:v["position"] for i,v in frame.items()}
        targets = self.goals.copy()
        for i, delta in deltas.items():
            if i not in self.servos or type(delta) is not int or not 1 <= abs(delta) <= 114:
                raise ValueError("Invalid move: maximum 114 ticks (ten degrees) per joint")
            targets[i] = positions[i] + delta
        self.check_targets(positions, targets)
        self.record(dict(event="move", positions=positions, targets=targets, deltas=deltas))
        for i in deltas:
            self.servos[i].write(42, targets[i], 2)
            self.goals[i] = targets[i]
        return self.settle()

    def settle(self):
        deadline, settled = time.monotonic()+4, 0
        while time.monotonic() < deadline:
            frame = self.observe()
            reached = all(abs(v["position"]-self.goals[i]) <= 4 for i,v in frame.items())
            settled = settled + 1 if reached else 0
            if settled >= 3:
                break
            time.sleep(0.1)
        # A timeout is reported, never converted into target success. Torque
        # stays on so an operator can inspect the actual achieved position.
        result = dict(targets_reached=settled >= 3, holding=True, goals=self.goals,
                      positions={i:v["position"] for i,v in frame.items()},
                      loads={i:v["load_raw"] for i,v in frame.items()})
        self.record(dict(event="move_result", **result))
        return result

    def park(self):
        # Return using small joint-space increments, checking each model path.
        for _ in range(80):
            frame = self.observe()
            p = {i:v["position"] for i,v in frame.items()}
            if all(abs(p[i]-self.start[i]) <= 16 for i in p):
                self.release()
                return dict(parked=True, released=True, positions=p)
            changes = {i:int(np.clip(self.start[i]-p[i],-24,24)) for i in p if abs(self.start[i]-p[i]) > 16}
            self.move(changes)
        raise RuntimeError("Could not return to session start")

    def release(self):
        errors = []
        for i,s in self.servos.items():
            for attempt in range(3):
                try:
                    s.write(40,0)
                    self.record(dict(event="torque_released", id=i, position=s.read(56,2)))
                    break
                except Exception as exc:
                    if attempt == 2:
                        errors.append(f"{i}: {exc}")
        self.enabled = False
        if errors:
            raise RuntimeError(f"TORQUE RELEASE UNCONFIRMED: {errors}")


def main():
    import scservo_sdk as sdk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--torque-limit", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--without-gripper", action="store_true",
                        help="Leave ID 6 untouched, torque off (e.g. when it reads outside its range)")
    args = parser.parse_args()
    port = sdk.PortHandler(args.port)
    session = None
    path = capture_dir()/"session.jsonl"
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError("Cannot open controller")
        with path.open("x") as output:
            def record(item):
                output.write(json.dumps(item)+"\n")
                output.flush()
            packet = sdk.PacketHandler(0)
            session = Session({i:Servo(port,packet,i) for i in range(1,6 if args.without_gripper else 7)},
                              json.loads(CALIBRATION.read_text()), record, args.torque_limit)
            print(json.dumps(dict(preflight=session.start, telemetry=str(path), execute=args.execute)),flush=True)
            if not args.execute:
                return
            try:
                session.enable()
                print(json.dumps(session.settle()),flush=True)
                last_command = time.monotonic()
                while session.enabled:
                    ready,_,_ = select.select([sys.stdin],[],[],0.5)
                    if not ready:
                        session.observe()
                        if time.monotonic()-last_command > 120:
                            print(json.dumps(dict(event="idle_return", result=session.park())),flush=True)
                            break
                        continue
                    line = sys.stdin.readline()
                    if not line:
                        break
                    last_command = time.monotonic()
                    try:
                        command = json.loads(line)
                        if "move" in command:
                            result = session.move({int(k):v for k,v in command["move"].items()})
                        elif command.get("park"):
                            result = session.park()
                        elif command.get("release"):
                            session.release()
                            result = dict(released=True)
                        elif command.get("status"):
                            result = dict(motors=session.observe())
                        else:
                            raise ValueError("Unknown command")
                        print(json.dumps(result),flush=True)
                    except (ValueError,KeyError) as exc:
                        print(json.dumps(dict(command_rejected=str(exc))),flush=True)
            finally:
                if session.enabled:
                    session.release()
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    main()
