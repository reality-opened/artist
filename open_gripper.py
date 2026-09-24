"""Bounded opening of confirmed SO101 gripper ID 6; no EEPROM writes.

Unlike nudge.py this makes one opening move, then releases torque. It is only
for an empty gripper inspected through the cameras, not for holding an object.
"""
import argparse
import json
import time

from arm import capture_dir
from nudge import Servo


def plan_open(values, ticks, torque_limit):
    if not 8 <= ticks <= 114 or not 1 <= torque_limit <= 300:
        raise ValueError("Opening requires 8..114 ticks and torque limit 1..300")
    if values["model"] != 777 or values["mode"] != 0 or values["torque"] != 0:
        raise ValueError("Requires torque-off STS3215 position mode")
    if values["temperature"] >= 50 or not values["voltage_min"] <= values["voltage"] <= values["voltage_max"]:
        raise ValueError("Temperature or supply outside operating bounds")
    low, high, start = values["low"], values["high"], values["position"]
    # A resting gripper can settle a few encoder counts below its lower limit.
    # Only an opening, toward the valid interval, can recover that condition.
    if not 0 <= low < high <= 4095 or not max(0, low - 4) <= start <= high - 8:
        raise ValueError("Position too far outside the gripper limits")
    target = start + ticks
    if not low + 8 <= target <= high - 8:
        raise ValueError("Opening target lacks an eight-tick limit margin")
    if not 0 < values["max_torque_limit"] <= 1000:
        raise ValueError("Invalid hardware torque limit")
    return dict(start=start, staged=max(start, low), target=target,
                torque_limit=min(torque_limit, values["max_torque_limit"]),
                voltage_min=values["voltage_min"], voltage_max=values["voltage_max"])


def open_once(servo, plan, record):
    if servo.read(40) != 0 or abs(servo.read(56, 2) - plan["start"]) > 2:
        raise RuntimeError("Gripper changed since inspection")
    try:
        servo.write(48, plan["torque_limit"], 2)
        servo.write(41, 254)
        servo.write(44, 0, 2)
        servo.write(46, 60, 2)
        servo.write(42, plan["staged"], 2)
        if abs(servo.read(56, 2) - plan["start"]) > 2:
            raise RuntimeError("Gripper shifted during staging")
        servo.write(40, 1)
        servo.write(55, 1)
        servo.write(42, plan["target"], 2)
        deadline, settled = time.monotonic() + 4, 0
        while time.monotonic() < deadline:
            values = {name: servo.read(address, size) for name, address, size in [
                ("position", 56, 2), ("voltage", 62, 1), ("temperature", 63, 1),
                ("status", 65, 1), ("load_raw", 60, 2), ("current_raw", 69, 2),
                ("goal", 42, 2), ("torque", 40, 1)]}
            record(dict(event="opening", time_monotonic=time.monotonic(), **values))
            if not plan["start"] - 4 <= values["position"] <= plan["target"] + 8:
                raise RuntimeError("Gripper left bounded opening interval")
            if values["status"] or values["temperature"] >= 50:
                raise RuntimeError("Motor fault or excessive temperature")
            if not plan["voltage_min"] <= values["voltage"] <= plan["voltage_max"]:
                raise RuntimeError("Supply outside configured bounds")
            if values["goal"] != plan["target"] or values["torque"] != 1:
                raise RuntimeError("Gripper command or torque changed")
            settled = settled + 1 if abs(values["position"] - plan["target"]) <= 4 else 0
            if settled >= 3:
                return
            time.sleep(0.05)
        raise RuntimeError("Gripper did not reach opening target within four seconds")
    finally:
        error = None
        for _ in range(3):
            try:
                servo.write(40, 0)
                record(dict(event="torque_released", position=servo.read(56, 2)))
                error = None
                break
            except Exception as exc:
                error = exc
        if error:
            raise RuntimeError(f"TORQUE RELEASE UNCONFIRMED: {error}")


def main():
    import scservo_sdk as sdk
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--ticks", type=int, default=24)
    parser.add_argument("--torque-limit", type=int, default=150)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    port = sdk.PortHandler(args.port)
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError("Cannot open controller")
        servo = Servo(port, sdk.PacketHandler(0), 6)
        values = servo.inspect()
        plan = plan_open(values, args.ticks, args.torque_limit)
        print(json.dumps(dict(preflight=values, plan=plan, execute=args.execute)), flush=True)
        if args.execute:
            path = capture_dir() / "gripper.jsonl"
            print(f"Telemetry: {path}", flush=True)
            with path.open("x") as output:
                def record(item):
                    output.write(json.dumps(item) + "\n")
                    output.flush()
                record(dict(plan=plan, preflight=values))
                try:
                    open_once(servo, plan, record)
                except BaseException as exc:
                    record(dict(result="failed", error=str(exc)))
                    raise
                record(dict(result="target_reached_and_torque_released"))
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    main()
