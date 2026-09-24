"""One small wrist-roll/gripper movement, followed by return and torque release.

For supervised bring-up in a supported rest pose. IDs 5/6 must first be confirmed
as wrist roll/gripper. Does not calibrate or write EEPROM. No motion by default.
"""
import argparse
import json
import time


def check_plan(motor_id, delta, values, torque_limit=150):
    if not 1 <= torque_limit <= 500:
        raise ValueError("Bring-up torque limit must be within 1..500.")
    cap = 114 if motor_id == 5 else 24
    if motor_id not in (5, 6) or not 8 <= abs(delta) <= cap:
        raise ValueError("Use confirmed wrist ID 5 (8..114 ticks) or gripper ID 6 (8..24 ticks).")
    if values["model"] != 777 or values["mode"] != 0 or values["torque"] != 0:
        raise ValueError("Requires an STS3215 in position mode with torque initially off.")
    if values["temperature"] >= 50:
        raise ValueError("Motor is too warm for bring-up.")
    if not values["voltage_min"] <= values["voltage"] <= values["voltage_max"]:
        raise ValueError("Supply voltage is outside the motor's configured bounds.")
    start = values["position"]
    target = start + delta
    low, high = values["low"], values["high"]
    if not 0 <= low < high <= 4095 or not low + 8 <= min(start, target) <= max(start, target) <= high - 8:
        raise ValueError("Start or target violates single-turn limits or their margin.")
    if not 0 < values["torque_limit"] <= 1000:
        raise ValueError("Unexpected torque limit.")
    if not 0 < values["max_torque_limit"] <= 1000:
        raise ValueError("Unexpected hardware maximum torque limit.")
    return {"motor_id": motor_id, "start_ticks": start, "target_ticks": target,
            "delta_ticks": delta, "torque_limit": min(torque_limit, values["max_torque_limit"]),
            "voltage_min": values["voltage_min"], "voltage_max": values["voltage_max"],
            "speed_register": 60, "acceleration_register": 254}


class Servo:
    def __init__(self, port, packet, motor_id):
        self.port, self.packet, self.id = port, packet, motor_id

    def read(self, address, size=1):
        fn = self.packet.read2ByteTxRx if size == 2 else self.packet.read1ByteTxRx
        value, result, error = fn(self.port, self.id, address)
        if result or error:
            raise RuntimeError(f"Read {address}: communication={result}, fault={error}")
        return value

    def write(self, address, value, size=1):
        fn = self.packet.write2ByteTxRx if size == 2 else self.packet.write1ByteTxRx
        result, error = fn(self.port, self.id, address, value)
        if result or error:
            raise RuntimeError(f"Write {address}: communication={result}, fault={error}")
        if self.read(address, size) != value:
            raise RuntimeError(f"Write {address} did not read back correctly")

    def inspect(self):
        return {name: self.read(address, size) for name, address, size in [
            ("model", 3, 2), ("mode", 33, 1), ("torque", 40, 1),
            ("position", 56, 2), ("low", 9, 2), ("high", 11, 2),
            ("voltage", 62, 1), ("voltage_min", 15, 1), ("voltage_max", 14, 1),
            ("temperature", 63, 1), ("torque_limit", 48, 2),
            ("max_torque_limit", 16, 2)]}


def execute(servo, plan, record):
    start, target = plan["start_ticks"], plan["target_ticks"]
    # No writes if the arm has shifted since the plan was read.
    if servo.read(40) != 0 or abs(servo.read(56, 2) - start) > 2:
        raise RuntimeError("Position or torque changed after preflight; inspect again.")
    release_error = None
    try:
        # Stage the present position BEFORE enabling torque to avoid a stale goal jump.
        servo.write(48, plan["torque_limit"], 2)
        servo.write(41, plan["acceleration_register"])
        servo.write(44, 0, 2)
        servo.write(46, 60, 2)
        servo.write(42, start, 2)
        if abs(servo.read(56, 2) - start) > 2:
            raise RuntimeError("Arm moved during setup.")
        servo.write(40, 1)
        # Same enable sequence as EXP-23's installed FeetechMotorsBus.
        servo.write(55, 1)

        for phase, goal in [("out", target), ("return", start)]:
            servo.write(42, goal, 2)
            deadline = time.monotonic() + 4
            settled = 0
            while time.monotonic() < deadline:
                position = servo.read(56, 2)
                telemetry = {name: servo.read(address, size) for name, address, size in [
                    ("voltage_raw", 62, 1), ("load_raw", 60, 2), ("current_raw", 69, 2),
                    ("torque_enabled", 40, 1), ("goal_readback", 42, 2),
                    ("speed_raw", 58, 2), ("temperature_c", 63, 1), ("status", 65, 1)]}
                record({"phase": phase, "goal": goal, "position": position,
                        **telemetry,
                        "time_monotonic": time.monotonic()})
                if position < min(start, target) - 8 or position > max(start, target) + 8:
                    raise RuntimeError("Position left the bounded test range.")
                if telemetry["temperature_c"] >= 50 or telemetry["status"]:
                    raise RuntimeError("Motor reported temperature or status fault.")
                if not plan["voltage_min"] <= telemetry["voltage_raw"] <= plan["voltage_max"]:
                    raise RuntimeError("Supply left the motor's configured voltage limits during movement.")
                if telemetry["torque_enabled"] != 1 or telemetry["goal_readback"] != goal:
                    raise RuntimeError("Motor torque or goal changed during movement.")
                settled = settled + 1 if abs(position - goal) <= 4 else 0
                if settled >= 3:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f"Motor did not reach {phase} goal within four seconds.")
    finally:
        # This test starts torque-off in a supported pose and ends the same way.
        # Retry release if an acknowledgement was lost; never hide a failed release.
        for _ in range(3):
            try:
                servo.write(40, 0)
                record({"event": "torque_released", "motor_id": plan["motor_id"],
                        "time_monotonic": time.monotonic()})
                release_error = None
                break
            except Exception as exc:
                release_error = exc
        if release_error:
            raise RuntimeError(f"TORQUE RELEASE UNCONFIRMED: {release_error}")


def main():
    import scservo_sdk as sdk
    from arm import capture_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", type=int, choices=[5, 6], required=True)
    parser.add_argument("--delta", type=int, required=True)
    parser.add_argument("--torque-limit", type=int, default=150,
                        help="SRAM torque limit, 1..500 (default 150)")
    parser.add_argument("--execute", action="store_true", help="Perform the supervised test; otherwise only inspect")
    args = parser.parse_args()
    port = sdk.PortHandler(args.port)
    try:
        if not port.setBaudRate(1_000_000):
            raise RuntimeError("Could not open motor adapter.")
        servo = Servo(port, sdk.PacketHandler(0), args.id)
        values = servo.inspect()
        plan = check_plan(args.id, args.delta, values, args.torque_limit)
        print(json.dumps({"preflight": values, "plan": plan, "execute": args.execute}, indent=2), flush=True)
        if args.execute:
            path = capture_dir() / "motion.jsonl"
            with path.open("x") as output:
                def record(item):
                    output.write(json.dumps(item) + "\n")
                    output.flush()
                record({"plan": plan, "preflight": values})
                try:
                    execute(servo, plan, record)
                except BaseException as exc:
                    record({"result": "failed", "error": str(exc)})
                    raise
                record({"result": "completed", "torque_enabled": servo.read(40)})
            print(f"Movement and return verified; telemetry: {path}")
    finally:
        if port.is_open:
            port.closePort()


if __name__ == "__main__":
    main()
