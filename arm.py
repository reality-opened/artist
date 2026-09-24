"""Hardware bring-up CLI. Motor commands only ping and read registers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def emit(value):
    print(json.dumps(value, indent=2))


def devices(_args):
    from serial.tools import list_ports

    report = {"serial_ports": [
        {"port": p.device, "description": p.description,
         "vid": p.vid, "pid": p.pid, "serial": p.serial_number}
        for p in list_ports.comports()
    ]}
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "-json", "SPCameraDataType", "SPUSBDataType"],
            capture_output=True, text=True, timeout=30, check=True,
        )
        report["macos"] = json.loads(result.stdout)
        report["note"] = "Empty inventories inside a sandbox do not prove devices are absent."
    emit(report)


def orbbec_list(_args):
    import pyorbbecsdk as ob

    ctx = ob.Context()
    found = ctx.query_devices()
    report = []
    for i in range(found.get_count()):
        device = found.get_device_by_index(i)
        info = device.get_device_info()
        pipeline = ob.Pipeline(device)
        profiles = {}
        for name, sensor in [("color", ob.OBSensorType.COLOR_SENSOR),
                             ("depth", ob.OBSensorType.DEPTH_SENSOR)]:
            profile = pipeline.get_stream_profile_list(sensor).get_default_video_stream_profile()
            profiles[name] = {"width": profile.get_width(), "height": profile.get_height(),
                              "fps": profile.get_fps(), "format": str(profile.get_format())}
        report.append({"name": info.get_name(), "serial": info.get_serial_number(),
                       "firmware": info.get_firmware_version(), "profiles": profiles})
        del pipeline, device
    emit({"orbbec_devices": report})


def capture_dir():
    from uuid import uuid4

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = Path(__file__).resolve().parent / "captures" / f"{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def webcam(args):
    import cv2

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError("Camera did not open. Check its index and macOS Camera permission.")
        frame = None
        for _ in range(10):
            ok, candidate = camera.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
        if frame is None:
            raise RuntimeError("Camera opened but returned no usable frames.")
        path = capture_dir() / "wrist.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError("Failed to save camera image.")
        emit({"image": str(path), "index": args.index, "shape": list(frame.shape)})
    finally:
        camera.release()


def motor_ids(value):
    try:
        ids = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated motor IDs.") from exc
    if not ids or len(ids) != len(set(ids)) or any(i < 1 or i > 253 for i in ids):
        raise argparse.ArgumentTypeError("Use unique IDs from 1 to 253; broadcast is disallowed.")
    return ids


def read_motors(port, packet, ids):
    # STS3215 model/register definitions verified against LeRobot's Feetech table.
    report = []
    for motor_id in ids:
        model, result, error = packet.ping(port, motor_id)
        item = {"id": motor_id, "model_number": model, "communication": result,
                "device_error": error}
        if result == 0 and error == 0 and model == 777:
            values = {}
            for name, address, size in [("position_ticks", 56, 2), ("voltage_raw", 62, 1),
                                        ("temperature_c", 63, 1), ("torque_enabled", 40, 1),
                                        ("mode", 33, 1), ("p_gain", 21, 1),
                                        ("cw_dead_zone", 26, 1), ("ccw_dead_zone", 27, 1),
                                        ("max_torque_limit", 16, 2), ("torque_limit", 48, 2),
                                        ("goal_position", 42, 2), ("speed_raw", 58, 2),
                                        ("load_raw", 60, 2), ("current_raw", 69, 2),
                                        ("status", 65, 1), ("moving", 66, 1)]:
                read = packet.read2ByteTxRx if size == 2 else packet.read1ByteTxRx
                value, status, fault = read(port, motor_id, address)
                if status != 0 or fault:
                    raise RuntimeError(f"Motor {motor_id}, {name}: communication={status}, error={fault}")
                if name == "position_ticks" and value & 0x8000:
                    value = -(value & 0x7FFF)
                values[name] = value
            item.update(values)
        else:
            item["note"] = "No telemetry decoded: ping failed, device fault, or model is not STS3215."
        report.append(item)
    return report


def motors(args):
    import scservo_sdk as sdk

    port = sdk.PortHandler(args.port)
    try:
        if not port.setBaudRate(args.baud):
            raise RuntimeError(f"Cannot open {args.port} at {args.baud} baud.")
        report = read_motors(port, sdk.PacketHandler(0), args.ids)
        emit({"motors": report, "units": "Uncalibrated encoder ticks; not joint angles."})
        if any("note" in item for item in report):
            raise RuntimeError("One or more motors could not be identified as healthy STS3215 devices.")
    finally:
        if port.is_open:
            port.closePort()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="List serial ports and macOS camera inventory").set_defaults(run=devices)
    commands.add_parser("orbbec-list", help="List Orbbec serials, firmware and default profiles").set_defaults(run=orbbec_list)
    cam = commands.add_parser("webcam", help="Save a frame from an explicitly selected camera index")
    cam.add_argument("--index", type=int, required=True)
    cam.set_defaults(run=webcam)
    motor = commands.add_parser("motors", help="Read STS3215 telemetry; never change torque or position")
    motor.add_argument("--port", required=True)
    motor.add_argument("--ids", type=motor_ids, required=True)
    motor.add_argument("--baud", type=int, default=1_000_000)
    motor.set_defaults(run=motors)
    args = parser.parse_args()
    try:
        args.run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
