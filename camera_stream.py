"""Local live camera view. Requires an explicit webcam index or Orbbec serial."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import subprocess
import tempfile
import threading
import time


class LatestFrame:
    def __init__(self, source):
        self.source = source
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0
        self.received = None
        self.error = None

    def publish(self, jpeg):
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.received = time.monotonic()
            self.error = None
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = str(error)
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            age = None if self.received is None else time.monotonic() - self.received
            live = self.jpeg is not None and not self.error and age < 2
            return self.jpeg, {"source": self.source, "live": bool(live),
                               "sequence": self.sequence, "frame_age_seconds": age,
                               "error": self.error}


def orbbec_bgr(frame):
    import cv2
    import numpy as np
    from pyorbbecsdk import OBFormat

    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    height, width, fmt = frame.get_height(), frame.get_width(), frame.get_format()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.BGR:
        return data.reshape(height, width, 3)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt in (OBFormat.YUYV, OBFormat.UYVY):
        code = cv2.COLOR_YUV2BGR_YUY2 if fmt == OBFormat.YUYV else cv2.COLOR_YUV2BGR_UYVY
        return cv2.cvtColor(data.reshape(height, width, 2), code)
    codes = {OBFormat.I420: cv2.COLOR_YUV2BGR_I420,
             OBFormat.NV12: cv2.COLOR_YUV2BGR_NV12, OBFormat.NV21: cv2.COLOR_YUV2BGR_NV21}
    if fmt in codes:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), codes[fmt])
    raise RuntimeError(f"Unsupported color format: {fmt}")


def webcam_frames(index, stop):
    import cv2
    import sys

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}; check device and Camera permission.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        last_frame = time.monotonic()
        while not stop.is_set():
            ok, frame = camera.read()
            if not ok:
                if time.monotonic() - last_frame > 5:
                    raise RuntimeError("Camera returned no frames for five seconds.")
                stop.wait(0.05)
                continue
            last_frame = time.monotonic()
            yield frame
    finally:
        camera.release()


def orbbec_frames(serial, stop):
    import pyorbbecsdk as ob

    context = ob.Context()
    devices = context.query_devices()
    if devices.get_count() == 0:
        raise RuntimeError("No Orbbec device detected.")
    device = devices.get_device_by_serial_number(serial)
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    config.enable_stream(profile)
    started = False
    try:
        pipeline.start(config)
        started = True
        last_frame = time.monotonic()
        while not stop.is_set():
            frames = pipeline.wait_for_frames(500)
            color = frames.get_color_frame() if frames else None
            if color is not None:
                last_frame = time.monotonic()
                yield orbbec_bgr(color)
            elif time.monotonic() - last_frame > 5:
                raise RuntimeError("Orbbec returned no color frames for five seconds.")
    finally:
        if started:
            pipeline.stop()


# The Gemini sometimes enumerates only its IR sensor (1280x800) instead of RGB.
NAMED_CAMERA_SIZES = {
    "USB Camera": "640x480",
    "Orbbec Gemini 2 RGB Camera": "640x480",
    "Orbbec Gemini 2 IR Camera": "1280x800",
}


def named_camera_frames(name, stop):
    """Select camera hardware by its exact name, independent of OpenCV indices."""
    import cv2
    import numpy as np
    import sys

    if name not in NAMED_CAMERA_SIZES:
        raise ValueError(f"Choose one of: {', '.join(NAMED_CAMERA_SIZES)}.")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
               *(["-f", "dshow", "-framerate", "30", "-video_size", NAMED_CAMERA_SIZES[name],
                  "-i", f"video={name}"] if sys.platform == "win32" else
                 ["-f", "avfoundation", "-framerate", "30", "-video_size", NAMED_CAMERA_SIZES[name],
                  "-i", f"{name}:none"]),
               "-an", "-c:v", "mjpeg", "-q:v", "4",
               "-f", "image2pipe", "pipe:1"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        pending = bytearray()
        try:
            while not stop.is_set():
                chunk = process.stdout.read(4096)
                if not chunk:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace")[-2000:] or "Camera capture ended")
                pending.extend(chunk)
                if len(pending) > 16_000_000:
                    raise RuntimeError("Invalid or oversized camera JPEG stream")
                while True:
                    start = pending.find(b"\xff\xd8")
                    end = pending.find(b"\xff\xd9", max(0, start) + 2)
                    if start < 0 or end < 0:
                        break
                    encoded = bytes(pending[start:end + 2])
                    del pending[:end + 2]
                    yield cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            process.stdout.close()


ROTATIONS = {"none": None, "cw": 0, "ccw": 2, "180": 1}  # cv2.ROTATE_* codes


def capture(frames, latest, stop, rotate="none"):
    import cv2

    try:
        for frame in frames:
            if stop.is_set():
                break
            if frame is None or frame.size == 0:
                raise RuntimeError("Empty camera frame.")
            if ROTATIONS[rotate] is not None:
                frame = cv2.rotate(frame, ROTATIONS[rotate])
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("JPEG encoding failed.")
            latest.publish(jpeg.tobytes())
    except Exception as exc:
        latest.fail(exc)
    finally:
        frames.close()
        if not stop.is_set() and latest.error is None:
            latest.fail("Capture ended.")


PAGE = b'''<!doctype html><meta charset="utf-8"><title>Robot camera</title>
<style>body{font:18px system-ui;background:#15191f;color:#eee;margin:24px}
img{max-width:100%;max-height:80vh} .stale{opacity:.2}</style>
<h1>Robot camera</h1><p id="state">Waiting for camera...</p><img id="view">
<script>
const view=document.querySelector('#view'), status=document.querySelector('#state');
async function refresh(){try{
 const s=await (await fetch('/status',{cache:'no-store'})).json();
 status.textContent=s.live?`${s.source} | Frame ${s.sequence} | ${s.frame_age_seconds.toFixed(2)}s old`:
 `Camera unavailable: ${s.error||'No fresh frames'}`;
 view.className=s.live?'':'stale';
 if(s.live&&!view.getAttribute('src'))view.src='/stream.mjpg';
 if(!s.live)view.removeAttribute('src');
}catch(e){status.textContent='Camera server disconnected';view.removeAttribute('src');}}
setInterval(refresh,500);refresh();
</script>'''


def handler_for(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def response(self, status, mime, body):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                jpeg, state = latest.snapshot()
                if self.path == "/":
                    self.response(200, "text/html; charset=utf-8", PAGE)
                elif self.path == "/status":
                    self.response(200, "application/json", json.dumps(state).encode())
                elif self.path == "/frame.jpg":
                    if state["live"]:
                        self.response(200, "image/jpeg", jpeg)
                    else:
                        self.response(503, "application/json", json.dumps(state).encode())
                elif self.path == "/stream.mjpg":
                    if not state["live"]:
                        self.response(503, "application/json", json.dumps(state).encode())
                        return
                    self.connection.settimeout(5)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sequence = -1
                    while True:
                        with latest.condition:
                            latest.condition.wait_for(lambda: latest.sequence != sequence or latest.error, timeout=1)
                            jpeg, state = latest.snapshot()
                        if not state["live"]:
                            return
                        if state["sequence"] == sequence:
                            continue
                        sequence = state["sequence"]
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                else:
                    self.response(404, "text/plain", b"Not found")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="Explicit OpenCV camera index")
    source.add_argument("--orbbec-serial", help="Orbbec serial from arm.py orbbec-list")
    source.add_argument("--device-name", choices=list(NAMED_CAMERA_SIZES),
                        help="Select macOS camera by name through installed ffmpeg")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rotate", choices=list(ROTATIONS), default="none",
                        help="Rotate frames before serving (the wrist webcam is mounted sideways: cw)")
    args = parser.parse_args()
    stop = threading.Event()
    if args.device_name:
        latest = LatestFrame(args.device_name)
        frames = named_camera_frames(args.device_name, stop)
    elif args.index is not None:
        latest = LatestFrame(f"webcam:{args.index}")
        frames = webcam_frames(args.index, stop)
    else:
        latest = LatestFrame(f"orbbec:{args.orbbec_serial}")
        frames = orbbec_frames(args.orbbec_serial, stop)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(latest))
    worker = threading.Thread(target=capture, args=(frames, latest, stop, args.rotate), daemon=True)
    worker.start()
    print(f"Camera view: http://127.0.0.1:{args.port} (check /status for live frames)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        latest.fail("Server stopping")
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
