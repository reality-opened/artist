import unittest
from unittest.mock import patch

from camera_stream import LatestFrame, handler_for, orbbec_bgr


class StreamHTTPTests(unittest.TestCase):
    def test_snapshot_stream_and_disconnect(self):
        import cv2
        import http.client
        import json
        import numpy as np
        import threading
        from http.server import ThreadingHTTPServer

        latest = LatestFrame("test-fixture")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(latest))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            connection.request("GET", "/frame.jpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 503)
            response.read()
            ok, encoded = cv2.imencode(".jpg", np.zeros((20, 30, 3), dtype=np.uint8))
            self.assertTrue(ok)
            jpeg = encoded.tobytes()
            latest.publish(jpeg)
            connection.request("GET", "/frame.jpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            decoded = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (20, 30, 3))
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.readline(), b"--frame\r\n")
            self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
            self.assertEqual(response.readline(), f"Content-Length: {len(jpeg)}\r\n".encode())
            self.assertEqual(response.readline(), b"\r\n")
            self.assertEqual(response.read(len(jpeg)), jpeg)
            latest.fail("Disconnected")
            response.close()
            connection.close()
            connection = http.client.HTTPConnection(*server.server_address, timeout=3)
            connection.request("GET", "/status")
            state = json.loads(connection.getresponse().read())
            self.assertFalse(state["live"])
            self.assertEqual(state["sequence"], 1)
            connection.request("GET", "/frame.jpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 503)
            response.read()
        finally:
            latest.fail("Test shutdown")
            connection.close()
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)


class StreamTests(unittest.TestCase):
    def test_missing_and_stale_frames_are_not_live(self):
        frame = LatestFrame("test")
        self.assertFalse(frame.snapshot()[1]["live"])
        with patch("camera_stream.time.monotonic", return_value=10):
            frame.publish(b"image")
            self.assertTrue(frame.snapshot()[1]["live"])
        with patch("camera_stream.time.monotonic", return_value=13):
            self.assertFalse(frame.snapshot()[1]["live"])

    def test_capture_failure_invalidates_previous_frame(self):
        frame = LatestFrame("test")
        frame.publish(b"image")
        frame.fail("Camera disconnected")
        self.assertFalse(frame.snapshot()[1]["live"])
        self.assertEqual(frame.snapshot()[1]["error"], "Camera disconnected")

    def test_rgb_and_bgr_conversion_preserve_colors(self):
        import numpy as np
        from pyorbbecsdk import OBFormat
        from unittest.mock import Mock

        for fmt, data in [(OBFormat.RGB, [255, 0, 0]), (OBFormat.BGR, [0, 0, 255])]:
            frame = Mock()
            frame.get_data.return_value = bytes(data)
            frame.get_width.return_value = 1
            frame.get_height.return_value = 1
            frame.get_format.return_value = fmt
            np.testing.assert_array_equal(orbbec_bgr(frame), [[[0, 0, 255]]])


if __name__ == "__main__":
    unittest.main()
