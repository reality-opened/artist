"""Key logger page: the arm's contact sensor while calibrating key positions.

  .venv/bin/python keylog.py [--port 8770]      then open http://127.0.0.1:8770 in Safari and keep it focused
(The system Python's Tk 8.5 aborts on macOS 26, so this is a local web page instead of a Tk window.)
Every key press/release on the page is appended as JSON to ctl/keylog.jsonl, timestamped on arrival:
  {"t": unix time, "type": "down" | "up", "keysym": "i", "char": "i", "repeat": false}
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).resolve().parent / "ctl" / "keylog.jsonl"

PAGE = b"""<!doctype html><meta charset=utf-8><title>arm keylog</title>
<style>body{font:28px Menlo,monospace;background:#111;color:#eee;margin:40px}#k{font-size:48px;color:#7f7}</style>
<p>arm keylog: keep this tab focused</p><div id=k></div>
<script>
const shown=[];
function send(type,e){fetch('/ev',{method:'POST',keepalive:true,
  body:JSON.stringify({type,keysym:e.key===' '?'space':e.key,char:e.key.length===1?e.key:'',repeat:e.repeat})});}
addEventListener('keydown',e=>{e.preventDefault();send('down',e);shown.push(e.key===' '?'\xe2\x90\xa3':e.key);
  document.getElementById('k').textContent=shown.slice(-16).join(' ');});
addEventListener('keyup',e=>{e.preventDefault();send('up',e);});
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(PAGE)

    def do_POST(self):
        ev = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ev["t"] = time.time()
        with open(LOG, "a") as f:
            f.write(json.dumps(ev) + "\n")
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    LOG.touch()
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
