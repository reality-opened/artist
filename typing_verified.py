"""Type with saved arm poses, releasing on accepted VS Code text and verifying every character.

VS Code's screen reader accessibility mode must be enabled for read-only AX text feedback.
No software key events or editor writes are used to enter the requested text.
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import ApplicationServices as AX

import keytype as kt
from drawbot import arm


def editor_text(app):
    err, focused = AX.AXUIElementCopyAttributeValue(app, "AXFocusedUIElement", None)
    if err:
        raise RuntimeError(f"cannot read focused editor: {err}")
    err, role = AX.AXUIElementCopyAttributeValue(focused, "AXRole", None)
    err2, value = AX.AXUIElementCopyAttributeValue(focused, "AXValue", None)
    if err or err2 or role != "AXTextArea" or not isinstance(value, str) or not value:
        raise RuntimeError("VS Code editor text is unavailable")
    return value


def preflight():
    st = arm.status()
    if time.time() - st["t"] > 1:
        raise RuntimeError("controller status is stale")
    if set(st["motors"]) != set("123456"):
        raise RuntimeError("missing motor telemetry")
    if any(m["err"] or m["temp"] >= 50 or not 4.5 <= m["volt"] <= 6.5
           for m in st["motors"].values()):
        raise RuntimeError("motor telemetry failed preflight")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text")
    ap.add_argument("--depth", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    st = kt.load()
    if not a.text or any(ch not in st["keys"] or "c2" not in st["keys"][ch] for ch in a.text):
        raise SystemExit("text contains uncalibrated keys")
    if not 0 <= a.depth <= 30 or not .1 <= a.timeout <= 2:
        raise SystemExit("depth or timeout outside trial bounds")
    if not a.yes:
        return print("dry run: add --yes")
    if kt.frontmost() != "Code":
        raise SystemExit("VS Code must be frontmost")
    pid = int(subprocess.check_output(["pgrep", "-x", "Code"]).decode().strip().splitlines()[0])
    app = AX.AXUIElementCreateApplication(pid)
    current = editor_text(app)
    targets = list(re.finditer(r'print\("([^"\n]*)"\)', current))
    if len(targets) != 1:
        raise SystemExit('expected one print string target')
    target = targets[0]
    insertion = target.start(1)
    existing = target.group(1)
    original = current[:insertion] + current[target.end(1):]
    start_idx = 0
    if a.resume:
        while start_idx < min(len(existing), len(a.text)) and existing[start_idx] == a.text[start_idx]:
            start_idx += 1
        if start_idx != len(existing):
            err, focused = AX.AXUIElementCopyAttributeValue(app, "AXFocusedUIElement", None)
            err, selected = AX.AXUIElementCopyAttributeValue(focused, "AXSelectedText", None)
            if err or selected != existing[start_idx:]:
                raise SystemExit("select the incorrect suffix before resuming")
    elif existing:
        raise SystemExit('expected an empty print string; use --resume for a verified prefix')
    session = Path("captures") / f"typing-verified-{time.time_ns()}"
    session.mkdir()
    arm.enable()
    with (session / "events.jsonl").open("w") as log:
        end_idx = len(a.text) if a.limit is None else min(len(a.text), start_idx + a.limit)
        for idx in range(start_idx, end_idx):
            ch = a.text[idx]
            preflight()
            if kt.frontmost() != "Code" or editor_text(app) != current:
                raise RuntimeError("app or editor changed before the next press")
            k = st["keys"][ch]
            kt.key_pose(k["q_contact"], k["c2"] - 30)
            preflight()
            if kt.frontmost() != "Code" or editor_text(app) != current:
                raise RuntimeError("app or editor changed during travel")
            expected = original[:insertion] + a.text[:idx + 1] + original[insertion:]
            t0 = time.monotonic()
            try:
                kt.send_checked({2: k["c2"] + a.depth}, 40)
                while time.monotonic() - t0 < a.timeout:
                    preflight()
                    if editor_text(app) != current:
                        break
                    time.sleep(.005)
            finally:
                kt.send_checked({2: k["c2"] - kt.RELEASE}, 80)
                if not arm.wait_idle(3):
                    raise RuntimeError("release ramp did not finish")
            time.sleep(.25)
            actual = editor_text(app)
            report = {"index": idx, "key": ch, "expected": expected, "actual": actual,
                      "accepted": actual == expected, "t": time.time(), "motors": arm.status()["motors"]}
            log.write(json.dumps(report) + "\n")
            log.flush()
            subprocess.run(["screencapture", "-x", "-l", "70476", str(session / f"editor-{idx:02d}.png")], check=True)
            if actual != expected:
                raise RuntimeError(f"key {ch!r} failed text verification: {actual!r}; report {session}")
            current = actual
            print(f"verified {idx+1}/{len(a.text)}: {a.text[:idx+1]!r}", flush=True)
            for port, name in ((8765, "wrist"), (8766, "gemini")):
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=1) as r:
                    (session / f"{name}-{idx:02d}.jpg").write_bytes(r.read())
    print(f"verified through {end_idx}/{len(a.text)}; evidence: {session}", flush=True)


if __name__ == "__main__":
    main()
