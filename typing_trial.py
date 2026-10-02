"""Bounded physical key trial; release on the first accepted browser key event."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import keytype as kt
from drawbot import arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if not a.yes:
        return print("dry run: add --yes")
    assert 0 <= a.depth <= 30 and 0 < a.timeout <= 3
    st = arm.status()
    assert time.time() - st["t"] < 1
    assert all(m["err"] == 0 for m in st["motors"].values())
    assert kt.frontmost() == "Safari"
    k = kt.load()["keys"][a.key]
    arm.enable()
    kt.key_pose(k["q_contact"], k["c2"] - 30)
    assert kt.frontmost() == "Safari"
    off = kt.KEYLOG.stat().st_size
    start = time.monotonic()
    samples = []
    evs = []
    captured = False
    try:
        kt.send_checked({2: k["c2"] + a.depth}, 40)
        while time.monotonic() - start < a.timeout:
            evs, _ = kt.new_events(off)
            elapsed = time.monotonic() - start
            st = arm.status()
            if time.time() - st["t"] > 1:
                raise RuntimeError("stale controller during press")
            if not samples or elapsed - samples[-1]["elapsed"] > 0.1:
                samples.append({"elapsed": round(elapsed, 3), "motors": st["motors"]})
            if any(e["type"] == "down" for e in evs):
                break
            if elapsed > .3 and not captured:
                for port, path in ((8766, "/tmp/typing-held.jpg"), (8765, "/tmp/typing-held-wrist.jpg")):
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=.5) as r:
                        Path(path).write_bytes(r.read())
                captured = True
            time.sleep(.01)
    finally:
        kt.send_checked({2: k["c2"] - kt.RELEASE}, 80)
        assert arm.wait_idle(3)
    time.sleep(.4)
    evs, _ = kt.new_events(off)
    report = {"key": a.key, "depth": a.depth, "timeout": a.timeout, "samples": samples, "events": evs}
    report_path = Path("captures") / f"typing-trial-{a.key}-{time.time_ns()}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"events": evs, "elapsed": round(time.monotonic() - start, 3),
                      "last_shoulder": samples[-1]["motors"]["2"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
