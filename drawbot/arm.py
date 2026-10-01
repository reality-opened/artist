"""Arm I/O through teleop.py's command dir ($ARM_CTL, default ./ctl): cmds.jsonl in, status.json out.

Every line appended to cmds.jsonl executes. send() refuses until enable() is called (the CLI only
calls it with --yes), so importing this module or running tests can never move the arm.
"""
import json
import os
import time
from pathlib import Path

from . import REPO

CTL = Path(os.environ.get("ARM_CTL", REPO / "ctl"))
ENABLED = False
# Hard guards (teleop itself rejects targets outside the EEPROM limits): ID2 physical stop ~2010,
# ID4 EEPROM 1718..3782.
HARD = {i: (0, 4095) for i in range(1, 7)} | {2: (2030, 4095), 4: (1718, 3782)}


def enable(on=True):
    global ENABLED
    ENABLED = on


def status():
    with open(CTL / "status.json") as f:
        return json.load(f)


def send(cmd):
    if not ENABLED:
        raise RuntimeError("arm commands disabled (drawbot.arm.enable() / CLI --yes)")
    line = json.dumps(cmd)
    with open(CTL / "cmds.jsonl", "a") as f:
        f.write(line + "\n")
    return line


def wait_idle(timeout=8.0):
    time.sleep(0.12)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if not status()["moving"]:
                return True
        except Exception:
            pass
        time.sleep(0.04)
    return False


def joints(st=None):
    """(targets, positions, raw motors) keyed by int ID."""
    m = (st or status())["motors"]
    return {i: m[str(i)]["target"] for i in range(1, 7)}, {i: m[str(i)]["pos"] for i in range(1, 7)}, m


def check_goal(goal):
    for k, v in goal.items():
        lo, hi = HARD[int(k)]
        if not lo <= int(v) <= hi:
            raise ValueError(f"joint {k} -> {int(v)} outside guard {lo}..{hi}")


def move(goal, speed=12, settle=0.5, timeout=20):
    """Absolute raw-tick goals {id: ticks}; waits until teleop reports idle. Raises if teleop rejected it."""
    check_goal(goal)
    t0 = time.time()
    send({"speed": speed})
    line = send({"joints": {str(k): int(round(v)) for k, v in goal.items()}})
    wait_idle(timeout)
    time.sleep(settle)
    try:
        st = status()
    except Exception:
        return
    last = st.get("last") or {}
    if st.get("t", 0) > t0 and isinstance(last, dict) and last.get("rejected") == line:
        raise RuntimeError(f"teleop rejected {line}: {last.get('error')}")


def level4(a2, a3):
    """Wrist ID4 that keeps the jaws level for shoulder a2 / elbow a3 (field formula)."""
    return int(round(2934 - ((a2 - 2149) + (a3 - 2047))))


def down4(a2, a3):
    """Wrist ID4 pointing the jaws straight down (level + 90 deg)."""
    return level4(a2, a3) + 1024


def pan_for_angle(ang, fr=None):
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    return fr.pan_for_angle(ang)
