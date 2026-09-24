"""Pen control on top of teleop.py: relative Cartesian moves with fixed jaw pitch.

Model (URDF) is only trusted for *relative* motion around the current pose.
Per-joint gravity sag (target - measured) is captured once and kept as offset.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

from joint_step import CALIBRATION
from kinematics import Kinematics, ticks_to_radians
from teleop import radians_to_ticks

# Same directory teleop.py uses for --commands/--status (default: ./ctl, gitignored).
D = Path(os.environ.get("ARM_CTL", Path(__file__).parent / "ctl"))
CAL = json.loads(CALIBRATION.read_text())
KIN = Kinematics()
# ID2 has a physical stop near 2010 (2026-09-24); ID4 EEPROM range is 1718..3782.
LIMITS = {1: (700, 3400), 2: (2030, 3300), 3: (700, 3300), 4: (1718, 3782)}
STATE = D / "pen_state.json"


def status():
    # On Windows the open can briefly fail while teleop replaces the file.
    for _ in range(20):
        try:
            with open(D / "status.json") as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.01)
    with open(D / "status.json") as f:
        return json.load(f)


def send(cmd):
    with open(D / "cmds.jsonl", "a") as f:
        f.write(json.dumps(cmd) + "\n")


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


def fk(ticks5):
    T = KIN.fk(ticks_to_radians(ticks5, CAL))
    return T[:3, 3], T[:3, 2]


def solve(q_ticks, target, axis_z):
    """4 joints (pan, lift, elbow, flex); roll fixed. Match xyz and jaw-axis z."""
    q = ticks_to_radians(q_ticks, CAL).astype(float)
    for _ in range(60):
        T = KIN.fk(q)
        err = np.concatenate([target - T[:3, 3], [0.2 * (axis_z - T[2, 2])]])
        if np.linalg.norm(err) < 2e-5:
            break
        J = np.zeros((4, 4))
        for i in range(4):
            dq = np.zeros(5); dq[i] = 1e-5
            Tp, Tm = KIN.fk(q + dq), KIN.fk(q - dq)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / 2e-5
            J[3, i] = 0.2 * (Tp[2, 2] - Tm[2, 2]) / 2e-5
        q[:4] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(4), err)
    return radians_to_ticks(q, CAL)


class Pen:
    """Keeps a *virtual* measured-pose chain so sag offsets don't accumulate."""

    def __init__(self, reset=False):
        s = status()
        m = s["motors"]
        if STATE.exists() and not reset:
            st = json.loads(STATE.read_text())
            self.q = st["q"]; self.offset = {int(k): v for k, v in st["offset"].items()}
            self.axis_z = st["axis_z"]
        else:
            self.q = [m[str(i)]["pos"] for i in range(1, 6)]
            self.offset = {i: m[str(i)]["target"] - m[str(i)]["pos"] for i in range(1, 5)}
            self.axis_z = float(fk(self.q)[1][2])
        self.p = fk(self.q)[0]
        self.save()

    def save(self):
        STATE.write_text(json.dumps(dict(q=[int(v) for v in self.q], offset=self.offset, axis_z=self.axis_z)))

    def goto(self, xyz, speed=4):
        xyz = np.asarray(xyz, float)
        q = solve(self.q, xyz, self.axis_z)
        joints = {}
        for i in range(1, 5):
            t = int(q[i - 1] + self.offset[i])
            lo, hi = LIMITS[i]
            if not lo <= t <= hi:
                raise ValueError(f"joint {i} -> {t} outside {LIMITS[i]}")
            joints[str(i)] = t
        send({"joints": joints, "speed": speed})
        self.q = list(q[:4]) + [self.q[4]]
        self.p = fk(self.q)[0]
        return wait_idle()

    def line(self, xyz, step=0.003, speed=4):
        a, b = self.p.copy(), np.asarray(xyz, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for k in range(1, n + 1):
            self.goto(a + (b - a) * k / n, speed)
        self.save()

    def rel(self, dx=0, dy=0, dz=0, **kw):
        self.line(self.p + np.array([dx, dy, dz]), **kw)


if __name__ == "__main__":
    pen = Pen(reset="--reset" in sys.argv)
    print("pen p", pen.p.round(4), "axis_z", round(pen.axis_z, 3), "offset", pen.offset)
