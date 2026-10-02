"""Type on the laptop keyboard with the SO101's closed jaw tip.

Calibration uses keylog.py (local page at :8770, open in Safari and keep it focused) as the contact sensor:
`calibrate KEY` predicts the key's plane xy from a keyboard-grid fit, travels there high, then descends
slowly along the model's vertical until keylog reports a key-down, and lifts at once. Each contact (whatever
key it hit) refines the grid fit; the contact joint targets of the wanted key are saved for fast taps.
`tap KEY` then presses a calibrated key quickly (no sensor needed), traveling high between keys.

  python keytype.py show
  python keytype.py calibrate i [--yes]
  python keytype.py tap i [--depth 3] [--hold 0.15] [--yes]
  python keytype.py lift [--yes]
State: drawbot_state/keymap.json. Model h is NOT trusted as key clearance (it was off by >25 mm).
"""
import argparse
import json
import time

import numpy as np

from drawbot import REPO, arm
from drawbot.kinmodel import model

KEYLOG = arm.CTL / "keylog.jsonl"
STATE = REPO / "drawbot_state" / "keymap.json"
PITCH = 19.05
HOVER_H = 95.0  # model h for travel; tilt allowed (jaws-down is wrist-limited up there)
DESCENT_SPEED = 5.0  # mm/s
MAX_DESCENT = 75.0
LIFT_MM = 15.0
SENSOR_APP = "Safari"  # app showing keylog.py's page

# MacBook keyboard: (column center in key units from the left edge, row: 0 number row .. 4 bottom row)
ROWS = {
    0: "` 1 2 3 4 5 6 7 8 9 0 - =",
    1: "q w e r t y u i o p [ ] \\",
    2: "a s d f g h j k l ; '",
    3: "z x c v b n m , . /",
}
ROW_START = {0: 0.5, 1: 2.0, 2: 2.25, 3: 2.75}
LAYOUT = {k: (ROW_START[r] + i, r) for r, s in ROWS.items() for i, k in enumerate(s.split())}
LAYOUT[" "] = (6.75, 4)
LAYOUT["ArrowLeft"] = (12.0, 4.25)  # half-height arrow keys sit in the lower half of the bottom row
LAYOUT["ArrowRight"] = (14.0, 4.25)
TK_NAMES = {"space": " ", "grave": "`", "minus": "-", "equal": "=", "bracketleft": "[", "bracketright": "]",
            "backslash": "\\", "semicolon": ";", "apostrophe": "'", "comma": ",", "period": ".", "slash": "/"}


def key_of(ev):
    k = TK_NAMES.get(ev["keysym"], ev["keysym"])
    return k.lower() if len(k) == 1 else k


def load():
    if STATE.exists():
        st = json.loads(STATE.read_text())
        st["obs"] = [o for o in st["obs"] if "h" in o]  # measured-frame obs from the first version are dropped
        return st
    # Seed: the closed tip pressed "7" at model (-35, -31.4) on 2026-10-02; keyboard up ~ +x, right ~ -y.
    return {"obs": [{"key": "7", "xy": [-35.0, -31.4]}], "keys": {}}


def save(st):
    STATE.write_text(json.dumps(st, indent=1))


def fit(obs):
    """Similarity map grid (col, row) -> plane xy, A = s R. Falls back to the assumed orientation."""
    pts = [(LAYOUT[o["key"]], o["xy"]) for o in obs if o["key"] in LAYOUT]
    g = np.array([p[0] for p in pts], float)
    p = np.array([p[1] for p in pts], float)
    A0 = np.array([[0.0, -PITCH], [-PITCH, 0.0]])  # col -> -y, row -> -x
    if not pts:
        raise SystemExit("no commanded-frame contacts yet")
    if len(pts) < 2 or np.ptp(g, 0).max() < 0.9:
        b = (p - g @ A0.T).mean(0)
        return A0, b
    # Procrustes on centered points, keeping the handedness of A0 (reflection) fixed.
    gc, pc = g - g.mean(0), p - p.mean(0)
    F = np.array([[1, 0], [0, -1.0]])  # A0 = s R F for some rotation R
    H = (gc @ F.T).T @ pc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = S.sum() / (gc ** 2).sum()
    if len({tuple(x) for x in g}) < 3:
        s = PITCH  # two points: trust the known pitch
    A = s * R @ F
    b = p.mean(0) - A @ g.mean(0)
    return A, b


def predict(st, key):
    A, b = fit(st["obs"])
    return A @ np.array(LAYOUT[key], float) + b


def ticks():
    tg, pos, _ = arm.joints()
    return np.array([tg[i] for i in range(1, 6)], float), np.array([pos[i] for i in range(1, 6)], float)


def vertical(pos, w_axis=80.0):
    """Joint ticks (ID1..ID4) per mm of tip descent, keeping the jaw axis (model Jacobian, relative only)."""
    m = model()
    J = np.zeros((6, 4))
    for j in range(4):
        dq = np.zeros(5)
        dq[j] = 2
        (p1, a1, _), (p0, a0, _) = m.fk(pos + dq), m.fk(pos - dq)
        J[:3, j], J[3:, j] = (p1 - p0) / 4, (a1 - a0) / 4
    W = np.diag([1, 1, 1, w_axis, w_axis, w_axis])
    dq, *_ = np.linalg.lstsq(W @ J, W @ np.r_[0, 0, -1.0, 0, 0, 0], rcond=None)
    return dq


def send_q(q4, speed):
    arm.send({"speed": int(speed)})
    arm.send({"joints": {str(i + 1): int(round(v)) for i, v in enumerate(q4)}})


def lift(mm=40.0):
    """Straight up along the model vertical from the current targets, then wait."""
    tg, pos = ticks()
    send_q(tg[:4] - mm * vertical(pos), 30)
    arm.wait_idle(8)


def roll():
    return ticks()[1][4]


def cmd_pose(q4):
    """Model xyz of commanded joint targets (the 'commanded frame': same command -> same physical pose)."""
    return model().fk(np.r_[np.asarray(q4, float)[:4], roll()])[0]


def goto_cmd(xyz, speed=15):
    """IK seeded at the current targets; sends the solution as-is (no sag correction: commanded frame)."""
    tg, pos = ticks()
    r = model().ik(np.asarray(xyz, float), axis=np.array([0, 0, -1.0]), w_axis=8.0, roll=pos[4], seed=tg[:4])
    if r.res > 4:
        raise RuntimeError(f"IK residual {r.res:.1f} mm for {np.round(xyz, 1).tolist()}")
    send_q(r.q, speed)
    arm.wait_idle(15)
    time.sleep(0.3)


def key_h(st):
    hs = [o["h"] for o in st["obs"] if "h" in o]
    return float(np.mean(hs)) if hs else None


def travel(st, xy):
    """Up to key height + 45 at the current xy, across, then down to key height + 25 (commanded frame)."""
    h = key_h(st)
    if h is None:
        h = HOVER_H - 45
    tg, _ = ticks()
    here = cmd_pose(tg)
    goto_cmd([here[0], here[1], max(here[2], h + 45)])
    goto_cmd([xy[0], xy[1], h + 45])
    goto_cmd([xy[0], xy[1], h + 25])
    print(f"travel to {np.round(xy, 1).tolist()}, hovering ~25 mm above key height {h:.1f} (commanded frame)")


def new_events(offset):
    if not KEYLOG.exists():
        return [], offset
    with open(KEYLOG) as f:
        f.seek(offset)
        lines = f.read()
        offset = f.tell()
    return [json.loads(x) for x in lines.splitlines() if x.strip()], offset


def descend_until_key():
    """Slow descent along the vertical; returns (key, q_contact_cmd, v, xy_at_contact) or None."""
    tg, pos = ticks()
    v = vertical(np.r_[tg[:4], pos[4]])
    q0 = tg[:4].copy()
    offset = KEYLOG.stat().st_size if KEYLOG.exists() else 0
    arm.send({"speed": 20})
    s, dt = 0.0, 0.05
    t_next = time.time()
    while s < MAX_DESCENT:
        evs, offset = new_events(offset)
        downs = [e for e in evs if e["type"] == "down"]
        if downs:
            q_c = q0 + s * v
            send_q(q_c - LIFT_MM * v, 80)  # lift first, think later
            t_lift = time.time()
            arm.wait_idle(3)
            _, pos_c = ticks()
            key = key_of(downs[0])
            print(f"contact: {key!r} at s={s:.1f} mm (lift sent {1000 * (t_lift - downs[0]['t']):.0f} ms after key-down)")
            return key, q_c, v, model().fk(pos_c)[0][:2]
        s += DESCENT_SPEED * dt
        arm.send({"joints": {str(i + 1): int(round(x)) for i, x in enumerate(q0 + s * v)}})
        t_next += dt
        while time.time() < t_next:
            evs, offset2 = new_events(offset)
            if any(e["type"] == "down" for e in evs):
                break
            time.sleep(0.005)
    print("no key within", MAX_DESCENT, "mm; lifting")
    send_q(q0, 30)
    return None


def frontmost():
    import subprocess
    return subprocess.run(["osascript", "-e", 'tell application "System Events" to get name of first '
                           'application process whose frontmost is true'], capture_output=True, text=True).stdout.strip()


def calibrate(key, tries=4):
    st = load()
    for _ in range(tries):
        app = frontmost()
        if app != SENSOR_APP:
            raise SystemExit(f"keylog window is not frontmost ({app!r}); refusing to press keys")
        xy = predict(st, key)
        print(f"target {key!r}: predicted xy {np.round(xy, 1).tolist()}")
        travel(st, xy)
        r = descend_until_key()
        if r is None:
            return False
        hit, q_c, v, _ = r
        p = cmd_pose(q_c)
        st["obs"].append({"key": hit, "xy": np.round(p[:2], 1).tolist(), "h": round(float(p[2]), 1)})
        st["keys"][hit] = {"q_contact": np.round(q_c, 1).tolist(), "v": np.round(v, 3).tolist()}
        save(st)
        if hit == key:
            print(f"calibrated {key!r}")
            return True
        print(f"hit {hit!r} instead of {key!r}; refitting")
    return False


LIMITS = {1: (8, 4087), 2: (2030, 4087), 3: (8, 4087), 4: (1718, 3782)}


def send_checked(goal, speed):
    """Absolute goals {id: ticks}, clipped to the EEPROM limits; raises if teleop rejected the line."""
    goal = {str(i): int(np.clip(round(v), *LIMITS[i])) for i, v in goal.items()}
    arm.send({"speed": int(speed)})
    line = arm.send({"joints": goal})
    time.sleep(0.12)
    last = arm.status().get("last") or {}
    if isinstance(last, dict) and last.get("rejected") == line:
        raise RuntimeError(f"teleop rejected {line}: {last.get('error')}")


RELEASE = 150  # ID2 ticks; gravity hysteresis on ID2 is ~60 ticks, so small lifts do not leave the key


def key_pose(q_c, up2):
    """Raise the shoulder where we are, set pan/elbow/wrist to the key pose, then lower ID2 to up2 from above."""
    tg, _ = ticks()
    # 250 ticks: a 60-tick lift let the elbow/wrist swing between key poses dip onto `b` (auto-repeat).
    send_checked({2: min(tg[1], up2) - 250}, 30)
    arm.wait_idle(8)
    send_checked({1: q_c[0], 3: q_c[2], 4: q_c[3]}, 20)
    arm.wait_idle(8)
    send_checked({2: up2}, 15)
    arm.wait_idle(5)
    time.sleep(0.4)


def cal2(key, start_above=70, max_ticks=140):
    """Shoulder-only contact search (same motion as tap): ID2 +1 tick per 50 ms until keylog sees key-down."""
    st = load()
    q_c = np.array(st["keys"][key]["q_contact"])
    up2 = q_c[1] - start_above
    key_pose(q_c, up2)
    offset = KEYLOG.stat().st_size
    arm.send({"speed": 4})
    for k in range(max_ticks):
        tgt = up2 + k
        arm.send({"joints": {"2": int(tgt)}})
        t_end = time.time() + 0.05
        while time.time() < t_end:
            evs, _ = new_events(offset)
            downs = [e for e in evs if e["type"] == "down"]
            if downs:
                send_checked({2: tgt - RELEASE}, 80)
                hit = key_of(downs[0])
                print(f"cal2 {key!r}: hit {hit!r} at ID2 target {int(tgt)} ({k} ticks below start)")
                arm.wait_idle(3)
                time.sleep(0.5)
                evs, _ = new_events(offset)
                print("   events:", [(e["type"][0], key_of(e), round(e["t"] - downs[0]["t"], 3)) for e in evs])
                if hit == key:
                    st["keys"][key]["c2"] = int(tgt)
                    save(st)
                return hit
            time.sleep(0.004)
    send_checked({2: up2 - RELEASE}, 30)
    print(f"cal2 {key!r}: no key within {max_ticks} ticks")
    return None


def press_live(key, window="hello.py", app="Code", above=30, max_past=70, step=3, near=25):
    """Same slow shoulder ramp as cal2, but the contact sensor is the target window's pixels (screensense):
    release the moment the window changes. Returns the ID2 target at detection, or None."""
    from screensense import ScreenSensor
    st = load()
    k = st["keys"][key]
    q_c, c2 = np.array(k["q_contact"]), k["c2"]
    key_pose(q_c, c2 - above)
    if frontmost() != app:
        raise SystemExit(f"{app} is not frontmost; not pressing {key!r}")
    sensor = ScreenSensor(window, owner=app)
    noise, thr, nb = sensor.calibrate_noise()
    if noise > 200:  # the window is already changing: something is pressing a key
        send_checked({2: c2 - above - 250}, 80)
        raise SystemExit(f"window changing before the press (noise {noise} px): lifted, aborting")
    if noise > 0:
        print(f"   sensor noise {noise} px (threshold {thr})")
    arm.send({"speed": 12})
    # 1 tick per 50 ms down to c2 - near, then `step` ticks per 50 ms through the actuation point
    # (crossing it slowly made r/o/b chatter: down/up/down within 60 ms).
    plan = list(range(c2 - above, c2 - near)) + list(range(c2 - near, c2 + max_past, step))
    for tgt in plan:
        arm.send({"joints": {"2": int(tgt)}})
        t_end = time.time() + 0.05
        while time.time() < t_end:
            if sensor.changed():
                send_checked({2: tgt - RELEASE}, 80)
                print(f"pressed {key!r}: screen changed at ID2 {int(tgt)} (c2 {c2}, {int(tgt - c2):+d})")
                arm.wait_idle(3)
                return int(tgt)
    send_checked({2: c2 - RELEASE}, 30)
    print(f"no screen change for {key!r} down to ID2 {c2 + max_past}")
    return None


def tap(key, above=30, depth=20, hold=0.06, press_speed=40):
    """Settle `above` ticks over the shoulder contact c2 (from above), press to c2 + depth, release by RELEASE."""
    st = load()
    k = st["keys"][key]
    q_c, c2 = np.array(k["q_contact"]), k["c2"]
    key_pose(q_c, c2 - above)
    send_checked({2: c2 + depth}, press_speed)
    time.sleep(hold)
    send_checked({2: c2 - RELEASE}, 80)
    arm.wait_idle(3)
    print(f"tapped {key!r} (ID2 {c2 - above} -> {c2 + depth} -> {c2 - RELEASE})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    c = sub.add_parser("calibrate")
    c.add_argument("key")
    t = sub.add_parser("tap")
    t.add_argument("key")
    t.add_argument("--above", type=int, default=30)
    t.add_argument("--depth", type=int, default=20)
    t.add_argument("--hold", type=float, default=0.06)
    ty = sub.add_parser("type")
    ty.add_argument("text")
    ty.add_argument("--app", default=None, help="frontmost app required before each key (e.g. Code)")
    pl = sub.add_parser("press")
    pl.add_argument("key")
    pl.add_argument("--window", default="hello.py")
    pl.add_argument("--app", default="Code")
    c2 = sub.add_parser("cal2")
    c2.add_argument("key")
    l = sub.add_parser("lift")
    l.add_argument("--mm", type=float, default=40)
    for s in (c, t, l, c2, ty, pl):
        s.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "show":
        st = load()
        A, b = fit(st["obs"])
        print("obs", st["obs"])
        print("calibrated", list(st["keys"]))
        print("grid map A", np.round(A, 2).tolist(), "b", np.round(b, 1).tolist())
        for k in "irobt ":
            print(repr(k), np.round(A @ np.array(LAYOUT[k]) + b, 1).tolist())
        return
    if not a.yes:
        print("dry run: add --yes")
        return
    age = time.time() - arm.status()["t"]
    if age > 1.0:
        raise SystemExit(f"teleop status is {age:.0f} s old: controller not running (serial drop?)")
    arm.enable()
    if a.cmd == "calibrate":
        calibrate(a.key)
    elif a.cmd == "tap":
        tap(a.key, a.above, a.depth, a.hold)
    elif a.cmd == "type":
        for ch in a.text:
            if a.app and frontmost() != a.app:
                raise SystemExit(f"{a.app} is not frontmost; stopping before {ch!r}")
            tap(ch)
    elif a.cmd == "press":
        if press_live(a.key, a.window, a.app) is None:
            raise SystemExit(2)
    elif a.cmd == "cal2":
        if frontmost() != SENSOR_APP:
            raise SystemExit("keylog page is not frontmost; refusing to press keys")
        cal2(a.key)
    elif a.cmd == "lift":
        lift(a.mm)


if __name__ == "__main__":
    main()
