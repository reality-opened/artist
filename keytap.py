"""Press laptop keys with the SO101 fixed-jaw tip (jaws down, gripper open so the moving jaw sits higher).

Cartesian steps go through the calibrated kinematic model's Jacobian (drawbot/kinmodel.py) and are sent as
teleop "delta" commands, so they stack on the current targets and keep whatever sag offset is in place.
  python keytap.py status
  python keytap.py move DX DY DH       (plane mm, keeps jaw axis)        needs --yes
  python keytap.py probe [--step 1.5] [--max 25]  lower until joints stop following   needs --yes
  python keytap.py tap DEPTH [--hold 0.12]   quick down DEPTH mm then back up         needs --yes
"""
import argparse
import json
import time

import numpy as np

from drawbot import arm
from drawbot.kinmodel import model

TCP = None  # fixed-jaw tip


def ticks5():
    tg, pos, _ = arm.joints()
    return np.array([tg[i] for i in range(1, 6)], float), np.array([pos[i] for i in range(1, 6)], float)


def jac(q5):
    """d[x, y, h, ax, ay, az]/d ticks for ID1..ID4 (central differences)."""
    m = model()
    J = np.zeros((6, 4))
    for j in range(4):
        dq = np.zeros(5)
        dq[j] = 2
        p1, a1, _ = m.fk(q5 + dq, TCP)
        p0, a0, _ = m.fk(q5 - dq, TCP)
        J[:3, j] = (p1 - p0) / 4
        J[3:, j] = (a1 - a0) / 4
    return J


def solve(q5, dxyz, w_axis=80.0):
    J = jac(q5)
    W = np.diag([1, 1, 1, w_axis, w_axis, w_axis])
    rhs = np.r_[dxyz, 0, 0, 0]
    dq, *_ = np.linalg.lstsq(W @ J, W @ rhs, rcond=None)
    return dq


def delta_cmd(dq):
    return {"delta": {str(i + 1): int(round(v)) for i, v in enumerate(dq) if round(v) != 0}}


def model_pose():
    tg, pos = ticks5()
    p, a, _ = model().fk(pos, TCP)
    return tg, pos, p, a


def show():
    tg, pos, p, a = model_pose()
    _, _, m = arm.joints()
    print("targets", tg.astype(int).tolist(), "pos", pos.astype(int).tolist())
    print("tip (measured joints)", np.round(p, 1).tolist(), "axis", np.round(a, 2).tolist())
    print("loads", {i: m[str(i)]["load"] for i in range(1, 7)})


def move(d):
    tg, pos = ticks5()
    dq = solve(pos, np.asarray(d, float))
    arm.send({"speed": 6})
    arm.send(delta_cmd(dq))
    arm.wait_idle(10)
    time.sleep(0.4)


def probe(step, max_mm):
    """Lower in `step` mm increments; contact = measured descent (model, measured joints) < 35 % of commanded."""
    tg, pos = ticks5()
    h0 = model().fk(pos, TCP)[0][2]
    went = 0.0
    lag_steps = 0
    arm.send({"speed": 4})
    while went < max_mm:
        _, pos_a = ticks5()
        ha = model().fk(pos_a, TCP)[0][2]
        dq = solve(pos_a, [0, 0, -step])
        arm.send(delta_cmd(dq))
        arm.wait_idle(5)
        time.sleep(0.5)
        _, pos_b = ticks5()
        hb = model().fk(pos_b, TCP)[0][2]
        went += step
        moved = ha - hb
        print(f"cmd -{went:.1f}  h {hb:.1f}  moved {moved:.2f}  pos {pos_b.astype(int).tolist()}", flush=True)
        if moved < 0.35 * step:
            lag_steps += 1
            if lag_steps >= 2:
                print(f"contact near h {hb:.1f} (started {h0:.1f})")
                return hb
        else:
            lag_steps = 0
    print("no contact within", max_mm)
    return None


def tap(depth, hold):
    _, pos = ticks5()
    dq = solve(pos, [0, 0, -depth])
    down, up = delta_cmd(dq), delta_cmd(-dq)
    arm.send({"speed": 80})
    t0 = time.time()
    arm.send(down)
    time.sleep(hold)
    arm.send(up)
    arm.wait_idle(3)
    arm.send({"speed": 6})
    print("tap", json.dumps(down), f"held {hold}s, done in {time.time() - t0:.2f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("move")
    for k in ("dx", "dy", "dh"):
        m.add_argument(k, type=float)
    p = sub.add_parser("probe")
    p.add_argument("--step", type=float, default=1.5)
    p.add_argument("--max", type=float, default=25)
    t = sub.add_parser("tap")
    t.add_argument("depth", type=float)
    t.add_argument("--hold", type=float, default=0.12)
    for s in (m, p, t):
        s.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        return show()
    if not a.yes:
        print("dry run: add --yes")
        return
    arm.enable()
    if a.cmd == "move":
        move([a.dx, a.dy, a.dh])
    elif a.cmd == "probe":
        probe(a.step, a.max)
    elif a.cmd == "tap":
        tap(a.depth, a.hold)
    show()


if __name__ == "__main__":
    main()
