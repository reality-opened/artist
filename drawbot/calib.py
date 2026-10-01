"""Kinematic calibration against the depth camera: collect samples, fit the URDF model, drop outliers, refit.

collect(): anchor poses x perturbations (known-safe set). At each pose the measured joint ticks and the tracked
orange marker (held in the gripper) are recorded. If the marker ends up lower than MIN_SAFE mm above the paper
the arm goes back to a safe home before the next pose. Output: $DRAWBOT_STATE/calib_samples.jsonl, one line per
sample: {"q": [ID1..ID5 measured pos], "target": [pan, ID2, ID3, ID4 commanded], "cam": [x, y, h] plane mm
(the key name is historical: it holds plane coordinates), "area", "px", "t", "cam_xyz": camera mm}.

fit(): Levenberg-Marquardt with a numeric Jacobian on 12 unknowns (see kinmodel: rotvec, t, offsets o2..o4, tool
point). Initial R, t from Kabsch on the zero-offset model. run(): 4 starts (zero offsets + 3 random offsets in
+-1.2 rad) for the given signs (default [1, 1, 1]; the optional sign search tries all 8), then drop samples with
error > 15 mm and refit from the previous solution. Field result: rms 4.2 mm on 52 of 53 samples.

Identifiability: with a single wrist roll (the field set: all 3746) the wrist-flex offset o4 and the tool point
trade off exactly (a rotation about the wrist-flex axis can be absorbed by moving the tool point), so the jaw axis
is not calibrated by the data; LM just stays near its start (o4 from 0). Samples at a second roll fix that. The
shoulder offset o2 is weakly separated from the base rotation (the pose set spans only ~+-25 deg of pan), so
compare fits by prediction error, not by parameters.
"""
import itertools
import json
import time

import numpy as np

from . import arm, path, save
from .kinmodel import MODEL, chain, mid_from_calibration, model_q, rodrigues, rotvec

SAMPLES = "calib_samples.jsonl"
MIN_SAFE = 60.0
OUTLIER_MM = 15.0
ANCHORS = [  # (pan, ID2, ID3, ID4 offset from level4)
    (1952, 3250, 1550, 0), (1952, 3400, 1550, 0), (1952, 3250, 1550, 400),
    (1952, 3100, 1650, 250), (1952, 2950, 1600, 700), (1952, 3250, 1750, -200),
]
PERT = [(0, 0, 0), (-220, 0, 0), (100, 0, 0), (0, -90, 0), (0, 90, 0), (0, 0, -100), (0, 0, 100),
        (-150, 60, -80), (100, -60, 80), (-300, 0, 0)]
HOME = {2: 3250, 3: 1550, 4: arm.level4(3250, 1550)}


def poses(start_anchor=0):
    """[(anchor index, {1: pan, 2: a2, 3: a3, 4: a4})] in collection order."""
    out = []
    for k, a in enumerate(ANCHORS[start_anchor:], start_anchor):
        for d in PERT:
            pan, a2, a3 = a[0] + d[0], a[1] + d[1], a[2] + d[2]
            a4 = int(np.clip(arm.level4(a2, a3) + a[3], 1760, 3760))
            out.append((k, {1: pan, 2: a2, 3: a3, 4: a4}))
    return out


def collect(fr=None, measure_fn=None, move_fn=None, joints_fn=None, start_anchor=0, name=SAMPLES, roll=None,
            out=print):
    """Run the pose set, appending samples to `name`. Returns the number of samples written.

    roll: first turn ID5 there (guarded, manip.flip_roll) from the safe home, e.g. 1698 for a second sample set
    that makes the wrist offset and the jaw axis observable."""
    move_fn = move_fn or arm.move
    joints_fn = joints_fn or arm.joints
    if roll is not None:
        from .manip import flip_roll
        move_fn(dict(HOME), speed=5)
        r = flip_roll(roll, move_fn=move_fn, joints_fn=joints_fn, out=out)
        if not r["ok"]:
            raise RuntimeError(f"roll {roll} not reached: {r['reason']}")
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    if measure_fn is None:
        from . import track
        measure_fn = lambda **kw: track.measure(fr, mode="orange", **kw)
    safe_home = lambda: move_fn(dict(HOME), speed=5)
    p_out = path(name)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    n, last = 0, None
    with open(p_out, "a") as f:
        for k, goal in poses(start_anchor):
            if last is not None and k != last:
                safe_home()
            last = k
            move_fn(goal, speed=6, settle=0.8)
            try:
                s = measure_fn(avg=5)
            except RuntimeError:
                out(f"  not visible {list(goal.values())}")
                continue
            t, p, m = joints_fn()
            q = [p[i] for i in range(1, 6)]
            xyh = np.asarray(s["xyh"], float)
            rec = dict(q=q, target=list(goal.values()), cam=xyh.tolist(), area=s.get("area"),
                       px=list(s["px"]) if s.get("px") is not None else None, t=time.time(),
                       cam_xyz=fr.to_cam(xyh).tolist())
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            out(f"{n:3d} q {q} xyh {xyh.round(1).tolist()} area {s.get('area')}")
            if xyh[2] < MIN_SAFE:
                out("  low -> home")
                safe_home()
        if last is not None:
            safe_home()
    return n


def load_samples(name=SAMPLES):
    return [json.loads(line) for line in open(path(name)) if line.strip()]


def _arrays(S):
    return np.array([s["q"] for s in S], float), np.array([s["cam"] for s in S], float)


def points(Q, signs, x, mid):
    """Tool points (N, 3) in base_link mm for ticks Q (N, 5) and parameters x."""
    T = chain().fk(model_q(Q, signs, x[6:9], mid))
    return (T[:, :3, :3] @ x[9:12] + T[:, :3, 3]) * 1000


def resid(x, Q, P, signs, mid):
    return (points(Q, signs, x, mid) @ rodrigues(x[:3]).T + x[3:6] - P).ravel()


def kabsch(A, B):
    """R, t minimizing |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def init_x(Q, P, signs, mid, offs=None):
    x = np.zeros(12)
    if offs is not None:
        x[6:9] = offs
    R, t = kabsch(points(Q, signs, x, mid), P)
    x[:3], x[3:6] = rotvec(R), t
    return x


def fit(S, signs=(1, 1, 1), x0=None, iters=200, mid=None):
    """LM fit. Returns (x, per-sample error mm)."""
    mid = mid_from_calibration() if mid is None else np.asarray(mid, float)
    signs = np.asarray(signs, float)
    Q, P = _arrays(S)
    x = init_x(Q, P, signs, mid) if x0 is None else np.asarray(x0, float).copy()
    steps = np.r_[np.full(9, 1e-5), np.full(3, 1e-6)]
    lam = 1e-2
    r = resid(x, Q, P, signs, mid)
    for _ in range(iters):
        J = np.zeros((len(r), 12))
        for i in range(12):
            dx = np.zeros(12)
            dx[i] = steps[i]
            J[:, i] = (resid(x + dx, Q, P, signs, mid) - resid(x - dx, Q, P, signs, mid)) / (2 * steps[i])
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
        xn = x + step
        rn = resid(xn, Q, P, signs, mid)
        if rn @ rn < r @ r:
            x, r, lam = xn, rn, lam * 0.3
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return x, np.sqrt((r.reshape(-1, 3) ** 2).sum(1))


def rms(e):
    return float(np.sqrt(np.mean(np.asarray(e) ** 2)))


def fit_best(S, signs=(1, 1, 1), sign_search=False, trials=4, mid=None, out=print):
    """Best of `trials` starts per sign pattern. Returns (rms, signs, x, errors)."""
    mid = mid_from_calibration() if mid is None else np.asarray(mid, float)
    Q, P = _arrays(S)
    patterns = itertools.product([1, -1], repeat=3) if sign_search else [tuple(signs)]
    best = None
    for sg in patterns:
        sg = np.array(sg, float)
        for trial in range(trials):
            x0 = None
            if trial:
                offs = np.random.default_rng(trial).uniform(-1.2, 1.2, 3)
                x0 = init_x(Q, P, sg, mid, offs)
            x, e = fit(S, sg, x0, mid=mid)
            if best is None or rms(e) < best[0]:
                best = (rms(e), sg, x, e)
        out(f"signs {sg.astype(int).tolist()} best so far rms {best[0]:.1f}")
    return best


def refit(S, x, signs, thresh=OUTLIER_MM, mid=None):
    """Drop samples whose error under x exceeds `thresh` mm and refit from x. Returns (x, errors, kept samples)."""
    mid = mid_from_calibration() if mid is None else np.asarray(mid, float)
    Q, P = _arrays(S)
    e = np.sqrt((resid(np.asarray(x, float), Q, P, np.asarray(signs, float), mid).reshape(-1, 3) ** 2).sum(1))
    kept = [s for s, ei in zip(S, e) if ei <= thresh]
    x2, e2 = fit(kept, signs, x, mid=mid)
    return x2, e2, kept


def run(S=None, signs=(1, 1, 1), sign_search=False, thresh=OUTLIER_MM, trials=4, mid=None, save_model=True,
        out=print):
    """Full fit: multi-start (optional sign search), outlier removal, refit; saves kin_model.json."""
    S = load_samples() if S is None else S
    mid = mid_from_calibration() if mid is None else np.asarray(mid, float)
    out(f"samples {len(S)}")
    if len({int(round(s["q"][4] / 50)) for s in S}) == 1:
        out("note: wrist roll was the same for every sample, so the wrist-flex offset o4 and the tool point are"
            " degenerate (only their combination is fitted) and the jaw axis direction is not calibrated."
            " Collect a second set at the flipped roll (calib-collect --roll 1698) to pin it.")
    r0, sg, x, e = fit_best(S, signs, sign_search, trials, mid, out)
    out(f"first fit rms {r0:.1f} mm, max {e.max():.1f}")
    x, e, kept = refit(S, x, sg, thresh, mid)
    res = dict(signs=sg.tolist(), x=x.tolist(), rms=rms(e), n=len(kept), dropped=len(S) - len(kept),
               median=float(np.median(e)), max=float(e.max()), mid=mid.tolist())
    out(f"signs {sg.astype(int).tolist()} rms {res['rms']:.1f} mm on {len(kept)}/{len(S)}  median {res['median']:.1f}"
        f"  max {res['max']:.1f}")
    out(f"offsets deg {np.degrees(x[6:9]).round(1).tolist()}  tool mm {(x[9:12] * 1000).round(1).tolist()}")
    if save_model:
        save(MODEL, res)
        out(f"saved {path(MODEL)}")
    return res
