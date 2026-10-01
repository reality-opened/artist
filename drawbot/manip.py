"""Manipulation on the depth-calibrated model (kinmodel): hover, poke/push, height map, wrist-roll flip, pick plan.

Field notes (2026-09-26):
- Wrist roll ID5 turns ~190 deg: 3746 -> ~1600, with cable load rising toward the end. flip_roll steps 100 ticks
  and backs off on load > 300 or lag > 40 (the field run tripped at 1546 and backed off to 1696). A 180-deg roll
  (3746 <-> 1698) flips a marker held across the jaws, so its tip direction at pickup does not matter.
- Reach at table height from BASE: jaws down to r ~250 mm, jaws tilted 35-50 deg outward to r ~300, nothing
  beyond ~300-310.
- Gripper: command only ~35-40 ticks past the stall point on the barrel (stall + heating otherwise).
- Pushing packed markers sideways moves the whole light plastic case; dragging on a marker top rolls the markers
  apart instead of pulling the case.
"""
import numpy as np

from . import arm, config
from .kinmodel import DOWN, ROLL_HOME, model as _model, unit

ROLL_WIN = (1600, 4090)      # usable ID5 range (cable load rises below ~1600)
FLIP = 2048                  # 180 deg of wrist roll
GRIP_DEFAULTS = dict(gripper_open=1000, gripper_stall=794, gripper_margin=38)


def _m(model):
    return model if model is not None else _model()


def hover(x, y, h=70, axis=DOWN, model=None, **kw):
    return _m(model).goto([x, y, h], axis=list(axis), rounds=2, verbose=False, **kw)


def poke(p0, p1, h, clear=55, steps=3, axis=DOWN, model=None, joints_fn=None, **kw):
    """Closed fingertip: descend at p0 to height h, slide to p1 in `steps`, lift to `clear`. Returns loads seen."""
    m = _m(model)
    joints_fn = joints_fn or arm.joints
    kw = dict(kw, joints_fn=joints_fn)
    hover(p0[0], p0[1], clear, axis, m, **kw)
    m.goto([p0[0], p0[1], h], axis=list(axis), rounds=2, verbose=False, max_res=6, **kw)
    loads = []
    for k in range(1, steps + 1):
        p = np.asarray(p0, float) + (np.asarray(p1, float) - np.asarray(p0, float)) * k / steps
        m.goto([p[0], p[1], h], axis=list(axis), rounds=1, verbose=False, max_res=6, **kw)
        t, pos, mot = joints_fn()
        loads.append([mot[str(i)]["load"] for i in range(1, 5)])
    hover(p1[0], p1[1], clear, axis, m, **kw)
    return loads


# --- height map ----------------------------------------------------------------------------------------------
def heightmap_points(Q, x0, x1, y0, y1, cell=6, pct=90, hmax=120, min_pts=3):
    """Per-cell `pct` percentile of h for plane points Q (N, 3). Returns xs, ys (cell corners), grid (NaN = empty)."""
    Q = np.asarray(Q, float)
    Q = Q[(Q[:, 0] >= x0) & (Q[:, 0] < x1) & (Q[:, 1] >= y0) & (Q[:, 1] < y1) & (Q[:, 2] < hmax)]
    xs, ys = np.arange(x0, x1, cell), np.arange(y0, y1, cell)
    g = np.full((len(ys), len(xs)), np.nan)
    ix = ((Q[:, 0] - x0) // cell).astype(int)
    iy = ((Q[:, 1] - y0) // cell).astype(int)
    ok = (ix < len(xs)) & (iy < len(ys))
    key = iy[ok] * len(xs) + ix[ok]
    h = Q[ok, 2]
    order = np.argsort(key, kind="stable")
    key, h = key[order], h[order]
    cells, start, count = np.unique(key, return_index=True, return_counts=True)
    for c, s, n in zip(cells, start, count):
        if n >= min_pts:
            g.flat[c] = np.percentile(h[s:s + n], pct)
    return xs, ys, g


def heightmap(x0, x1, y0, y1, cell=6, pct=90, avg=7, hmax=120, fr=None, frame=None):
    """Height map of a plane-frame region from Gemini depth (median of `avg` frames)."""
    from .rgbd import backproject
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    if frame is None:
        from .rgbd import grab
        frame = grab(avg=avg)
    _, d, K = frame
    v, u = np.nonzero(d > 0)
    return heightmap_points(fr.plane(backproject(u, v, d[v, u], K)), x0, x1, y0, y1, cell, pct, hmax)


def show_heightmap(xs, ys, g):
    rows = ["       " + "".join(f"{int(x):4d}" for x in xs)]
    for i, y in enumerate(ys):
        rows.append(f"{int(y):6d} " + "".join("   ." if np.isnan(v) else f"{v:4.0f}" for v in g[i]))
    return "\n".join(rows)


# --- wrist roll ----------------------------------------------------------------------------------------------
def load_mag(raw):
    """Feetech Present_Load (sign-magnitude, bit 10 = direction; teleop reports the raw register)."""
    raw = int(raw)
    return raw & 0x3FF if raw >= 0 else -raw


def flip_target(cur, win=ROLL_WIN):
    """The roll 180 deg away from `cur` that stays inside `win` (3746 -> 1698, 1698 -> 3746)."""
    for cand in (cur - FLIP, cur + FLIP):
        if win[0] <= cand <= win[1]:
            return int(cand)
    raise ValueError(f"no 180-deg flip from roll {cur} inside {win}")


def flip_roll(target, step=100, max_load=300, max_lag=40, backoff=150, speed=4, settle=0.3, win=ROLL_WIN,
              move_fn=None, joints_fn=None, out=print):
    """Turn ID5 to `target` in `step`-tick moves; on |load| > max_load or lag > max_lag, back off `backoff`
    ticks toward the start and stop. Returns dict(ok, roll (last commanded), reason, trace)."""
    move_fn = move_fn or arm.move
    joints_fn = joints_fn or arm.joints
    target = int(target)
    if not win[0] <= target <= win[1]:
        raise ValueError(f"roll target {target} outside {win}")
    t, p, m = joints_fn()
    start = cmd = int(t[5])
    sgn = 1 if target > cmd else -1
    trace = []
    while cmd != target:
        nxt = cmd + sgn * min(step, abs(target - cmd))
        move_fn({5: nxt}, speed=speed, settle=settle)
        t, p, m = joints_fn()
        load, lag = load_mag(m["5"]["load"]), abs(nxt - p[5])
        trace.append(dict(cmd=nxt, pos=p[5], load=load, lag=lag))
        out(f"   roll {nxt} pos {p[5]} load {load} lag {lag}")
        if load > max_load or lag > max_lag:
            back = nxt - sgn * backoff
            back = max(back, start) if sgn > 0 else min(back, start)
            move_fn({5: back}, speed=speed, settle=settle)
            reason = f"{'load' if load > max_load else 'lag'} guard at {nxt} (load {load}, lag {lag}); backed off to {back}"
            out("   " + reason)
            return dict(ok=False, roll=back, reason=reason, trace=trace)
        cmd = nxt
    return dict(ok=True, roll=cmd, reason="", trace=trace)


# --- gripper -------------------------------------------------------------------------------------------------
def grip_cfg():
    return {k: config().get(k, v) for k, v in GRIP_DEFAULTS.items()}


def grip_close_ticks(stall=None, margin=None):
    """Close command = stall point minus ~35-40 ticks (never squeeze to the end: the stalled servo heats)."""
    g = grip_cfg()
    return int((g["gripper_stall"] if stall is None else stall) - (g["gripper_margin"] if margin is None else margin))


def gripper(ticks, send_fn=None, wait=True):
    arm.check_goal({6: ticks})
    (send_fn or arm.send)({"gripper": int(ticks)})
    if wait and send_fn is None:
        arm.wait_idle()


# --- pick planning -------------------------------------------------------------------------------------------
def choose_rolls(m, q4, axis2, win=ROLL_WIN, roll_ref=ROLL_HOME, step=4, tol=0.05):
    """Rolls (ID5) that put the jaw closing direction perpendicular to the marker axis (plane xy) at arm pose q4.
    The solutions come in pairs 180 deg apart (the marker flips with the roll, so either grasps it); the in-window
    ones are returned nearest roll_ref first."""
    rolls = np.arange(win[0], win[1] + 1, step)
    T = np.column_stack([np.tile(np.asarray(q4, float), (len(rolls), 1)), rolls])
    _, _, R = m.fk_batch(T)
    cd = R @ m.chain.close_local
    cxy = cd[:, :2] / np.maximum(np.linalg.norm(cd[:, :2], axis=1, keepdims=True), 1e-9)
    cost = np.abs(cxy @ unit(axis2))
    good = np.nonzero(cost <= max(tol, cost.min() + 1e-9))[0]
    clusters = np.split(good, np.nonzero(np.diff(good) > 2)[0] + 1)
    best = [int(rolls[c[np.argmin(cost[c])]]) for c in clusters if len(c)]
    return sorted(best, key=lambda r: abs(r - roll_ref))


def choose_roll(m, q4, axis2, win=ROLL_WIN, roll_ref=ROLL_HOME, step=4, tol=0.05):
    return choose_rolls(m, q4, axis2, win, roll_ref, step, tol)[0]


def _plan_branch(m, target, axis, roll0, ax2, tool, w_axis, clear, lift, max_res, max_axis_deg, min_tcp_h):
    """One (axis, roll branch) candidate: (plan legs dict or None, tcp_h, roll, note)."""
    g = m.ik(target, axis, roll0, tool=tool, w_axis=w_axis)
    roll = roll0
    for _ in range(2):  # the roll depends on the pose and the pose (grasp offset) on the roll: stay on this branch
        roll = min(choose_rolls(m, g.q, ax2, roll_ref=roll), key=lambda r: abs(r - roll))
        g = m.ik(target, axis, roll, seed=g.q, tool=tool, w_axis=w_axis)
    note = f"roll {roll}: res {g.res:.1f} mm axis err {g.axis_err:.1f} deg"
    if not g.ok(max_res, max_axis_deg):
        return None, None, roll, note
    # Jaws down, the wrist limit (ID4 <= 3770) caps the height at ~60 mm: shorten approach/lift if needed.
    legs = {"approach": lambda d: target - axis * d, "grasp": None, "lift": lambda d: target + np.array([0, 0, d])}
    sol = {}
    for name, leg in legs.items():
        if leg is None:
            p, s = target, g
        else:
            for frac in (1.0, 0.7, 0.5):
                p = leg((clear if name == "approach" else lift) * frac)
                s = m.ik(p, axis, roll, seed=g.q, tool=tool, w_axis=w_axis)
                if s.ok(max_res, max_axis_deg):
                    break
        sol[name] = dict(xyh=p.round(1).tolist(), q=[int(v) for v in s.q], res=round(s.res, 2),
                         axis_err=round(s.axis_err, 1), ok=bool(s.ok(max_res, max_axis_deg)),
                         joints={**{i: int(s.q[i - 1]) for i in range(1, 5)}, 5: roll})
    if not all(v["ok"] for v in sol.values()):
        return None, None, roll, note + " (approach/lift unreachable)"
    tcp_h = float(m.fk(list(g.q) + [roll])[0][2])
    if tcp_h < min_tcp_h:
        note += f" (fixed-jaw tip at h {tcp_h:.1f} < {min_tcp_h})"
    return sol, tcp_h, roll, note


def pick_marker(pose, model=None, roll_ref=ROLL_HOME, clear=40.0, lift=40.0, grasp_dh=0.0, tilts=(35, 40, 45, 50),
                max_res=4.0, max_axis_deg=8.0, min_tcp_h=0.0, tool=None, base=None, w_axis=60.0, max_r=310.0):
    """Plan only (no motion): approach / grasp / lift joint targets for a marker lying at `pose` (scene.describe or
    track.describe: "c"/"xyh" center, "axis2"/"axis" direction). Grasp center = marker center + grasp_dh, placed
    with the model grasp offset. Jaws straight down if reachable, else tilted 35-50 deg outward (away from BASE).
    Both roll branches (180 deg apart) are tried; the one nearest roll_ref that keeps the fixed-jaw tip (TCP) at or
    above min_tcp_h wins. Nothing beyond max_r (field: ~300-310 mm) is attempted even if the model claims it.
    Returns dict(ok, mode, tilt_deg, axis, roll, approach, grasp, lift, gripper, ...) or dict(ok=False, reason)."""
    m = _m(model)
    c = np.asarray(pose.get("c", pose.get("xyh")), float)
    ax2 = pose.get("axis2")
    ax2 = unit(ax2 if ax2 is not None else np.asarray(pose["axis"], float)[:2])
    target = np.array([c[0], c[1], c[2] + grasp_dh])
    tool = m.grasp_tool if tool is None else tool
    base = m.base if base is None else np.asarray(base, float)
    rel = target[:2] - base
    r = float(np.hypot(*rel))
    radial = np.r_[rel / max(r, 1e-9), 0.0]
    cands = [("down", 0.0, np.array(DOWN))]
    cands += [("tilted", float(t), unit(radial * np.sin(np.radians(t)) + np.array(DOWN) * np.cos(np.radians(t))))
              for t in tilts]
    tried = []
    if r > max_r:
        return dict(ok=False, r=round(r, 1), reason=f"r {r:.0f} mm from BASE > {max_r:.0f}: out of reach", tried=tried)
    fallback = None
    for mode, tilt, axis in cands:
        g0 = m.ik(target, axis, roll_ref, tool=tool, w_axis=w_axis)
        for roll0 in choose_rolls(m, g0.q, ax2, roll_ref=roll_ref):
            sol, tcp_h, roll, note = _plan_branch(m, target, axis, roll0, ax2, tool, w_axis, clear, lift, max_res,
                                                  max_axis_deg, min_tcp_h)
            tried.append(f"{mode} {tilt:.0f} {note}")
            if sol is None:
                continue
            q5 = list(sol["grasp"]["q"]) + [roll]
            plan = dict(ok=tcp_h >= min_tcp_h, mode=mode, tilt_deg=tilt, axis=axis.round(3).tolist(), roll=roll,
                        r=round(r, 1), close_dir=m.close_dir(q5).round(2).tolist(), tcp_h=round(tcp_h, 1),
                        gripper=dict(open=grip_cfg()["gripper_open"], close=grip_close_ticks()),
                        tool_mm=(np.asarray(tool) * 1000).round(1).tolist(), tried=tried,
                        warnings=[] if tcp_h >= min_tcp_h else [f"fixed-jaw tip at h {tcp_h:.1f} mm (raise grasp_dh)"],
                        **sol)
            if plan["ok"]:
                return plan
            if fallback is None or tcp_h > fallback["tcp_h"]:
                fallback = plan
    if fallback is not None:
        return fallback
    return dict(ok=False, r=round(r, 1), reason=f"no reachable grasp at r {r:.0f} mm from BASE", tried=tried)


def execute_pick(plan, model=None, speed=5, move_fn=None, joints_fn=None, send_fn=None, out=print):
    """Run a pick_marker plan: open, roll (guarded), approach, grasp, close, lift. Needs arm.enable()."""
    if not plan.get("ok"):
        raise RuntimeError(f"plan not ok: {plan.get('reason') or plan.get('warnings')}")
    m = _m(model)
    kw = dict(move_fn=move_fn, joints_fn=joints_fn)
    tool = np.asarray(plan["tool_mm"], float) / 1000.0
    gripper(plan["gripper"]["open"], send_fn)
    fl = flip_roll(plan["roll"], out=out, **kw)
    if not fl["ok"]:
        raise RuntimeError(f"roll not reached: {fl['reason']}")
    got = {}
    for name, rounds, max_res in (("approach", 3, 4.0), ("grasp", 2, 6.0)):
        leg = plan[name]
        got[name] = m.goto(leg["xyh"], axis=plan["axis"], roll=plan["roll"], seed=leg["q"], tool=tool, speed=speed,
                           rounds=rounds, max_res=max_res, w_axis=60.0, out=out, **kw)[0]
    gripper(plan["gripper"]["close"], send_fn)
    leg = plan["lift"]
    got["lift"] = m.goto(leg["xyh"], axis=plan["axis"], roll=plan["roll"], seed=leg["q"], tool=tool, speed=speed,
                         w_axis=60.0, out=out, **kw)[0]
    return got


