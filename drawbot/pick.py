"""Marker picking, field-proven 2026-09-26 (3 successful picks: pink on the table, orange from the case edge, blue on
paper). Model-only: the depth camera locates the marker by SHAPE, the kinematic model places the jaws.

Pipeline:
1. find_marker(color): dark-robust colour blob (brightness normalised to mean 90 before HSV) gives a rough spot;
   locate_shape() refines it from depth alone. Colour is unreliable in a dark room, shape is not.
2. locate_shape(guess): plane points 6..28 mm above the table within `radius` of the guess, largest connected
   cluster on a 3 mm grid, PCA -> center, axis, length, width, top height.
3. plan_pick(center, axis): the model TCP (gripper_frame_link) is the FIXED-jaw fingertip, not the grasp center.
   C_G is the in-gripper direction from the fixed jaw toward the moving jaw. It is anchored by one wrist-camera
   roll sweep (config `pick_anchor`: roll 2330 closed across a marker with axis (-0.48, 0.88) at tilt 20 over
   (-210, 143)), because the roll zero of the model is not calibrated. best_roll() searches roll 1660..4080 so that
   R(q, roll) C_G is perpendicular to the marker axis in the plane. TCP target = center - offset * u
   (u = in-plane R C_G, offset 25 mm) puts the marker between the open fingers. Jaws tilted 20 deg outward,
   vertical descent 60 -> 30 -> 18 -> 9 mm, close to ~730 (~90 ticks past a barrel stall of ~822-827), lift.
   Do NOT ease the grip afterwards: easing to 60 past the stall let two markers slip out while lifting.
4. execute_pick(plan): joint-position feedback per waypoint, then verify_grasp() from the gripper stall position.
Nothing here moves the arm unless drawbot.arm.enable() was called (CLI: --yes).
"""
import time

import numpy as np

from . import arm, config
from .kinmodel import WIN, model as _model

PICK_ROLL = (1660, 4080)     # roll search window (field): cable load rises below ~1600
BANDS = {"blue": ((95, 128), 60), "yellow": ((18, 34), 90), "green": ((35, 80), 90), "orange": ((172, 12), 120),
         "pink": ((140, 171), 70)}          # hue (OpenCV, lo > hi wraps), min sat; after brightness normalisation
FIND_REGION = ((-200, 150), (-220, 200))
PICK_DEFAULTS = dict(
    pick_anchor=dict(p=[-210.0, 143.0, 70.0], tilt=20.0, roll=2330, close_dir=[0.88, 0.48]),
    pick_offset_mm=25.0, pick_tilt=20.0, pick_hover=60.0, pick_descent=[30.0, 18.0], pick_grasp_h=9.0,
    pick_lift=60.0, pick_open=1000, pick_close=730, grasp_min_gap=30, grasp_max_pos=950,
)


def cfg():
    c = config()
    return {k: c.get(k, v) for k, v in PICK_DEFAULTS.items()}


def _m(m):
    return m if m is not None else _model()


def unit2(v):
    v = np.asarray(v, float)[:2]
    return v / max(np.linalg.norm(v), 1e-9)


# --- sensing ---------------------------------------------------------------------------------------------------
def _largest_cluster(xy, cell=3.0):
    """Indices of the largest 8-connected cluster of points binned on a `cell` mm grid."""
    g = np.floor(xy / cell).astype(int)
    cells = {}
    for i, k in enumerate(map(tuple, g)):
        cells.setdefault(k, []).append(i)
    seen, best = set(), []
    for k in cells:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            a = stack.pop()
            comp += cells[a]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = (a[0] + dx, a[1] + dy)
                    if b in cells and b not in seen:
                        seen.add(b)
                        stack.append(b)
        if len(comp) > len(best):
            best = comp
    return np.asarray(best, int)


def locate_points(Q, guess, base=None, radius=60.0, hmin=6.0, hmax=28.0, exclude_above=40.0, cell=3.0, min_pts=30,
                  axis=None, band=8.0):
    """Marker lying near `guess` (plane xy) from plane points Q (N, 3). Returns dict(center, axis, length, width,
    top, n, r, near_end, far_end, tall_nearby) or None (< min_pts points in the height band).
    With an `axis` prior the search region is a strip: |along| < radius and |lateral| < band around the guess
    (captures the whole length without the neighbours packed beside it)."""
    Q = np.asarray(Q, float)
    rel = Q[:, :2] - np.asarray(guess, float)[:2]
    if axis is None:
        near = np.linalg.norm(rel, axis=1) < radius
    else:
        a = unit2(axis)
        near = (np.abs(rel @ a) < radius) & (np.abs(rel @ np.array([-a[1], a[0]])) < band)
    pts = Q[near & (Q[:, 2] > hmin) & (Q[:, 2] < hmax)]
    tall = int((near & (Q[:, 2] >= exclude_above)).sum())
    if len(pts) < min_pts:
        return None
    q = pts[_largest_cluster(pts[:, :2], cell)]
    ctr = q[:, :2].mean(0)
    ax = np.linalg.svd(q[:, :2] - ctr, full_matrices=False)[2][0]
    perp = np.array([-ax[1], ax[0]])
    proj, side = (q[:, :2] - ctr) @ ax, (q[:, :2] - ctr) @ perp
    lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
    slo, shi = np.percentile(side, 2), np.percentile(side, 98)
    # center of the 2..98 % extent, not the mean: the camera sees more of the near end and of one side
    ctr = ctr + ax * (lo + hi) / 2 + perp * (slo + shi) / 2
    L = float(hi - lo)
    out = dict(center=ctr, axis=ax, length=L, width=float(shi - slo),
               top=float(np.percentile(q[:, 2], 95)), n=int(len(q)), tall_nearby=tall)
    ends = [ctr + ax * L / 2, ctr - ax * L / 2]
    if base is not None:
        base = np.asarray(base, float)[:2]
        ends.sort(key=lambda e: np.linalg.norm(e - base))
        out["r"] = float(np.linalg.norm(ctr - base))
    out["near_end"], out["far_end"] = ends
    return out


def plane_cloud(fr, frame):
    from .rgbd import backproject
    _, d, K = frame
    v, u = np.nonzero(d > 0)
    return fr.plane(backproject(u, v, d[v, u], K))


def locate_shape(guess, fr=None, frame=None, avg=7, **kw):
    """Depth-only marker pose near a rough plane-xy guess (Gemini median of `avg` frames)."""
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    if frame is None:
        from .rgbd import grab
        frame = grab(avg=avg)
    return locate_points(plane_cloud(fr, frame), guess, fr.BASE, **kw)


def normalize_brightness(img, mean=90.0):
    """Scale BGR so the mean is `mean`: makes hue/sat/value thresholds work in a dark room."""
    g = img.astype(np.float32)
    return np.clip(g / max(float(g.mean()), 1.0) * mean, 0, 255).astype(np.uint8)


def color_candidates(color, fr, frame, region=FIND_REGION, min_area=40, h_band=(3.0, 60.0)):
    """Blobs of `color` after brightness normalisation, largest first: [dict(c=plane xyh median, axis2, area)]."""
    import cv2
    from .rgbd import backproject
    img, d, K = frame
    hsv = cv2.cvtColor(normalize_brightness(img), cv2.COLOR_BGR2HSV)
    (h0, h1), smin = BANDS[color]
    hh = hsv[..., 0]
    band = ((hh >= h0) & (hh <= h1)) if h0 <= h1 else ((hh >= h0) | (hh <= h1))
    m = (band & (hsv[..., 1] > smin) & (hsv[..., 2] > 60) & (d > 250) & (d < 900)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    (x0, x1), (y0, y1) = region
    for k in 1 + np.argsort(-st[1:, cv2.CC_STAT_AREA])[:6]:
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        v, u = np.nonzero(lab == k)
        q = fr.plane(backproject(u, v, d[v, u], K))
        xyh = np.median(q, 0)
        if x0 < xyh[0] < x1 and y0 < xyh[1] < y1 and h_band[0] < xyh[2] < h_band[1]:
            ax = np.linalg.svd(q[:, :2] - q[:, :2].mean(0), full_matrices=False)[2][0]
            out.append(dict(c=xyh, axis2=ax, area=int(st[k, cv2.CC_STAT_AREA])))
    return out


def find_marker(color, fr=None, frame=None, avg=5, region=FIND_REGION, shape_radius=35.0):
    """Dark-robust colour find + shape refinement (radius-35 disc around the blob, then a +-8 mm strip along
    that axis). Returns dict(color, blob, shape, pose) or None.
    pose = what plan_pick needs (center, axis, top): from the shape when found, else from the colour blob."""
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    if frame is None:
        from .rgbd import grab
        frame = grab(avg=avg)
    cands = color_candidates(color, fr, frame, region)
    if not cands:
        return None
    blob = cands[0]
    Q = plane_cloud(fr, frame)
    shape = locate_points(Q, blob["c"][:2], fr.BASE, radius=shape_radius)
    if shape is not None:   # second pass: strip along the first axis, so a clipped end does not bias the center
        full = locate_points(Q, shape["center"], fr.BASE, radius=90.0, axis=shape["axis"])
        if full is not None and full["length"] >= shape["length"]:
            shape = full
    if shape is not None:
        pose = dict(center=shape["center"], axis=shape["axis"], top=shape["top"], source="shape")
    else:
        pose = dict(center=blob["c"][:2], axis=blob["axis2"], top=float(blob["c"][2]) + 5.0, source="color")
    return dict(color=color, blob=blob, shape=shape, pose=pose)


# --- roll / offset from the anchored closing direction -----------------------------------------------------------
def axis_for(base, p, tilt):
    """Jaw axis tilted `tilt` deg outward (tips away from BASE) from straight down."""
    rad = unit2(np.asarray(p, float)[:2] - np.asarray(base, float)[:2])
    t = np.radians(tilt)
    return np.r_[rad * np.sin(t), -np.cos(t)]


def close_dir_gripper(m=None, anchor=None):
    """C_G: gripper-frame unit vector from the fixed jaw toward the moving jaw, from the roll-sweep anchor."""
    m = _m(m)
    a = anchor or cfg()["pick_anchor"]
    q = m.ik(a["p"], axis=axis_for(m.base, a["p"], a["tilt"]), roll=a["roll"]).q
    R = m.fk(list(q) + [a["roll"]])[2]
    c = R.T @ np.r_[np.asarray(a["close_dir"], float)[:2], 0.0]
    return c / np.linalg.norm(c)


def in_plane_dir(m, q4, roll, C_G):
    return unit2(m.fk(list(q4) + [roll])[2] @ C_G)


def best_roll(p, tilt, marker_axis, C_G, m=None, rolls=range(PICK_ROLL[0], PICK_ROLL[1], 20)):
    """Roll (ID5) whose in-plane R(q, roll) C_G is most perpendicular to the marker axis at TCP p.
    Returns (roll, error deg, q4 at that roll)."""
    m = _m(m)
    ma = unit2(marker_axis)
    axis = axis_for(m.base, p, tilt)
    best, seed = None, None
    for roll in rolls:
        r = m.ik(list(p), axis=axis, roll=roll, seed=seed)
        seed = r.q
        err = abs(float(in_plane_dir(m, r.q, roll, C_G) @ ma))
        if best is None or err < best[0]:
            best = (err, int(roll), r.q)
    return best[1], float(np.degrees(np.arcsin(min(1.0, best[0])))), best[2]


def in_window(joints5):
    return all(WIN[i][0] <= joints5[i] <= WIN[i][1] for i in range(1, 5)) and \
        PICK_ROLL[0] - 10 <= joints5[5] <= PICK_ROLL[1] + 10


def plan_pick(center, axis, m=None, C_G=None, top=None, tilt=None, offset=None, grasp_h=None, max_res=2.5,
              lift_max_res=10.0, max_r=310.0):
    """Plan only: joint waypoints hover -> descent -> grasp, then lift, for a marker at plane xy `center` with
    in-plane `axis`. Returns dict(ok, roll, roll_err_deg, u, tcp_xy, waypoints=[dict(name, xyh, joints, res)],
    lift, gripper, warnings[, reason])."""
    m = _m(m)
    c = cfg()
    C_G = close_dir_gripper(m) if C_G is None else np.asarray(C_G, float)
    tilt = c["pick_tilt"] if tilt is None else tilt
    off = c["pick_offset_mm"] if offset is None else offset
    grasp_h = c["pick_grasp_h"] if grasp_h is None else grasp_h
    ctr = np.asarray(center, float)[:2]
    r = float(np.linalg.norm(ctr - m.base))
    warnings = []
    base = dict(center=ctr.round(1).tolist(), axis=unit2(axis).round(3).tolist(), r=round(r, 1), tilt=tilt,
                offset=off, grasp_h=grasp_h)
    if r > max_r:
        return dict(ok=False, reason=f"r {r:.0f} mm from BASE > {max_r:.0f}: out of reach (try slide)", **base)
    if top is not None and top > 25:
        warnings.append(f"marker top at h {top:.0f} mm (in the case?): grasp_h {grasp_h} may be too low")
    hov = np.r_[ctr, c["pick_hover"]]
    roll, _, q0 = best_roll(hov, tilt, axis, C_G, m)
    u = in_plane_dir(m, q0, roll, C_G)
    p = hov - np.r_[off * u, 0.0]                 # fixed-jaw tip offset so the marker sits between the fingers
    roll, err, _ = best_roll(p, tilt, axis, C_G, m)
    heights = [c["pick_hover"]] + list(c["pick_descent"]) + [grasp_h]
    names = ["hover"] + [f"down{int(h)}" for h in c["pick_descent"]] + ["grasp"]
    wps, seed = [], None
    for name, h, lim in list(zip(names, heights, [max_res] * len(heights))) + [("lift", c["pick_lift"], lift_max_res)]:
        x = np.r_[p[:2], h]
        s = m.ik(list(x), axis=axis_for(m.base, x, tilt), roll=roll, seed=seed)
        seed = s.q
        res = m.residual(x, s.q, roll)
        j = {**{i: int(s.q[i - 1]) for i in range(1, 5)}, 5: int(roll)}
        wps.append(dict(name=name, xyh=x.round(1).tolist(), joints=j, res=round(res, 2), ok=bool(res <= lim)))
    u_grasp = in_plane_dir(m, [wps[-2]["joints"][i] for i in range(1, 5)], roll, C_G)   # at the grasp pose
    plan = dict(ok=True, roll=roll, roll_err_deg=round(err, 1), u=u_grasp.round(3).tolist(), tcp_xy=p[:2].round(1).tolist(),
                waypoints=wps[:-1], lift=wps[-1], gripper=dict(open=c["pick_open"], close=c["pick_close"]),
                warnings=warnings, **base)
    bad = [w["name"] for w in wps if not w["ok"]]
    if bad:
        plan.update(ok=False, reason=f"IK miss at {bad} (res {[w['res'] for w in wps if not w['ok']]} mm)")
    elif not all(in_window(w["joints"]) for w in wps):
        plan.update(ok=False, reason="joint window violated")
    elif err > 10:
        plan.update(ok=False, reason=f"no roll closes across the marker (best {err:.0f} deg off)")
    return plan


# --- motion ----------------------------------------------------------------------------------------------------
def go_joints(want, move_fn=None, joints_fn=None, rounds=3, tol=5, gain=0.9, speed=3):
    """Command joints {id: ticks}, then correct sag/backlash from the measured positions (field pick2.go)."""
    move_fn = move_fn or arm.move
    joints_fn = joints_fn or arm.joints
    want = {int(k): int(v) for k, v in want.items()}
    cmd = dict(want)
    for _ in range(rounds):
        move_fn(cmd, speed=speed, settle=0.4)
        _, pos, _ = joints_fn()
        err = {i: want[i] - pos[i] for i in want}
        if max(abs(v) for v in err.values()) <= tol:
            break
        cmd = {i: int(np.clip(cmd[i] + gain * err[i], *arm.HARD[i])) for i in want}
    _, pos, _ = joints_fn()
    return {i: pos[i] for i in want}


def verify_grasp(close=None, joints_fn=None, min_gap=None, max_pos=None):
    """Holding a marker = the gripper stalled on the barrel: measured pos at least `min_gap` ticks short of the
    close command (empty jaws reach the command) and below `max_pos` (not left open)."""
    from .manip import load_mag
    c = cfg()
    close = c["pick_close"] if close is None else close
    min_gap = c["grasp_min_gap"] if min_gap is None else min_gap
    max_pos = c["grasp_max_pos"] if max_pos is None else max_pos
    _, pos, mot = (joints_fn or arm.joints)()
    g = int(pos[6])
    load = load_mag(mot["6"]["load"]) if "6" in mot else None
    held = g - close >= min_gap and g <= max_pos
    reason = "stalled on the barrel" if held else ("closed to the command: empty" if g - close < min_gap
                                                   else "jaws still open")
    return dict(held=bool(held), pos=g, gap=g - close, load=load, reason=reason)


def execute_pick(plan, move_fn=None, joints_fn=None, send_fn=None, sleep_fn=None, out=print):
    """Run a plan_pick plan: open, waypoints with joint feedback, close (no easing), verify, lift, verify.
    Needs arm.enable() when using the real arm."""
    if not plan.get("ok"):
        raise RuntimeError(f"plan not ok: {plan.get('reason')}")
    real = send_fn is None
    send_fn = send_fn or arm.send
    sleep_fn = sleep_fn or (time.sleep if real else (lambda s: None))
    kw = dict(move_fn=move_fn, joints_fn=joints_fn)

    def grip(ticks):
        arm.check_goal({6: ticks})
        send_fn({"gripper": int(ticks)})
        if real:
            arm.wait_idle(8)

    grip(plan["gripper"]["open"])
    for w in plan["waypoints"]:
        got = go_joints(w["joints"], **kw)
        out(f"   {w['name']:7s} h {w['xyh'][2]:5.1f} want {list(w['joints'].values())} got {list(got.values())}")
    grip(plan["gripper"]["close"])
    sleep_fn(0.8)
    g0 = verify_grasp(plan["gripper"]["close"], joints_fn)
    out(f"   gripper {g0['pos']} (gap {g0['gap']}, load {g0['load']}): {g0['reason']}")
    go_joints(plan["lift"]["joints"], **kw)
    g1 = verify_grasp(plan["gripper"]["close"], joints_fn)
    out(f"   after lift: gripper {g1['pos']}: {g1['reason']}")
    return dict(grasp=g0, after_lift=g1, held=g1["held"])
