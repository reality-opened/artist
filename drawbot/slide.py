"""Slide a lying marker ALONG its own axis toward the robot by pressing a closed fingertip on its top (field
drag_marker.py, 2026-09-26: pulled an out-of-reach marker from r ~339 to ~262 mm). Pushing sideways instead shoves
the whole (light) marker case.

The press point starts `inset` mm in from the marker end nearest BASE, `press_dh` mm below the marker top, and
steps `step` mm along the axis toward that end. Each point is solved for jaw tilts 0..65 deg outward (tilted jaws
are fine here: only the fingertip touches) and the tilt with the smallest joint change wins. Roll is fixed
(ID5 3746, commanded with the first waypoint). Nothing moves without drawbot.arm.enable() (CLI --yes).
"""
import numpy as np

from . import arm
from .kinmodel import ROLL_HOME, model as _model
from .pick import axis_for, go_joints, unit2

TILTS = (0, 20, 35, 45, 55, 65)
SLIDE_GRIPPER = 700      # jaws closed (nothing held): the fixed-jaw tip presses on the marker top


def plan_point(m, p, h, prev=None, roll=ROLL_HOME, tilts=TILTS, max_res=2.0):
    """(q4, tilt) at plane xy p, TCP height h: the reachable tilt closest in joints to `prev` (first by TILTS
    order when prev is None). Raises RuntimeError when no tilt reaches."""
    cands = []
    for tilt in tilts:
        x = [p[0], p[1], h]
        r = m.ik(x, axis=axis_for(m.base, x, tilt), roll=roll, seed=prev)
        if r.res < max_res:
            cands.append((0 if prev is None else float(np.abs(np.asarray(r.q) - np.asarray(prev)).sum()), tilt, r.q))
    if not cands:
        raise RuntimeError(f"unreachable {np.round(p, 1).tolist()} h {h:.0f}")
    cands.sort(key=lambda c: c[0])
    return cands[0][2], cands[0][1]


def plan_slide(marker, m=None, dist=80.0, step=10.0, inset=8.0, press_dh=7.0, clear=40.0, roll=ROLL_HOME, stop_r=240.0):
    """marker: dict(center, axis, length, top) (pick.locate_points / find_marker()["shape"]). Plan only.
    Steps stop once the marker center would be within `stop_r` of BASE (pickable; field 339 -> 262).
    Returns dict(ok, press_h, start, dir, waypoints=[dict(name, xyh, tilt, joints)], expected_center, r0, r1)."""
    m = _m(m)
    c = np.asarray(marker["center"], float)[:2]
    ax = unit2(marker["axis"])
    L = float(marker.get("length", 120.0))
    ends = sorted([c + ax * L / 2, c - ax * L / 2], key=lambda e: np.linalg.norm(e - m.base))
    near = ends[0]
    d = unit2(near - c)                         # along the marker axis, toward the robot
    start = near - d * inset
    press_h = float(marker["top"]) - press_dh
    steps = [s for s in np.arange(step, dist + 1e-6, step) if np.linalg.norm(c + d * (s - step) - m.base) > stop_r]
    if not steps:
        return dict(ok=False, reason=f"marker already within {stop_r:.0f} mm of BASE: pick it", waypoints=[],
                    press_h=round(press_h, 1))
    dist = float(steps[-1])
    wps, q = [], None
    try:
        for name, p, h in ([("hover", start, press_h + clear), ("press", start, press_h)]
                           + [(f"d{int(s)}", start + d * s, press_h) for s in steps]):
            q, tilt = plan_point(m, p, h, q, roll)
            p_end = p
            wps.append(dict(name=name, xyh=np.r_[p, h].round(1).tolist(), tilt=tilt,
                            joints={**{i: int(q[i - 1]) for i in range(1, 5)}, 5: int(roll)}))
        q, tilt = plan_point(m, p_end, press_h + clear, q, roll)
        wps.append(dict(name="lift", xyh=np.r_[p_end, press_h + clear].round(1).tolist(), tilt=tilt,
                        joints={**{i: int(q[i - 1]) for i in range(1, 5)}, 5: int(roll)}))
    except RuntimeError as exc:
        return dict(ok=False, reason=str(exc), waypoints=wps, press_h=round(press_h, 1))
    moved = c + d * dist
    return dict(ok=True, press_h=round(press_h, 1), start=start.round(1).tolist(), dir=d.round(3).tolist(),
                waypoints=wps, expected_center=moved.round(1).tolist(),
                r0=round(float(np.linalg.norm(c - m.base)), 1), r1=round(float(np.linalg.norm(moved - m.base)), 1))


def _m(m):
    return m if m is not None else _model()


def execute_slide(plan, move_fn=None, joints_fn=None, send_fn=None, out=print, max_load=None):
    """Close the jaws, hover, press, drag in steps (loads logged; optional abort above max_load), lift."""
    from .manip import load_mag
    if not plan.get("ok"):
        raise RuntimeError(f"plan not ok: {plan.get('reason')}")
    joints_fn = joints_fn or arm.joints
    real = send_fn is None
    arm.check_goal({6: SLIDE_GRIPPER})
    (send_fn or arm.send)({"gripper": SLIDE_GRIPPER})
    if real:
        arm.wait_idle(8)
    trace = []
    for w in plan["waypoints"]:
        speed = 4 if w["name"] in ("hover", "lift") else 3
        go_joints(w["joints"], move_fn, joints_fn, rounds=1, speed=speed)
        _, pos, mot = joints_fn()
        loads = [load_mag(mot[str(i)]["load"]) for i in range(1, 5)]
        trace.append(dict(name=w["name"], tilt=w["tilt"], loads=loads))
        out(f"   {w['name']:6s} xyh {w['xyh']} tilt {w['tilt']} loads {loads}")
        if max_load is not None and w["name"] not in ("hover", "lift") and max(loads) > max_load:
            go_joints(plan["waypoints"][-1]["joints"], move_fn, joints_fn, rounds=1)
            raise RuntimeError(f"load {max(loads)} > {max_load} at {w['name']}: lifted and stopped")
    return trace
