"""drawbot CLI: sensing, state and guarded motion for the SO101 marker-drawing setup (docs/DRAWBOT.md).

Anything that moves the arm prints its plan and exits unless --yes is given.
State dir: $DRAWBOT_STATE (default ./drawbot_state). Teleop dir: $ARM_CTL (default ./ctl).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from drawbot import arm, config, load, path, save, skills, smiley, track
from drawbot.frame import Frame, fit_from_paper


def fmt(v, nd=1):
    if isinstance(v, np.ndarray):
        return np.round(v, nd).tolist()
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, (list, tuple)):
        return [fmt(x, nd) for x in v]
    return v


def make_ctl():
    from drawbot.control import Ctl2
    return Ctl2(Frame.load())


def cmd_status(a):
    st = arm.status()
    print(f"moving {st['moving']}  age {time.time() - st['t']:.1f}s  model_xyz {st.get('model_xyz')}")
    for i in range(1, 7):
        m = st["motors"][str(i)]
        print(f"  ID{i} pos {m['pos']:5d} target {m['target']:5d} load {m['load']:5d} temp {m['temp']:3d}C"
              f"{'  TORQUE OFF' if m.get('torque_off') else ''}")
    print("last", json.dumps(st.get("last"))[:300])


def cmd_snap(a):
    import cv2
    from drawbot import rgbd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = a.name or time.strftime("snap_%H%M%S")
    g = rgbd.decode(rgbd.fetch(rgbd.COLOR_PORT, "frame.jpg"))
    cv2.imwrite(str(out / f"{stamp}_gemini.jpg"), g)
    saved = [f"{stamp}_gemini.jpg"]
    try:
        cv2.imwrite(str(out / f"{stamp}_wrist.jpg"), rgbd.wrist_frame())
        saved.append(f"{stamp}_wrist.jpg")
    except Exception as exc:
        print("wrist cam:", exc)
    if a.crop:
        x0, y0, x1, y1 = a.crop
        crop = g[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out / f"{stamp}_crop.jpg"), crop)
        saved.append(f"{stamp}_crop.jpg")
    print("saved", [str(out / s) for s in saved])


def cmd_blobs(a):
    fr = Frame.load()
    rows = track.list_blobs(fr, a.colors)
    if not rows:
        print("no colored blobs near the paper")
    for d in rows:
        tip = fmt(d["tipdir"], 2) if d["tip_known"] else "?"
        print(f"{d['color']:7s} xyh {fmt(d['xyh'])} area {d['area']:5d} px {fmt(list(d['px']), 0)}"
              f" axis {fmt(d['axis'], 2)} tip {tip} len {d['length']:.0f} w {d['width']:.0f}"
              f" r {d['r']:.0f} ang {d['ang']:.1f} tip_vs_radial {d['tip_vs_radial']:.0f}"
              f"{' table' if d['on_table'] else ''}{' marker?' if d['marker_like'] else ''}")


def cmd_track(a):
    fr = Frame.load()
    prior = None
    for _ in range(a.n):
        s = track.measure(fr, mode=a.mode, avg=a.avg, prior=prior)
        prior = s["xyh"] if a.follow else None
        print(f"xyh {fmt(s['xyh'])} tip {fmt(s['tip'])} r {s['r']:.0f} ang {s['ang']:.1f}"
              f" tilt r/t {s['tilt_r']:.0f}/{s['tilt_t']:.0f} area {s['area']} px {fmt(list(s['px']), 0)}")


def parse_goal(items):
    goal = {}
    for it in items:
        k, v = it.split("=")
        k = int(re.sub(r"^(ID|J)", "", k.strip().upper()) or 0)
        if not 1 <= k <= 6:
            raise SystemExit(f"bad joint {it}")
        goal[k] = int(v)
    return goal


def guarded_move(a, goal, speed):
    arm.check_goal(goal)
    t, p, _ = arm.joints()
    for k, v in goal.items():
        print(f"  ID{k}: target {t[k]} -> {v} ({v - t[k]:+d}), pos {p[k]}")
    if not a.yes:
        print("dry run (add --yes to move)")
        return False
    arm.move(goal, speed=speed)
    t, p, _ = arm.joints()
    print("done:", {k: p[k] for k in goal})
    return True


def cmd_move(a):
    guarded_move(a, parse_goal(a.goal), a.speed)


def cmd_level(a):
    t, _, _ = arm.joints()
    w = arm.down4(t[2], t[3]) if a.down else arm.level4(t[2], t[3])
    print(f"ID2 {t[2]} ID3 {t[3]} -> {'down4' if a.down else 'level4'} ID4 = {w}")
    guarded_move(a, {4: w}, a.speed)


def cmd_pan(a):
    tick = Frame.load().pan_for_angle(a.angle)
    print(f"plane angle {a.angle} deg -> pan {tick}")
    guarded_move(a, {1: tick}, a.speed)


def cmd_jacobian(a):
    c = make_ctl()
    if c.J is not None:
        print("stored J (mm per 100 ticks; rows x y h, cols", c.ids, "):\n", (c.J * 100).round(2))
    if not a.yes:
        print(f"dry run: --yes estimates J by +-{a.steps} tick moves around u {c.u.astype(int).tolist()}")
        return
    J = c.estimate(steps=tuple(a.steps))
    print("new J/100:\n", (J * 100).round(2))


def cmd_probe(a):
    c = make_ctl()
    y = c.y()
    print(f"barrel {fmt(y)} u {c.u.astype(int).tolist()} stored contact_h {c.contact_h}")
    if not a.yes:
        print(f"dry run: --yes lowers in {a.step} mm steps (max {a.max_steps}) until the barrel stops following")
        return
    r = skills.probe_contact(c, step=a.step, max_steps=a.max_steps)
    ref = load("tilt_ref.json")
    tip = skills.barrel_to_tip(r["barrel"], skills.tilt(np.array(r["trace"][-1]["u"]), ref), config()["pen_L"])
    print(f"contact_h {r['contact_h']:.1f} barrel {fmt(r['barrel'])} tip {fmt(tip)}")
    if a.save_site:
        site = load("site.json", {})
        site.update(tip_contact=tip.tolist(), contact_h=r["contact_h"])
        save("site.json", site)
        print("site.json updated")


def cmd_relocate(a):
    c = make_ctl()
    y = c.y()
    tgt = np.array([a.x, a.y, a.h])
    print(f"barrel {fmt(y)} -> {tgt.tolist()} ({np.linalg.norm(tgt - y):.0f} mm), u {c.u.astype(int).tolist()}")
    if not a.yes:
        print("dry run (add --yes to move)")
        return
    skills.relocate(c, tgt)


def cmd_draw(a):
    site = load("site.json")
    strokes = skills.layer_strokes(a.marker, site, config()["step"])
    for s in strokes:
        print(f"stroke {len(s)} pts, x {s[:, 0].min():.0f}..{s[:, 0].max():.0f} y {s[:, 1].min():.0f}..{s[:, 1].max():.0f}")
    if a.dry or not a.yes:
        print("dry run (add --yes to draw)")
        return
    pen = skills.Pen(make_ctl(), site)
    skills.draw_layer(pen, a.marker, start=a.start)


def cmd_park(a):
    pose = {int(k): v for k, v in load("park.json").items()}
    print("park pose", pose)
    if not a.yes:
        guarded_move(a, pose, 8)
        return
    skills.park(make_ctl(), pose)


def cmd_frame(a):
    try:
        old = Frame.load()
        print("frame.json:", json.dumps(fmt(old.to_dict(), 3)))
    except FileNotFoundError as exc:
        old = None
        print(exc)
    if not a.fit_paper:
        return
    from drawbot import rgbd
    P = rgbd.paper()
    fr = fit_from_paper(P)
    print(f"paper: inliers {P['inlier_frac']:.0%} resid {P['resid_mm']:.1f} mm corners {len(P['corners'])}"
          f" edges {[round(float(np.linalg.norm(P['corners'][i] - P['corners'][i - 1]))) for i in range(len(P['corners']))]}")
    if old is not None:
        fr.BASE, fr.pan_ang, fr.extra = old.BASE, old.pan_ang, old.extra
        print(f"ctr moved {np.linalg.norm(fr.CTR - old.CTR):.1f} mm, e1 angle "
              f"{np.degrees(np.arccos(np.clip(fr.E1 @ old.E1, -1, 1))):.1f} deg (base kept; re-fit it if the paper moved)")
    print("new:", json.dumps(fmt(fr.to_dict(), 3)))
    if a.save:
        fr.save()
        print("saved", path("frame.json"))


# --- kinematic model, calibration, manipulation ----------------------------------------------------------------
def axis_of(a):
    from drawbot.kinmodel import DOWN
    if getattr(a, "down", False):
        return list(DOWN)
    if getattr(a, "axis", None):
        v = np.array(a.axis, float)
        return (v / np.linalg.norm(v)).tolist()
    return None


def current_q5():
    _, p, _ = arm.joints()
    return [p[i] for i in range(1, 6)]


def cmd_calib_collect(a):
    from drawbot import calib
    ps = calib.poses(a.start_anchor)
    if a.roll is not None:
        print(f"first: safe home, then roll ID5 -> {a.roll} in guarded steps")
    print(f"{len(ps)} poses ({len(calib.ANCHORS) - a.start_anchor} anchors x {len(calib.PERT)} perturbations),"
          f" home {calib.HOME}, min safe h {calib.MIN_SAFE} mm -> {path(calib.SAMPLES)} (appends)")
    for k, g in ps[:3]:
        print(f"  anchor {k}: {g}")
    if not a.yes:
        print("dry run: hold the orange marker in the gripper, clear the table, add --yes to collect")
        return
    n = calib.collect(start_anchor=a.start_anchor, roll=a.roll)
    print("samples", n)


def cmd_kin_fit(a):
    from drawbot import calib
    calib.run(sign_search=a.signs, thresh=a.thresh, save_model=not a.no_save)


def cmd_fk(a):
    from drawbot import kinmodel
    m = kinmodel.model()
    q = a.q if a.q else current_q5()
    if len(q) != 5:
        raise ValueError("fk needs 5 joint values (ID1..ID5) or none for the measured pose")
    tcp, ax, R = m.fk(q)
    r = float(np.hypot(*(tcp[:2] - m.base)))
    print(f"q {q}\n  tcp (fixed-jaw tip) {fmt(tcp)}  r {r:.0f} from BASE  jaw axis {fmt(ax, 2)}"
          f"  close dir {fmt(m.close_dir(q), 2)}")
    print(f"  marker tool {fmt(m.fk(q, m.TOOL)[0])}  grasp center {fmt(m.fk(q, m.grasp_tool)[0])}")


def cmd_ik(a):
    from drawbot import kinmodel
    m = kinmodel.model()
    tool = kinmodel.tool_arg(a.tool, m)
    roll = a.roll if a.roll is not None else kinmodel.ROLL_HOME
    r = m.ik([a.x, a.y, a.h], axis_of(a), roll, tool=tool)
    res = m.residual([a.x, a.y, a.h], r.q, roll, tool)
    ok = r.ok(a.max_res)
    print(f"q {r.q.tolist()} roll {roll}  residual {res:.1f} mm  axis {fmt(r.axis, 2)} (err {r.axis_err:.1f} deg)"
          f"  {'REACHABLE' if ok else 'UNREACHABLE'}")
    return 0 if ok else 2


def cmd_goto(a):
    from drawbot import kinmodel
    m = kinmodel.model()
    tool = kinmodel.tool_arg(a.tool, m)
    q5 = current_q5()
    roll = a.roll if a.roll is not None else q5[4]
    tgt = [a.x, a.y, a.h]
    r = m.ik(tgt, axis_of(a), roll, seed=q5[:4], tool=tool, w_axis=25.0)
    res = m.residual(tgt, r.q, roll, tool)
    print(f"now {q5} at {fmt(m.fk(q5, tool)[0])} -> q {r.q.tolist()} (roll {roll} kept), residual {res:.1f} mm,"
          f" axis {fmt(r.axis, 2)}")
    if res > a.max_res:
        raise RuntimeError(f"unreachable: residual {res:.1f} mm > {a.max_res}")
    if not a.yes:
        print("dry run (add --yes to move)")
        return
    got, q = m.goto(tgt, axis=axis_of(a), roll=roll, tool=tool, speed=a.speed, max_res=a.max_res)
    print(f"reached (model, measured joints) {fmt(got)}")


def cmd_heightmap(a):
    from drawbot import manip
    xs, ys, g = manip.heightmap(a.x0, a.x1, a.y0, a.y1, cell=a.cell, pct=a.pct, avg=a.avg)
    print(manip.show_heightmap(xs, ys, g))


def cmd_markers(a):
    from drawbot import scene
    region = ((a.region[0], a.region[1]), (a.region[2], a.region[3])) if a.region else scene.REGION
    sc = scene.scan(region=region, all_blobs=a.all)
    print(f"image brightness {sc['brightness']:.0f}{'  DARK: turn the light on' if sc['dark'] else ''}")
    if not sc["markers"]:
        print("no markers in", region)
    for name, v in sc["markers"].items():
        for d in (v if isinstance(v, list) else [v]):
            print(f"{name:7s} c {fmt(d['c'])} axis {fmt(d['axis2'], 2)} len {d['length']:.0f} area {d['area']}"
                  f" V {d['brightness']:.0f} r {d['r']:.0f} ang {d['ang']:.1f} {d['reach']}"
                  f"{'' if d['reachable'] else '  OUT OF REACH'}")


def cmd_poke(a):
    from drawbot import kinmodel, manip
    m = kinmodel.model()
    p0, p1 = [a.x0, a.y0], [a.x1, a.y1]
    for name, t in (("hover p0", p0 + [a.clear]), ("down p0", p0 + [a.h]), ("end p1", p1 + [a.h]),
                    ("lift p1", p1 + [a.clear])):
        r = m.ik(t, kinmodel.DOWN, current_q5()[4], w_axis=25.0)
        print(f"  {name:8s} {t} -> q {r.q.tolist()} residual {r.res:.1f} mm")
    if not a.yes:
        print("dry run (add --yes to poke). Note: pushing packed markers sideways moves the whole case.")
        return
    loads = manip.poke(p0, p1, a.h, clear=a.clear, steps=a.steps)
    print("loads per step (ID1..ID4):", loads)


def cmd_flip(a):
    from drawbot import manip
    t, p, m = arm.joints()
    target = a.to if a.to is not None else manip.flip_target(t[5])
    print(f"roll ID5 target {t[5]} pos {p[5]} load {manip.load_mag(m['5']['load'])} -> {target}"
          f" in {a.step}-tick steps (guard load > {a.max_load}, lag > {a.max_lag})")
    if not a.yes:
        print("dry run (add --yes to turn the wrist)")
        return
    r = manip.flip_roll(target, step=a.step, max_load=a.max_load, max_lag=a.max_lag)
    print("done" if r["ok"] else f"stopped: {r['reason']}", "roll", r["roll"])


def cmd_pick(a):
    from drawbot import manip, scene
    sc = scene.scan()
    if sc["dark"]:
        print(f"warning: image brightness {sc['brightness']:.0f} (dark)")
    pose = sc["markers"].get(a.color)
    if pose is None:
        raise RuntimeError(f"no {a.color} marker found (seen: {list(sc['markers'])})")
    print(f"{a.color}: c {fmt(pose['c'])} axis {fmt(pose['axis2'], 2)} r {pose['r']:.0f} ({pose['reach']})")
    plan = manip.pick_marker(pose, roll_ref=current_q5()[4] if a.from_current else 3746)
    for k in ("ok", "mode", "tilt_deg", "roll", "axis", "tcp_h", "close_dir", "gripper", "tool_mm", "warnings",
              "reason", "tried"):
        if k in plan:
            print(f"  {k}: {plan[k]}")
    for leg in ("approach", "grasp", "lift"):
        if leg in plan:
            L = plan[leg]
            print(f"  {leg:8s} xyh {L['xyh']} joints {L['joints']} res {L['res']} mm")
    if a.dry or not a.yes:
        print("plan only (add --yes to execute)")
        return
    got = manip.execute_pick(plan)
    print("done:", {k: fmt(v) for k, v in got.items()})


# --- marker picking (field-proven 2026-09-26; drawbot/pick.py, drawbot/slide.py) -------------------------------
def show_shape(r):
    if r is None:
        return "no marker shape (fewer than 30 points 6..28 mm above the table)"
    return (f"center {fmt(r['center'])} axis {fmt(r['axis'], 2)} len {r['length']:.0f} w {r['width']:.0f}"
            f" top {r['top']:.1f} n {r['n']} r {r.get('r', float('nan')):.0f} near_end {fmt(r['near_end'])}"
            f"{'  TALL OBJECT NEARBY' if r['tall_nearby'] > 50 else ''}")


def cmd_locate(a):
    from drawbot import pick
    print(show_shape(pick.locate_shape([a.x, a.y], radius=a.radius)))


def find_or_die(color):
    from drawbot import pick
    f = pick.find_marker(color)
    if f is None:
        raise RuntimeError(f"no {color} blob found (dark-normalised colour); try `locate X Y` on a rough spot")
    b = f["blob"]
    print(f"{color}: colour blob {fmt(b['c'])} area {b['area']} | shape: {show_shape(f['shape'])}")
    return f


def cmd_find(a):
    find_or_die(a.color)


def cmd_pick2(a):
    from drawbot import pick
    if len(a.target) == 1:
        pose = find_or_die(a.target[0])["pose"]
        center, axis, top = pose["center"], pose["axis"], pose["top"]
        print(f"pose from {pose['source']}")
    elif len(a.target) == 4:
        x, y, ax, ay = (float(v) for v in a.target)
        center, axis, top = [x, y], [ax, ay], None
    else:
        raise ValueError("pick2 takes COLOR or X Y AX AY")
    plan = pick.plan_pick(center, axis, top=top, tilt=a.tilt, offset=a.offset, grasp_h=a.grasp_h)
    for k in ("ok", "reason", "center", "axis", "r", "roll", "roll_err_deg", "u", "tcp_xy", "tilt", "offset",
              "grasp_h", "gripper", "warnings"):
        if k in plan:
            print(f"  {k}: {plan[k]}")
    for w in plan.get("waypoints", []) + ([plan["lift"]] if "lift" in plan else []):
        print(f"  {w['name']:7s} xyh {w['xyh']} joints {w['joints']} res {w['res']} mm")
    if a.dry or not a.yes or not plan["ok"]:
        print("plan only (add --yes to execute)" if plan["ok"] else "not executable")
        return 0 if plan["ok"] else 2
    r = pick.execute_pick(plan)
    print("HOLDING" if r["held"] else "NOT holding", r)
    return 0 if r["held"] else 3


def cmd_slide(a):
    from drawbot import pick, slide
    f = find_or_die(a.color)
    if f["shape"] is None:
        raise RuntimeError("slide needs the marker shape (top height, ends); none found")
    plan = slide.plan_slide(f["shape"], dist=a.dist, step=a.step)
    for k in ("ok", "reason", "press_h", "start", "dir", "r0", "r1", "expected_center"):
        if k in plan:
            print(f"  {k}: {plan[k]}")
    for w in plan["waypoints"]:
        print(f"  {w['name']:6s} xyh {w['xyh']} tilt {w['tilt']} joints {w['joints']}")
    if not a.yes or not plan["ok"]:
        print("plan only (add --yes to slide)" if plan["ok"] else "not executable")
        return 0 if plan["ok"] else 2
    slide.execute_slide(plan, max_load=a.max_load)
    after = pick.find_marker(a.color)
    print("after:", show_shape(after["shape"]) if after else "not found")


MOVES = {"move", "level", "pan", "jacobian", "probe-contact", "relocate", "draw-layer", "park",
         "calib-collect", "goto", "poke", "flip", "pick", "pick2", "slide"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status", help="teleop status: joints, loads, temps")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("snap", help="save Gemini + wrist frames (+ enlarged crop)")
    p.add_argument("--out", default="captures")
    p.add_argument("--name")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--scale", type=float, default=3.0)
    p.set_defaults(fn=cmd_snap)
    p = sub.add_parser("blobs", help="every colored blob near the paper, in plane coords")
    p.add_argument("--colors", nargs="+", choices=list(track.PALETTE))
    p.set_defaults(fn=cmd_blobs)
    p = sub.add_parser("track", help="tracked marker/tape position")
    p.add_argument("--mode", help="orange|red|tape|blue|green|pink|yellow|bluemarker|hue:LO-HI[:SAT]")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--avg", type=int, default=3)
    p.add_argument("--follow", action="store_true", help="use each reading as the next prior")
    p.set_defaults(fn=cmd_track)
    p = sub.add_parser("frame", help="show frame.json; --fit-paper re-fits the plane from the paper")
    p.add_argument("--fit-paper", action="store_true")
    p.add_argument("--save", action="store_true")
    p.set_defaults(fn=cmd_frame)
    p = sub.add_parser("move", help="raw joint goals, e.g. move 1=1900 4=2100 --yes")
    p.add_argument("goal", nargs="+")
    p.add_argument("--speed", type=int, default=8)
    p.set_defaults(fn=cmd_move)
    p = sub.add_parser("level", help="wrist ID4 to level4(a2,a3) (or --down)")
    p.add_argument("--down", action="store_true")
    p.add_argument("--speed", type=int, default=8)
    p.set_defaults(fn=cmd_level)
    p = sub.add_parser("pan", help="point the pan at a plane angle (deg) about the base")
    p.add_argument("angle", type=float)
    p.add_argument("--speed", type=int, default=8)
    p.set_defaults(fn=cmd_pan)
    p = sub.add_parser("jacobian", help="show / re-estimate the control Jacobian")
    p.add_argument("--steps", type=int, nargs=3, default=[35, 40, 40])
    p.set_defaults(fn=cmd_jacobian)
    p = sub.add_parser("probe-contact", help="lower the pen until contact; saves contact_h")
    p.add_argument("--step", type=float, default=1.5)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--no-save-site", dest="save_site", action="store_false")
    p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("relocate", help="move the barrel in the air to plane X Y H")
    for k in ("x", "y", "h"):
        p.add_argument(k, type=float)
    p.set_defaults(fn=cmd_relocate)
    p = sub.add_parser("draw-layer", help="draw one smiley layer")
    p.add_argument("marker", choices=list(smiley.layers()))
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--dry", action="store_true")
    p.set_defaults(fn=cmd_draw)
    p = sub.add_parser("park", help="lift the pen and go to park.json")
    p.set_defaults(fn=cmd_park)
    p = sub.add_parser("calib-collect", help="collect kinematic calibration samples (orange marker in the gripper)")
    p.add_argument("--start-anchor", type=int, default=0)
    p.add_argument("--roll", type=int, help="turn ID5 here first (guarded), e.g. 1698 for a second-roll set")
    p.set_defaults(fn=cmd_calib_collect)
    p = sub.add_parser("kin-fit", help="fit kin_model.json to calib_samples.jsonl (outliers > --thresh dropped)")
    p.add_argument("--signs", action="store_true", help="also search joint sign flips for ID2..ID4 (8x slower)")
    p.add_argument("--thresh", type=float, default=15.0)
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(fn=cmd_kin_fit)
    p = sub.add_parser("fk", help="model pose of ID1..ID5 ticks (default: measured joints)")
    p.add_argument("q", type=int, nargs="*")
    p.set_defaults(fn=cmd_fk)
    tool_help = "tool point: bare = fitted marker point, 'grasp' = grasp center, 'tcp', or X Y Z mm (gripper frame)"
    for name, hlp in (("ik", "IK for plane X Y H (exit 2 if unreachable)"), ("goto", "IK + move + sag correction")):
        p = sub.add_parser(name, help=hlp)
        for k in ("x", "y", "h"):
            p.add_argument(k, type=float)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--down", action="store_true", help="jaws straight down")
        g.add_argument("--axis", type=float, nargs=3, metavar=("AX", "AY", "AZ"), help="jaw direction (plane frame)")
        p.add_argument("--tool", nargs="*", help=tool_help)
        p.add_argument("--roll", type=int, help="ID5 ticks assumed (not commanded); default 3746 / current")
        p.add_argument("--max-res", type=float, default=4.0)
        p.add_argument("--speed", type=int, default=5)
    sub.choices["ik"].set_defaults(fn=cmd_ik)
    sub.choices["goto"].set_defaults(fn=cmd_goto)
    p = sub.add_parser("heightmap", help="max-height grid of a plane region from depth")
    for k in ("x0", "x1", "y0", "y1"):
        p.add_argument(k, type=float)
    p.add_argument("--cell", type=float, default=6.0)
    p.add_argument("--pct", type=float, default=90.0)
    p.add_argument("--avg", type=int, default=7)
    p.set_defaults(fn=cmd_heightmap)
    p = sub.add_parser("markers", help="markers per color in the case region: pose, brightness, reach")
    p.add_argument("--region", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"))
    p.add_argument("--all", action="store_true", help="every blob per color, not just the largest")
    p.set_defaults(fn=cmd_markers)
    p = sub.add_parser("poke", help="closed fingertip: down at X0 Y0 to H, slide to X1 Y1, lift")
    for k in ("x0", "y0", "x1", "y1", "h"):
        p.add_argument(k, type=float)
    p.add_argument("--clear", type=float, default=55.0)
    p.add_argument("--steps", type=int, default=3)
    p.set_defaults(fn=cmd_poke)
    p = sub.add_parser("flip", help="turn the wrist roll 180 deg (or --to ROLL) with load/lag guard")
    p.add_argument("--to", type=int)
    p.add_argument("--step", type=int, default=100)
    p.add_argument("--max-load", type=int, default=300)
    p.add_argument("--max-lag", type=int, default=40)
    p.set_defaults(fn=cmd_flip)
    p = sub.add_parser("pick", help="plan (and with --yes execute) picking up a marker by color")
    p.add_argument("color", choices=["green", "pink", "blue", "yellow", "orange"])
    p.add_argument("--dry", action="store_true")
    p.add_argument("--from-current", action="store_true", help="prefer the roll nearest the current one")
    p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("locate", help="marker pose by SHAPE from depth near plane X Y (read-only)")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("--radius", type=float, default=60.0)
    p.set_defaults(fn=cmd_locate)
    p = sub.add_parser("find", help="dark-robust colour find + shape refinement (read-only)")
    p.add_argument("color", choices=["green", "pink", "blue", "yellow", "orange"])
    p.set_defaults(fn=cmd_find)
    p = sub.add_parser("pick2", help="field-proven pick: COLOR or X Y AX AY; plan, --yes executes")
    p.add_argument("target", nargs="+", metavar="COLOR | X Y AX AY")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--tilt", type=float, help="jaw tilt outward, deg (default config pick_tilt 20)")
    p.add_argument("--offset", type=float, help="fixed-jaw tip offset, mm (default config pick_offset_mm 25)")
    p.add_argument("--grasp-h", type=float, help="fixed-jaw tip height at the grasp (default pick_grasp_h 9)")
    p.set_defaults(fn=cmd_pick2)
    p = sub.add_parser("slide", help="drag a marker along its axis toward the robot (closed fingertip on top)")
    p.add_argument("color", choices=["green", "pink", "blue", "yellow", "orange"])
    p.add_argument("--dist", type=float, default=80.0)
    p.add_argument("--step", type=float, default=10.0)
    p.add_argument("--max-load", type=int, help="abort (lift) when a joint load exceeds this while dragging")
    p.set_defaults(fn=cmd_slide)
    for name, sp in sub.choices.items():
        if name in MOVES:
            sp.add_argument("--yes", action="store_true", help="actually move the arm")
    a = ap.parse_args(argv)
    np.set_printoptions(suppress=True)
    if getattr(a, "yes", False):
        arm.enable()
    try:
        return a.fn(a)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
