"""Higher-level moves on top of Ctl2: contact probing, relocation in air, drawing a smiley layer, park.

site.json: tip_contact [x, y] (plane mm), contact_h (barrel h at pen contact), center [x, y] (face
center, default tip_contact + (r_face + 3, 0)), r_face (mm), hover_u [pan, ID3, ID4] (drawing pose).
tilt_ref.json: {"id3", "id4"} where the pen is vertical.
"""
import json
import time

import numpy as np

from . import arm, config, load, path, save
from . import smiley


def tilt(u, ref):
    """Pen tilt (rad) from elbow+wrist ticks vs the vertical reference; + = pen bottom toward the base."""
    return (u[1] - ref["id3"] + u[2] - ref["id4"]) * 2 * np.pi / 4096


def tip_to_barrel(p_tip, th, L=28.0):
    """Barrel xy that puts the tip at p_tip for tilt th (compensated along plane x only, the base side)."""
    return np.array([p_tip[0] + L * np.sin(th), p_tip[1]])


def barrel_to_tip(b, th, L=28.0):
    return np.array([b[0] - L * np.sin(th), b[1]])


def face_center(site):
    r = site.get("r_face", 32.0)
    return site.get("center", [site["tip_contact"][0] + r + 3.0, site["tip_contact"][1]])


def face_to_plane(p, center, scale):
    """Face upright as seen from the Gemini: face +x -> plane -x, face +y -> plane -y."""
    return np.array([center[0] - scale * p[0], center[1] - scale * p[1]])


def resample(pts, step):
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def layer_strokes(marker, site, step=2.0):
    scale = site.get("r_face", 32.0) / smiley.R
    ctr = face_center(site)
    return [resample([face_to_plane(p, ctr, scale) for p in s], step) for s in smiley.layers()[marker]]


class Pen:
    """Pen-in-gripper geometry + the controller: tip targets, pen up/down."""

    def __init__(self, c, site=None, ref=None, cfg=None):
        cfg = cfg or config()
        self.c, self.site = c, site if site is not None else load("site.json")
        self.ref = ref if ref is not None else load("tilt_ref.json")
        self.L, self.UP, self.PRESS, self.STEP = cfg["pen_L"], cfg["pen_up"], cfg["press"], cfg["step"]
        self.cfg = cfg
        if c.contact_h is None:
            c.contact_h = self.site.get("contact_h")

    def barrel_for(self, p_tip):
        return tip_to_barrel(p_tip, tilt(self.c.u, self.ref), self.L)

    def go(self, p_tip, h, freeze_h=False, tol=1.5, iters=6):
        y = None
        for _ in range(1 if freeze_h else 2):  # re-solve the barrel target as the tilt changes
            bxy = self.barrel_for(p_tip)
            y = self.c.servo([bxy[0], bxy[1], h], tol=tol, iters=iters, freeze_h=freeze_h)
        return y

    def hover(self, p_tip, tol=2.0, iters=8):
        return self.go(p_tip, self.c.contact_h + self.UP, tol=tol, iters=iters)

    def pen_down(self, p_tip):
        self.go(p_tip, self.c.contact_h + 2.5, tol=1.5, iters=6)
        self.c.command(self.c.u + self.c.solve([0, 0, -(2.5 + self.PRESS)]))

    def pen_up(self):
        self.c.command(self.c.u + self.c.solve([0, 0, self.UP + self.PRESS]))

    def site_bounds(self):
        h = np.array(self.site["hover_u"])
        return [(h[k] + lo, h[k] + hi) for k, (lo, hi) in enumerate(self.cfg["site_bounds"])]


def reacquire(c, tries=5, wide_gate=100.0):
    """After a dropout: normal tracking, then a wider gate around the last position."""
    for _ in range(tries):
        try:
            return c.y(prior=c.last_y)
        except RuntimeError:
            try:
                y = np.asarray(c.measure(prior=c.last_y, gate=wide_gate)["xyh"], float)
                c.last_y = y
                return y
            except RuntimeError:
                time.sleep(1.0)
    return None


def draw_layer(pen, marker, start=1, log=None, out=print):
    """Draw one smiley layer. start: resume the first stroke at waypoint K (1-based). Dropouts lift the pen,
    re-acquire (wider gate), return to the waypoint and resume. Returns per-stroke (mean, max) errors."""
    c = pen.c
    strokes = layer_strokes(marker, pen.site, pen.STEP)
    c.bounds = pen.site_bounds()
    logp = log or path(f"layer_{marker}.log")
    t0, summary = time.time(), []
    for si, s in enumerate(strokes):
        first = s[start - 1] if (si == 0 and start > 1) else s[0]
        out(f"[{marker} {si}] travel to start {first.round(1).tolist()}")
        pen.hover(first)
        pen.pen_down(first)
        errs = []
        for k, p in enumerate(s[1:], 1):
            if si == 0 and k < start:
                continue
            try:
                c.blend = 0.5
                y = pen.go(p, 0, freeze_h=True, tol=1.0, iters=1)
                c.blend = 0.0
            except RuntimeError as exc:
                out(f"   pt {k}: {exc} -> lift, re-acquire, resume")
                pen.pen_up()
                if reacquire(c) is None:
                    raise
                c.rejects = 0
                pen.go(p, c.contact_h + pen.UP, tol=2.5, iters=6)
                pen.pen_down(p)
                y = c.y()
            e = float(np.linalg.norm(pen.barrel_for(p) - y[:2]))
            errs.append(e)
            with open(logp, "a") as f:
                f.write(json.dumps(dict(t=time.time(), stroke=si, k=k, target=p.tolist(), barrel=y.tolist(),
                                        u=c.u.tolist(), err=e)) + "\n")
            if k % 10 == 0:
                out(f"   pt {k}/{len(s) - 1} barrel {y.round(1).tolist()} err {e:.1f} mm  u {c.u.astype(int).tolist()}"
                    f"  {time.time() - t0:.0f}s")
        pen.pen_up()
        if errs:
            summary.append((float(np.mean(errs)), float(np.max(errs))))
            out(f"[{marker} {si}] done: mean err {summary[-1][0]:.1f} max {summary[-1][1]:.1f} mm")
    try:
        import cv2
        cv2.imwrite(str(path(f"after_{marker}.jpg")), c.measure()["color"])
    except Exception as exc:  # the drawing is done; a missing snapshot is not an error
        out(f"after-image skipped: {exc}")
    return summary


def relocate(c, target, bounds=None, leg=25.0, legs=12, tol=3.0, out=print):
    """Move the barrel in the air to target (x, y, h) in <= leg mm steps; re-estimate J on poor progress."""
    c.bounds = bounds if bounds is not None else config()["relocate_bounds"]
    tgt = np.asarray(target, float)
    y = c.y()
    out(f"start {y.round(1).tolist()} u {c.u.astype(int).tolist()}")
    if c.contact_h is not None and y[2] < c.contact_h + 8:
        c.command(c.u + np.array([0, 0, -60]))  # wrist up first: never drag the pen
        y = c.y()
        out(f"lifted {y.round(1).tolist()}")
    c.estimate(steps=(35, 40, 30))
    out(f"J/100:\n{(c.J * 100).round(1)}")
    for k in range(legs):
        y = c.y()
        d = tgt - y
        if np.linalg.norm(d) < tol:
            break
        step = d if np.linalg.norm(d[:2]) < leg else np.r_[d[:2] / np.linalg.norm(d[:2]) * leg, d[2]]
        before = np.linalg.norm(d)
        y = c.servo(y + step, tol=2.5, iters=4, verbose=True)
        after = np.linalg.norm(tgt - y)
        out(f"leg {k}: dist {before:.1f} -> {after:.1f}")
        if after > before - 5:
            out("  poor progress: re-estimating J")
            c.estimate(steps=(35, 40, 30))
    y = c.y()
    out(f"final {y.round(1).tolist()} u {c.u.astype(int).tolist()}")
    save("relocate_out.json", dict(u=c.u.tolist(), y=y.tolist()))
    return y


def commanded_ok(c, st):
    """Verified motion: teleop's targets equal what we commanded (it accepted and is executing it)."""
    t = arm.joints(st)[0]
    return all(abs(t[i] - c.u[k]) <= 1 for k, i in enumerate(c.ids))


def probe_contact(c, step=1.5, frac=0.35, n_stall=2, min_free=2, max_steps=30, status_fn=None, out=print):
    """Lower the pen in `step` mm increments (through J) until the camera stops seeing the barrel follow.

    A step counts as free when measured dh >= frac * predicted. Contact = n_stall consecutive non-free
    steps, but only after min_free verified free steps (so a stopped controller or lost tracking is not
    mistaken for contact) and only when teleop's targets equal ours. Lifts back up afterwards.
    """
    status_fn = status_fn or arm.status
    y = c.y()
    trace, free, stalled = [], 0, []
    for k in range(max_steps):
        du = c.solve([0, 0, -step])
        u_prev, y_prev = c.u.copy(), y
        c.command(c.u + du)
        st = status_fn()
        if not commanded_ok(c, st):
            raise RuntimeError(f"teleop targets {arm.joints(st)[0]} != commanded {c.u.tolist()}: not verified")
        y = c.y(prior=y_prev)
        pred = float(c.J[2] @ (c.u - u_prev))
        dh = float(y[2] - y_prev[2])
        _, pos, m = arm.joints(st)
        lag = int(sum(abs(c.u[j] - pos[i]) for j, i in enumerate(c.ids)))
        moving = pred < 0 and dh <= frac * pred
        trace.append(dict(k=k, u=c.u.tolist(), y=y.tolist(), dh=dh, pred=pred, lag=lag,
                          load=[m[str(i)]["load"] for i in c.ids]))
        out(f"  step {k}: h {y[2]:.1f} dh {dh:+.1f} (pred {pred:+.1f}) lag {lag} {'free' if moving else 'STALL'}")
        if moving:
            free, stalled = free + 1, []
            continue
        if free < min_free:
            raise RuntimeError("no verified free motion before the stall: check teleop / tracking")
        stalled.append(y)
        if len(stalled) >= n_stall:
            break
    else:
        raise RuntimeError(f"no contact within {max_steps * step:.0f} mm")
    ys = np.array(stalled)
    contact_h = float(ys[:, 2].mean())
    barrel = ys.mean(0)
    c.contact_h = contact_h
    c.save()
    c.command(c.u + c.solve([0, 0, config()["pen_up"] + len(stalled) * step]))
    return dict(contact_h=contact_h, barrel=barrel, u=c.u.tolist(), trace=trace)


def park(c, pose=None, speed=8):
    """Lift the pen (through J) then go to the park pose (park.json {id: ticks})."""
    pose = pose or {int(k): v for k, v in load("park.json").items()}
    if c.J is not None:
        c.command(c.u + c.solve([0, 0, config()["pen_up"] + config()["press"]]))
    arm.move(pose, speed=speed)
