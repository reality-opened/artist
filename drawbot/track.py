"""Depth-camera tracking of the marker (or tape) in the paper-plane frame, and markers lying on the table.

Why depth: the Gemini gives metric 3D for every pixel, so a colored blob maps straight to plane
(x, y, h) with no hand-eye model; the URDF is not trusted for absolute positions.
"""
import os

import cv2
import numpy as np

from . import config
from .rgbd import backproject, color_blob, grab, project

# Tracking modes (field-tested: orange, tape). min_h: lowest allowed blob-center height (mm).
MODES = {
    "orange": dict(hue=(170, 12), min_sat=150, max_mm=900, min_h=-20.0),
    "tape": dict(hue=(100, 125), min_sat=60, max_mm=550, min_h=15.0),   # blue tape on gripper+marker
    "green": dict(hue=(35, 85), min_sat=80, max_mm=900, min_h=-20.0),
    "pink": dict(hue=(155, 178), min_sat=70, max_sat=185, min_val=100, max_mm=900, min_h=-20.0),
    "yellow": dict(hue=(27, 38), min_sat=100, min_val=140, max_mm=900, min_h=-20.0),
    "bluemarker": dict(hue=(100, 125), min_sat=80, min_val=60, max_mm=900, min_h=-20.0),
}
MODES["red"] = MODES["orange"]
MODES["blue"] = MODES["tape"]  # TRACK=blue in the field scripts meant the tape
# Palette for list_blobs (markers in the case / on the table). Orange sat raised so pink caps don't leak in.
PALETTE = {
    "orange": dict(MODES["orange"], hue=(170, 14), min_sat=185),
    "pink": MODES["pink"], "green": MODES["green"], "yellow": MODES["yellow"], "blue": MODES["bluemarker"],
}
X_MAX, Y_MAX, H_MAX, MIN_AREA = 250.0, 120.0, 250.0, 150


def mode_spec(mode=None):
    mode = mode or os.environ.get("TRACK") or config()["track"]
    if isinstance(mode, dict):
        return mode
    if mode.startswith("hue:"):  # hue:LO-HI[:SAT]
        parts = mode[4:].split(":")
        lo, hi = (int(v) for v in parts[0].split("-"))
        return dict(hue=(lo, hi), min_sat=int(parts[1]) if len(parts) > 1 else 80, max_mm=900, min_h=-20.0)
    return MODES[mode]


def blob_kw(spec):
    return {k: spec[k] for k in ("hue", "min_sat", "max_mm", "min_val", "max_sat") if k in spec}


def select(cands, prior=None, gate=40.0, min_h=-20.0):
    """cands: [(blob, xyh)] sorted by area. Keep blobs inside the work box; with a prior, the nearest
    within `gate` mm wins, else the largest. Returns (blob, xyh) or None."""
    ok = [(b, q) for b, q in cands if b["area"] > MIN_AREA and abs(q[0]) < X_MAX and abs(q[1]) < Y_MAX
          and min_h < q[2] < H_MAX]
    if prior is not None:
        prior = np.asarray(prior, float)
        ok = sorted([c for c in ok if np.linalg.norm(c[1] - prior) < gate], key=lambda c: np.linalg.norm(c[1] - prior))
    return ok[0] if ok else None


def tip_estimate(fr, b, xyh):
    """Pen axis from the blob's 3D principal direction (barrel ~4x longer than wide), tilts vs base
    radial/tangential, and tip = lowest colored points near the barrel."""
    rel = xyh[:2] - fr.BASE
    ang = np.arctan2(rel[1], rel[0])
    q = fr.plane(b["pts"][:: max(1, len(b["pts"]) // 800)])
    q = q[np.linalg.norm(q - np.median(q, 0), axis=1) < 60]
    ax = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
    ax = ax if ax[2] > 0 else -ax  # point up
    radial = ax[0] * np.cos(ang) + ax[1] * np.sin(ang)
    tang = -ax[0] * np.sin(ang) + ax[1] * np.cos(ang)
    allq = fr.plane(b["pts"])
    allq = allq[np.linalg.norm(allq[:, :2] - xyh[:2], axis=1) < 40]
    low = allq[allq[:, 2] <= np.percentile(allq[:, 2], 3)]
    tip = np.r_[low[:, :2].mean(0), np.percentile(allq[:, 2], 1)]
    return dict(tip=tip, r=float(np.hypot(*rel)), ang=float(np.degrees(ang)),
                tilt_r=float(np.degrees(np.arctan2(radial, ax[2]))), tilt_t=float(np.degrees(np.arctan2(tang, ax[2]))))


def measure(fr, mode=None, avg=3, tries=8, prior=None, gate=40.0, grab_fn=grab):
    """Tracked blob in plane coords. prior: last known (x, y, h); take the qualifying blob nearest to it."""
    spec = mode_spec(mode)
    for _ in range(tries):
        color, depth, K = grab_fn(avg=avg)
        cands = [(b, fr.plane(b["p"])) for b in color_blob(color, depth, K, every=True, **blob_kw(spec))]
        hit = select(cands, prior, gate, spec.get("min_h", -20.0))
        if hit is None:
            continue
        b, xyh = hit
        return dict(xyh=xyh, px=b["px"], area=b["area"], color=color, depth=depth, K=K, **tip_estimate(fr, b, xyh))
    raise RuntimeError("marker not visible")


def tip_from_image(fr, s, max_drop=150):
    """Pen tip height: walk down the vertical line through the barrel center until the orange ends."""
    hsv = cv2.cvtColor(s["color"], cv2.COLOR_BGR2HSV)
    h_ = hsv[..., 0]
    orange = ((h_ >= 170) | (h_ <= 12)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70)
    x, y, hc = s["xyh"]
    last = None
    for dh in np.arange(0, max_drop, 1.0):
        u, v = project(fr.to_cam([x, y, hc - dh]), s["K"])[0]
        u, v = int(round(u)), int(round(v))
        if not (0 <= v < orange.shape[0] and 0 <= u < orange.shape[1]):
            break
        if orange[max(0, v - 1):v + 2, max(0, u - 3):u + 4].any():
            last = hc - dh
        elif last is not None and (hc - dh) < last - 6:
            break
    return last


def angdiff(a, b):
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def white_frac(hsv, uv, r=5):
    u, v = int(uv[0]), int(uv[1])
    patch = hsv[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    return float(((patch[..., 1] < 70) & (patch[..., 2] > 140)).mean()) if patch.size else 0.0


def describe(fr, b, hsv, K, name=""):
    """Plane pose of one blob: center xyh, 2D axis, length/width, tip direction (white tip end), r/ang."""
    q = fr.plane(b["pts"])
    ctr = np.median(q, 0)
    xy = q[:, :2] - q[:, :2].mean(0)
    ax = np.linalg.svd(xy, full_matrices=False)[2][0] if len(q) > 2 else np.array([1.0, 0])
    proj = (q[:, :2] - ctr[:2]) @ ax
    perp = (q[:, :2] - ctr[:2]) @ np.array([-ax[1], ax[0]])
    white = []
    for sgn in (1, -1):
        e = ctr[:2] + sgn * ax * (np.percentile(sgn * proj, 99) + 6)
        white.append(white_frac(hsv, project(fr.to_cam([e[0], e[1], 4]), K)[0]))
    tipdir = ax if white[0] > white[1] else -ax
    r, ang = fr.polar(ctr)
    radial = (ctr[:2] - fr.BASE) / max(r, 1e-9)
    length = float(np.percentile(proj, 98) - np.percentile(proj, 2))
    width = float(np.percentile(perp, 95) - np.percentile(perp, 5))
    return dict(color=name, xyh=ctr, area=b["area"], px=b["px"], axis=ax, tipdir=tipdir, white=white,
                tip_known=abs(white[0] - white[1]) > 0.2, r=r, ang=ang, tip_vs_radial=angdiff(radial, tipdir),
                length=length, width=width, on_table=bool(ctr[2] < 40),
                marker_like=bool(25 < length < 170 and width < 35 and ctr[2] > 2))


def list_blobs(fr, colors=None, frame=None, region=250.0, specs=None, min_area=MIN_AREA, dedupe_px=12.0):
    """Every colored blob near the paper (|x|,|y| < region), largest first per color, overlaps removed."""
    specs = specs or PALETTE
    colors = colors or list(specs)
    color, depth, K = frame if frame is not None else grab(avg=5)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    out = []
    for name in colors:
        for b in color_blob(color, depth, K, every=True, **blob_kw(specs[name])):
            if b["area"] <= min_area:
                continue
            c = fr.plane(b["p"])
            if abs(c[0]) < region and abs(c[1]) < region and c[2] < 300:
                out.append(describe(fr, b, hsv, K, name))
    out.sort(key=lambda d: -d["area"])
    kept = []
    for d in out:  # overlapping hue bands (pink/orange): the larger blob keeps the pixels
        if all(np.hypot(d["px"][0] - k["px"][0], d["px"][1] - k["px"][1]) > dedupe_px for k in kept):
            kept.append(d)
    return kept


def marker_pose(fr, mode=None, frame=None):
    """The (largest) tracked-color marker lying on the table (h < 40)."""
    spec = mode_spec(mode)
    lying = [d for d in list_blobs(fr, ["m"], frame, specs={"m": spec}) if d["on_table"]]
    if not lying:
        raise RuntimeError("no marker on the table")
    return lying[0]


def _cloud(fr, avg=5, max_pts=None):
    c, d, K = grab(avg=avg)
    v, u = np.nonzero(d > 0)
    if max_pts and len(u) > max_pts:
        sel = np.random.default_rng(0).choice(len(u), max_pts, replace=False)
        u, v = u[sel], v[sel]
    return fr.plane(backproject(u, v, d[v, u], K)), cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[v, u]


def jaws(fr, marker_xy, radius=90, hmin=12, hmax=90, avg=5):
    """White jaw fingertips above the paper near a lying marker (noisy). 2-means split on xy."""
    Q, hsv = _cloud(fr, avg)
    white = (hsv[:, 1] < 60) & (hsv[:, 2] > 110)
    sel = white & (np.linalg.norm(Q[:, :2] - marker_xy, axis=1) < radius) & (Q[:, 2] > hmin) & (Q[:, 2] < hmax)
    pts = Q[sel]
    if len(pts) < 10:
        return None
    lo = np.percentile(pts[:, 2], 3)
    pts = pts[pts[:, 2] < lo + 25]  # lowest 25 mm band = fingertip region
    xy = pts[:, :2]
    a, b = xy[np.argmin(xy[:, 0])], xy[np.argmax(xy[:, 0])]
    for _ in range(20):
        lab = np.linalg.norm(xy - a, axis=1) < np.linalg.norm(xy - b, axis=1)
        a, b = xy[lab].mean(0), xy[~lab].mean(0)
    return dict(n=len(pts), lowest_h=float(lo), f1=a, f2=b, center=(a + b) / 2, gap=float(np.linalg.norm(a - b)),
                close_dir=(b - a) / np.linalg.norm(b - a))


def fingertips(fr, marker_xy, radius=70, avg=5):
    """Lowest non-orange points of the gripper near a lying marker (depth, noisy)."""
    Q, hsv = _cloud(fr, avg, 400000)
    orange = ((hsv[:, 0] >= 170) | (hsv[:, 0] <= 12)) & (hsv[:, 1] > 120)
    near = (np.linalg.norm(Q[:, :2] - marker_xy, axis=1) < radius) & (Q[:, 2] > 4) & (Q[:, 2] < 150) & ~orange
    pts = Q[near]
    if len(pts) < 30:
        return None
    low = pts[pts[:, 2] < np.percentile(pts[:, 2], 5)]
    return dict(n=len(pts), min_h=float(np.percentile(pts[:, 2], 2)), low_xy=low[:, :2].mean(0),
                low_spread=np.ptp(low[:, :2], 0))


def grasp_alignment(fr, mode=None):
    """Jaw center / closing direction vs the lying marker, in base radial/tangential terms."""
    m = marker_pose(fr, mode)
    j = jaws(fr, m["xyh"][:2])
    if j is None:
        raise RuntimeError("jaws not found")
    rel = m["xyh"][:2] - fr.BASE
    radial = rel / np.linalg.norm(rel)
    tang = np.array([-radial[1], radial[0]])
    off = j["center"] - m["xyh"][:2]
    return dict(marker=m["xyh"].round(1).tolist(), marker_tip_vs_radial=round(m["tip_vs_radial"], 1),
                jaw_center=j["center"].round(1).tolist(), jaw_lowest_h=round(j["lowest_h"], 1), gap=round(j["gap"], 1),
                offset_radial=round(float(off @ radial), 1), offset_tang=round(float(off @ tang), 1),
                closedir_vs_markeraxis_deg=round(angdiff(j["close_dir"], m["axis"]), 1), n=j["n"])
