"""Markers in the case / on the table: per-color blobs with plane pose, brightness and a reachability flag.

Hue bands (OpenCV 0..179, min saturation) are the field case_state.py ones plus orange (track.PALETTE). Pink
(140..172) and orange (170..14) overlap at 170..172; overlapping detections keep the larger blob.
Reach from BASE (pan axis, frame.json): jaws down to r ~250 mm at table height, jaws tilted 35-50 deg outward
to r ~300, nothing beyond ~300-310.
"""
import cv2
import numpy as np

from .rgbd import color_blob, grab

COLORS = {"green": ((35, 80), 80), "pink": ((140, 172), 60), "blue": ((95, 125), 120), "yellow": ((20, 34), 80),
          "orange": ((170, 14), 185)}
REGION = ((-200, 150), (-200, 60))   # plane x, y window of the case area
H_BAND = (3.0, 60.0)                 # blob-center height window (mm)
DOWN_R, REACH_R, MAX_R = 250.0, 300.0, 310.0
DARK_V = 50.0                        # mean image V below this: hue masks (min V 80) fail; turn the light on


def reach_class(r):
    return "down" if r <= DOWN_R else "tilted" if r <= REACH_R else "out"


def describe(fr, b, hsv, name, base=None):
    q = fr.plane(b["pts"])
    xyh = np.median(q, 0)
    ax = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
    ax2 = ax[:2] / max(np.linalg.norm(ax[:2]), 1e-9)
    proj = (q[:, :2] - xyh[:2]) @ ax2
    base = fr.BASE if base is None else np.asarray(base, float)
    rel = xyh[:2] - base
    r = float(np.hypot(*rel))
    u, v = b["uv"]
    return dict(color=name, c=xyh, axis=ax, axis2=ax2, area=b["area"], px=b["px"],
                length=float(np.percentile(proj, 98) - np.percentile(proj, 2)),
                brightness=float(hsv[v, u, 2].mean()), r=r, ang=float(np.degrees(np.arctan2(rel[1], rel[0]))),
                reachable=bool(r <= REACH_R), reach=reach_class(r))


def scan(fr=None, frame=None, region=REGION, colors=None, h_band=H_BAND, min_area=150, all_blobs=False,
         avg=5, dedupe_px=12.0, base=None):
    """dict(markers={color: info} (or {color: [info, ...]} with all_blobs), brightness=mean V, dark=bool, color=img).

    Without all_blobs each color keeps its largest qualifying blob (area > min_area inside region and h_band)."""
    if fr is None:
        from .frame import Frame
        fr = Frame.load()
    img, depth, K = frame if frame is not None else grab(avg=avg)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (x0, x1), (y0, y1) = region
    found = []
    for name in colors or list(COLORS):
        hue, sat = COLORS[name]
        for b in color_blob(img, depth, K, hue=hue, every=True, min_sat=sat, max_mm=900):
            if b["area"] <= min_area:
                continue
            c = fr.plane(b["p"])
            if x0 < c[0] < x1 and y0 < c[1] < y1 and h_band[0] < c[2] < h_band[1]:
                found.append(describe(fr, b, hsv, name, base))
    found.sort(key=lambda d: -d["area"])
    kept = []
    for d in found:
        if all(np.hypot(d["px"][0] - k["px"][0], d["px"][1] - k["px"][1]) > dedupe_px for k in kept):
            kept.append(d)
    out = {}
    for d in kept:
        if all_blobs:
            out.setdefault(d["color"], []).append(d)
        else:
            out.setdefault(d["color"], d)
    V = float(hsv[..., 2].mean())
    return dict(markers=out, brightness=V, dark=V < DARK_V, color=img)


def markers(fr=None, **kw):
    return scan(fr, **kw)["markers"]
