"""Gemini RGB-D helpers: fetch aligned color+depth, back-project, fit the paper plane, color blobs.

Read-only HTTP: color :8766/frame.jpg, depth :8767/depth.png (uint16 mm aligned to color),
:8767/intrinsics, wrist cam :8765/frame.jpg.
"""
import json
import os
import urllib.request
import warnings

import cv2
import numpy as np

HOST = os.environ.get("DRAWBOT_CAM_HOST", "127.0.0.1")
COLOR_PORT, DEPTH_PORT, WRIST_PORT = 8766, 8767, 8765


def fetch(port, path, timeout=3):
    return urllib.request.urlopen(f"http://{HOST}:{port}/{path}", timeout=timeout).read()


def decode(buf, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), flags)


def median_depth(frames):
    """Per-pixel median of uint16 mm frames; 0 and >= 5 m readings are ignored. Returns float mm, 0 = none."""
    stack = [np.where((d > 0) & (d < 5000), d, np.nan).astype(np.float32) for d in frames]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        return np.nan_to_num(np.nanmedian(np.stack(stack), axis=0), nan=0.0)


def grab(avg=5):
    """Color BGR, depth mm (float, median of `avg` frames, 0 = none), intrinsics dict."""
    K = json.loads(fetch(DEPTH_PORT, "intrinsics"))
    color = decode(fetch(COLOR_PORT, "frame.jpg"))
    depth = median_depth([decode(fetch(DEPTH_PORT, "depth.png"), cv2.IMREAD_UNCHANGED) for _ in range(avg)])
    return color, depth, K


def wrist_frame():
    return decode(fetch(WRIST_PORT, "frame.jpg"))


def backproject(u, v, z, K):
    """Pixel(s) + depth mm -> camera-frame XYZ in mm (x right, y down, z forward)."""
    u, v, z = np.asarray(u, float), np.asarray(v, float), np.asarray(z, float)
    return np.stack([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z], -1)


def project(p, K):
    p = np.atleast_2d(p)
    return np.stack([K["fx"] * p[:, 0] / p[:, 2] + K["cx"], K["fy"] * p[:, 1] / p[:, 2] + K["cy"]], -1)


def hue_mask(hsv, hue, min_sat=150, min_val=80, max_sat=255):
    """hue=(lo, hi) in OpenCV units (0..179); lo > hi wraps through red."""
    h = hsv[..., 0]
    band = (h >= hue[0]) & (h <= hue[1]) if hue[0] <= hue[1] else (h >= hue[0]) | (h <= hue[1])
    return band & (hsv[..., 1] > min_sat) & (hsv[..., 1] <= max_sat) & (hsv[..., 2] > min_val)


def paper_mask(color):
    """Largest bright, low-saturation blob = the paper."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 170) & (hsv[..., 1] < 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise RuntimeError("no paper found")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == best).astype(np.uint8)


def fit_plane(pts, iters=300, thresh=3.0, seed=0):
    """RANSAC plane: returns (normal n, offset d, inlier mask) with n.p = d, n toward the camera (n_z < 0)."""
    rng = np.random.default_rng(seed)
    best, best_in = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.integers(0, len(pts), 3)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(pts @ n - n @ a) < thresh
        if inl.sum() > best_in:
            best, best_in = inl, inl.sum()
    q = pts[best]
    centroid = q.mean(0)
    # full_matrices=False: with >100k points the full U would be N x N (hundreds of GB).
    n = np.linalg.svd(q - centroid, full_matrices=False)[2][-1]
    if n[2] > 0:
        n = -n  # camera looks along +z; plane normal should point back toward it
    return n, float(n @ centroid), best


def paper(color=None, depth=None, K=None):
    if color is None:
        color, depth, K = grab()
    mask = paper_mask(color)
    inner = cv2.erode(mask, np.ones((15, 15), np.uint8))  # edge pixels have mixed depth
    v, u = np.nonzero(inner & (depth > 0))
    pts = backproject(u, v, depth[v, u], K)
    n, d, inl = fit_plane(pts)
    cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(cnt)  # shadows of the arm notch the mask; the hull fills them
    for eps in np.linspace(0.01, 0.1, 30):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(quad) <= 4:
            break
    # intersect corner rays with the plane (robust to missing depth at the edges)
    rays = backproject(quad[:, 0], quad[:, 1], np.ones(len(quad)), K)
    corners = rays * (d / (rays @ n))[:, None]
    resid = np.abs(pts[inl] @ n - d)
    return dict(normal=n, d=d, corners_px=quad, corners=corners, mask=mask,
                inlier_frac=float(inl.mean()), resid_mm=float(resid.std()), color=color, depth=depth, K=K)


def color_blob(color, depth, K, hue=(0, 12), max_mm=900, min_sat=150, min_mm=250, every=False,
               min_val=80, max_sat=255):
    """3D centroid (mm, camera frame) of the largest saturated blob in a hue band within min_mm..max_mm.

    every=True returns all blobs sorted by area (blobs < 100 px are dropped once one is found).
    """
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = (hue_mask(hsv, hue, min_sat, min_val, max_sat) & (depth > min_mm) & (depth < max_mm)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return [] if every else None
    blobs = []
    for idx in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        if stats[idx, cv2.CC_STAT_AREA] < 100 and blobs:
            break
        v, u = np.nonzero(lab == idx)
        pts = backproject(u, v, depth[v, u], K)
        blobs.append(dict(p=np.median(pts, 0), px=(float(u.mean()), float(v.mean())), area=int(len(u)),
                          pts=pts, uv=(u, v)))
        if not every:
            break
    return blobs if every else blobs[0]


def wrist_blob(img=None, hue=(170, 14), min_sat=110):
    """Wrist camera: largest orange blob centroid + axis angle (deg from image vertical)."""
    img = wrist_frame() if img is None else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = hue_mask(hsv, hue, min_sat, 80).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    v, u = np.nonzero(lab == k)
    X = np.stack([u, v], 1).astype(float)
    ax = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0]
    ang = (float(np.degrees(np.arctan2(ax[0], ax[1]))) + 90) % 180 - 90  # 0 = vertical in image
    edge = u.min() < 3 or v.min() < 3 or u.max() > img.shape[1] - 4 or v.max() > img.shape[0] - 4
    return dict(cx=float(u.mean()), cy=float(v.mean()), ang=ang, area=int(len(u)), edge=bool(edge),
                w=img.shape[1], h=img.shape[0])
