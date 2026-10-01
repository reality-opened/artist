"""Smiley face as pen strokes (mm, face frame: x right, y up, origin = face center).

One layer per marker color (one marker swap per layer), drawn in LAYERS order.
"""
import numpy as np

R = 35  # face radius, mm; draw_layer scales by site r_face / R


def arc(cx, cy, r, a0, a1, step_mm=2.0):
    n = max(8, int(abs(np.radians(a1 - a0)) * r / step_mm))
    a = np.radians(np.linspace(a0, a1, n + 1))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).round(2).tolist()


def circle(cx, cy, r):
    return arc(cx, cy, r, 90, 450)


LAYERS = [  # (marker name, BGR for render, strokes)
    ("orange", (30, 90, 240), [circle(0, 0, R)]),
    ("blue", (200, 110, 30), [circle(-12, 10, 4), circle(-12, 10, 2), circle(12, 10, 4), circle(12, 10, 2)]),
    ("green", (60, 180, 40), [arc(0, 2, 20, 205, 335)]),
    ("pink", (170, 110, 240), [circle(-24, 0, 4), circle(24, 0, 4)]),
    ("yellow", (40, 220, 245), [circle(0, 0, R - 3)]),  # inner ring; faint on white
]


def layers():
    return {m: s for m, _, s in LAYERS}


def render(path, scale=8, margin=10):
    import cv2
    size = int((2 * R + 2 * margin) * scale)
    img = np.full((size, size, 3), 255, np.uint8)
    to_px = lambda p: (int((p[0] + R + margin) * scale), int((R + margin - p[1]) * scale))
    for _, bgr, strokes in LAYERS:
        for s in strokes:
            cv2.polylines(img, [np.array([to_px(p) for p in s], np.int32)], False, bgr,
                          thickness=int(1.5 * scale), lineType=cv2.LINE_AA)
    cv2.imwrite(path, img)
