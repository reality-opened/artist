"""Paper-plane frame: x/y in mm on the paper plane (origin = paper center), h = height above the paper.

frame.json: {"ctr","e1","e2","n","d"} in camera mm (e1 x e2 = n, n toward the camera, n.p = d on the
paper), "base" = robot pan axis in plane xy, "pan_ang" = {pan tick: plane angle deg of the arm}.
Optional: "pan_ref" (tick key in pan_ang, default 1967), "ticks_per_deg" (default 250/21.45),
"pan_offset" (default +10 ticks, from the field fit).
"""
import numpy as np

from . import load, save


class Frame:
    def __init__(self, ctr, e1, e2, n, d, base=None, pan_ang=None, **extra):
        self.CTR, self.E1, self.E2, self.N = (np.asarray(v, float) for v in (ctr, e1, e2, n))
        self.D = float(d)
        self.BASE = None if base is None else np.asarray(base, float)
        self.pan_ang = {str(k): float(v) for k, v in (pan_ang or {}).items()}
        self.extra = extra

    @classmethod
    def load(cls, name="frame.json"):
        return cls(**load(name))

    def to_dict(self):
        out = dict(ctr=self.CTR.tolist(), e1=self.E1.tolist(), e2=self.E2.tolist(), n=self.N.tolist(), d=self.D,
                   **self.extra)
        if self.BASE is not None:
            out["base"] = self.BASE.tolist()
        if self.pan_ang:
            out["pan_ang"] = self.pan_ang
        return out

    def save(self, name="frame.json"):
        save(name, self.to_dict())

    def plane(self, p):
        """Camera mm (..., 3) -> plane (x, y, h)."""
        p = np.asarray(p, float)
        q = p - self.CTR
        return np.stack([q @ self.E1, q @ self.E2, p @ self.N - self.D], -1)

    def to_cam(self, xyh):
        """Inverse of plane(): p = CTR + x E1 + y E2 + (h - (N.CTR - D)) N."""
        xyh = np.asarray(xyh, float)
        x, y, h = xyh[..., 0:1], xyh[..., 1:2], xyh[..., 2:3]
        return self.CTR + x * self.E1 + y * self.E2 + (h - (self.N @ self.CTR - self.D)) * self.N

    def polar(self, xy):
        """(r mm, angle deg) of plane xy about the base axis."""
        rel = np.asarray(xy, float)[:2] - self.BASE
        return float(np.hypot(*rel)), float(np.degrees(np.arctan2(rel[1], rel[0])))

    def pan_for_angle(self, ang):
        """Pan ticks that point the arm at plane angle `ang` (deg), from the pan-circle fit."""
        ref = str(self.extra.get("pan_ref", 1967))
        tpd = self.extra.get("ticks_per_deg", 250 / 21.45)
        return int(round(int(ref) + self.extra.get("pan_offset", 10) - (ang - self.pan_ang[ref]) * tpd))


def fit_from_paper(P):
    """Frame from rgbd.paper(): origin = corner mean, e1 = paper edge nearest the camera's -x (image left)
    projected into the plane, e2 = n x e1 (toward the camera / image bottom). Base axis must be re-fit."""
    n, d, corners = P["normal"], P["d"], np.asarray(P["corners"], float)
    guess = np.array([-1.0, 0, 0]) - n * (n @ np.array([-1.0, 0, 0]))
    e1 = guess / np.linalg.norm(guess)
    if len(corners) == 4:
        edges = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
        edges = [e - n * (n @ e) for e in edges]
        edges = [e / np.linalg.norm(e) for e in edges]
        best = max(edges, key=lambda e: abs(e @ e1))
        e1 = best if best @ e1 > 0 else -best
    e2 = np.cross(n, e1)
    ctr = corners.mean(0)
    ctr = ctr - n * (n @ ctr - d)
    return Frame(ctr, e1, e2, n, d)


def fit_circle(xy):
    """Least-squares (Kasa) circle through >= 3 points: center (2,), radius, rms residual."""
    xy = np.asarray(xy, float)
    A = np.c_[xy, np.ones(len(xy))]
    b = -(xy ** 2).sum(1)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    r = float(np.sqrt(c @ c - F))
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(xy - c, axis=1) - r) ** 2)))


def base_from_pan(samples):
    """samples: [(pan_tick, plane_xy)] of the tracked marker at >= 3 pans (other joints fixed).

    Circle center = BASE (pan axis in plane xy); returns keys to merge into frame.json. The field frame
    used pan_offset=10 (empirical); a fresh fit starts at 0.
    """
    ticks = np.array([s[0] for s in samples], float)
    xy = np.array([s[1][:2] for s in samples], float)
    c, r, rms = fit_circle(xy)
    ang = np.degrees(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
    ang = np.degrees(np.unwrap(np.radians(ang)))
    slope = np.polyfit(ticks, ang, 1)[0]  # deg per tick (negative: pan up -> angle down)
    return dict(base=c.tolist(), radius=r, rms=rms, pan_ang={str(int(t)): float(a) for t, a in zip(ticks, ang)},
                pan_ref=int(np.sort(ticks)[len(ticks) // 2]), pan_offset=0, ticks_per_deg=float(-1 / slope))
