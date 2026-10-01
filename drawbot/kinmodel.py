"""Depth-calibrated kinematic model: URDF FK/IK in the paper-plane frame (mm), plus a sag-correcting goto.

Model ($DRAWBOT_STATE/kin_model.json, from `drawbot_cli.py kin-fit`; see calib.py):
    p_plane = R_PB @ (fk_urdf(q_model)[tool] * 1000) + T_PB
    q_model_j = (ticks_j - MID_j) * 2pi/4095;  q_model_j = s_j * q_model_j + o_j for ID2..ID4
    x = [rotvec R_PB (3), T_PB mm (3), offsets o2..o4 rad (3), tool point in the gripper frame m (3)]
The tool point is where the tracked orange marker sat in the gripper during calibration. `tool=None` means the
TCP = gripper_frame_link origin = fixed-jaw tip. The fitted rms is ~4 mm (52 samples).

fk(ticks5) -> (xyh mm, jaw axis (gripper-frame z, pointing out of the jaws), full rotation), all plane frame.
ik(target, axis, roll, seed) -> IKResult(q = ticks for IDs 1..4, res = position residual mm, axis, axis_err deg);
roll (ID5) is kept fixed. Joint window WIN: ID2 >= 2030 (shoulder stop), ID3 <= 2080 (elbow fold stop),
ID4 1730..3770.
goto() = IK, command, then correct joint sag from the *measured* joint positions (not the camera).

Wrist roll (ID5) was 3746 for every calibration sample, so the roll zero is not calibrated: tool/grasp points
are only trustworthy near roll 3746 (and the 180-degree flip 1698, up to the symmetric jaw geometry).
"""
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import REPO, config, load

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from kinematics import JOINTS, Kinematics  # noqa: E402  (repo-root module; URDF path lives there)

MODEL = "kin_model.json"
TICK_RAD = 2 * np.pi / 4095  # as kin_fit.py / kinematics.ticks_to_radians
CALIBRATION = Path("/Users/zhangbocheng/.cache/huggingface/lerobot/calibration/robots/so_follower/exp23_follower.json")
# (range_min + range_max) / 2 of exp23_follower.json; used when the calibration file is not available.
MID_DEFAULT = np.array([1877.0, 2047.5, 2047.5, 2750.0, 2047.5])
WIN = {1: (800, 3300), 2: (2030, 3950), 3: (900, 2080), 4: (1730, 3770)}
SEEDS = [(1952, 3250, 1550, 2330), (1952, 3450, 1550, 2130), (1952, 3000, 1600, 3500), (1952, 3700, 1500, 1900),
         (1952, 3100, 1300, 2600), (1952, 2900, 1800, 3000)]
ROLL_HOME = 3746
DOWN = (0.0, 0.0, -1.0)
FIELD_BASE = (-242.58124097476326, -11.692911146644937)
# One field observation (2026-09-26, teleop log): jaws down at pan 2164, roll 4080 (joints 2164/2870/1550/3734,
# gripper closed on a marker at 780). Model TCP vs the grasped marker center measured by depth, plane mm.
GRASP_OBS = dict(q=(2164, 2990, 1550, 3614), roll=4080, axis=DOWN, tcp=(-108.0, -51.6, 22.0),
                 grasp=(-109.4, -39.9, 29.0))


def mid_from_calibration(p=CALIBRATION):
    try:
        cal = json.loads(Path(p).read_text())
        return np.array([(cal[n]["range_min"] + cal[n]["range_max"]) / 2 for n in JOINTS], float)
    except (OSError, KeyError, ValueError):
        return MID_DEFAULT.copy()


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return w * ang


def _rot_batch(axis, ang):
    """Rotations about a fixed unit axis by angles ang (N,) -> (N, 3, 3)."""
    x, y, z = axis
    Kx = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


class Chain:
    """URDF chain base_link -> gripper_frame_link, batched over joint vectors (same math as Kinematics.fk)."""

    def __init__(self, kin=None, link="gripper_frame_link"):
        kin = kin or Kinematics()
        chain = []
        while link in kin.by_child:
            j = kin.by_child[link]
            chain.append(j)
            link = j["parent"]
        fixed = lambda j: j["kind"] == "fixed"
        self.steps = [(j["origin"], None if fixed(j) else JOINTS.index(j["name"]),
                       None if fixed(j) else j["axis"] / np.linalg.norm(j["axis"])) for j in reversed(chain)]
        # Closing direction of the jaws in the gripper frame: perpendicular to the jaw axis (z) and the hinge.
        g, f = kin.joints["gripper"], kin.joints["gripper_frame_joint"]
        hinge = f["origin"][:3, :3].T @ (g["origin"][:3, :3] @ g["axis"])
        c = np.cross([0.0, 0.0, 1.0], hinge)
        self.close_local = c / np.linalg.norm(c)

    def fk(self, Q):
        """Q (..., 5) radians -> (..., 4, 4) poses of gripper_frame_link in base_link (m)."""
        Q = np.asarray(Q, float)
        shape = Q.shape[:-1]
        Q = Q.reshape(-1, 5)
        T = np.broadcast_to(np.eye(4), (len(Q), 4, 4)).copy()
        for origin, idx, axis in self.steps:
            T = T @ origin
            if idx is not None:
                T[:, :3, :3] = T[:, :3, :3] @ _rot_batch(axis, Q[:, idx])
        return T.reshape(shape + (4, 4))


_CHAIN = None


def chain():
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = Chain()
    return _CHAIN


def model_q(ticks, signs, offs, mid=None):
    """Raw ticks (..., 5) -> URDF radians with signs/offsets on ID2..ID4 (kin_fit.py convention)."""
    mid = MID_DEFAULT if mid is None else mid
    q = (np.asarray(ticks, float) - mid) * TICK_RAD
    q[..., 1:4] = q[..., 1:4] * np.asarray(signs, float) + np.asarray(offs, float)
    return q


class IKResult(NamedTuple):
    q: np.ndarray        # ticks for IDs 1..4 (int)
    res: float           # position residual, mm
    axis: np.ndarray     # achieved jaw axis (plane frame)
    axis_err: float      # angle to the requested axis, deg (0 when no axis was requested)

    def ok(self, max_res=4.0, max_axis_deg=8.0):
        return self.res <= max_res and self.axis_err <= max_axis_deg


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class KinModel:
    def __init__(self, signs, x, mid=None, base=None, grasp_tool=None, **extra):
        self.signs = np.asarray(signs, float)
        self.x = np.asarray(x, float)
        self.mid = mid_from_calibration() if mid is None else np.asarray(mid, float)
        self.R_PB, self.T_PB = rodrigues(self.x[:3]), self.x[3:6]
        self.offs, self.TOOL = self.x[6:9], self.x[9:12]
        self.base = np.asarray(FIELD_BASE if base is None else base, float)[:2]
        self._grasp_tool = None if grasp_tool is None else np.asarray(grasp_tool, float)
        self.extra = extra
        self.chain = chain()

    @classmethod
    def load(cls, name=MODEL):
        d = load(name)
        if d.get("base") is None:
            try:
                d["base"] = load("frame.json")["base"]
            except (FileNotFoundError, KeyError):
                pass
        return cls(**d)

    def to_dict(self):
        out = dict(signs=self.signs.tolist(), x=self.x.tolist(), mid=self.mid.tolist(), **self.extra)
        if self._grasp_tool is not None:
            out["grasp_tool"] = self._grasp_tool.tolist()
        return out

    # --- forward kinematics ------------------------------------------------------------------------
    def fk_batch(self, ticks, tool=None):
        """ticks (N, 5) -> points (N, 3) plane mm, jaw axes (N, 3), rotations (N, 3, 3)."""
        T = self.chain.fk(model_q(np.atleast_2d(ticks), self.signs, self.offs, self.mid))
        p = T[:, :3, 3] if tool is None else T[:, :3, :3] @ np.asarray(tool, float) + T[:, :3, 3]
        R = self.R_PB @ T[:, :3, :3]
        return p * 1000 @ self.R_PB.T + self.T_PB, R[:, :, 2], R

    def fk(self, ticks, tool=None):
        p, a, R = self.fk_batch(np.asarray(ticks, float)[None], tool)
        return p[0], a[0], R[0]

    def close_dir(self, ticks):
        """Jaw closing direction (unit, plane frame) at ticks5."""
        return self.fk(ticks)[2] @ self.chain.close_local

    # --- inverse kinematics ------------------------------------------------------------------------
    def pan_seed(self, target):
        ang = np.degrees(np.arctan2(target[1] - self.base[1], target[0] - self.base[0]))
        return 1977 - (ang - 0.7) * 250 / 21.45

    def ik(self, target, axis=None, roll=ROLL_HOME, seed=None, tool=None, w_axis=60.0, iters=80):
        """Multi-start: `seed` (if given) plus SEEDS with the pan aimed at the target; keep the best position fit."""
        pan = self.pan_seed(target)
        starts = ([tuple(seed)] if seed is not None else []) + [(pan,) + s[1:] for s in SEEDS]
        best = None
        for st in starts:
            r = self._ik(target, axis, roll, st, tool, w_axis, iters)
            if best is None or r.res < best.res:
                best = r
            if best.res < 0.5:
                break
        return best

    def _ik(self, target, axis, roll, seed, tool, w_axis, iters):
        """Damped least squares on (pan, ID2, ID3, ID4); axis = desired jaw direction (unit, plane frame)."""
        q = np.array(list(seed)[:4] + [roll], float)
        target = np.asarray(target, float)
        ax = None if axis is None else unit(axis)
        lo = np.array([WIN[i][0] for i in range(1, 5)])
        hi = np.array([WIN[i][1] for i in range(1, 5)])
        D = np.zeros((9, 5))
        for i in range(4):
            D[1 + 2 * i, i], D[2 + 2 * i, i] = -2.0, 2.0

        def errs(Q):
            p, a, _ = self.fk_batch(Q, tool)
            e = target - p
            return e if ax is None else np.hstack([e, w_axis * (ax - a)])

        for _ in range(iters):
            E = errs(q + D)  # rows: q, then q - d, q + d per joint
            e = E[0]
            if np.linalg.norm(e) < 0.3:
                break
            J = np.stack([(E[1 + 2 * i] - E[2 + 2 * i]) / 4.0 for i in range(4)], 1)  # d(p)/dq
            dq = np.linalg.solve(J.T @ J + 1e-2 * np.eye(4), J.T @ e)
            q[:4] = np.clip(q[:4] + np.clip(dq, -150, 150), lo, hi)
        p, a, _ = self.fk(q, tool)
        aerr = 0.0 if ax is None else float(np.degrees(np.arccos(np.clip(a @ ax, -1, 1))))
        return IKResult(q[:4].round().astype(int), float(np.linalg.norm(target - p)), a, aerr)

    def residual(self, target, q4, roll, tool=None):
        """Position residual (mm) of the rounded joint solution."""
        return float(np.linalg.norm(np.asarray(target, float) - self.fk(list(q4) + [roll], tool)[0]))

    # --- motion ------------------------------------------------------------------------------------
    def goto(self, target, axis=None, roll=None, seed=None, tool=None, speed=5, rounds=3, tol=6, verbose=True,
             w_axis=25.0, max_res=4.0, move_fn=None, joints_fn=None, out=print):
        """IK then command; correct sag using measured joint positions. Returns (fk of measured joints, q)."""
        from . import arm
        move_fn = move_fn or arm.move
        joints_fn = joints_fn or arm.joints
        t, p, m = joints_fn()
        roll = p[5] if roll is None else roll
        seed = [p[i] for i in range(1, 5)] if seed is None else seed
        r = self.ik(target, axis, roll, seed, tool, w_axis=w_axis)
        q = r.q
        pos_res = self.residual(target, q, roll, tool)
        if pos_res > max_res:
            raise RuntimeError(f"IK position residual {pos_res:.1f} mm for {np.round(target, 1).tolist()}"
                               f" (axis {np.round(r.axis, 2).tolist()})")
        cmd = {i: int(q[i - 1]) for i in range(1, 5)}
        for k in range(rounds):
            move_fn(cmd, speed=speed, settle=0.4)
            t, p, m = joints_fn()
            err = {i: int(q[i - 1]) - p[i] for i in range(1, 5)}
            if verbose:
                out(f"   goto round {k}: want {q.tolist()} pos {[p[i] for i in range(1, 5)]} err {list(err.values())}")
            if max(abs(v) for v in err.values()) <= tol:
                break
            cmd = {i: int(np.clip(cmd[i] + 0.8 * err[i], *WIN[i])) for i in range(1, 5)}
        t, p, m = joints_fn()
        return self.fk([p[i] for i in range(1, 6)], tool)[0], q

    # --- grasp point -------------------------------------------------------------------------------
    def grasp_tool_from_obs(self, obs=GRASP_OBS):
        """Gripper-frame offset (m) of the grasp center from one observation: (grasp - tcp) rotated into the
        gripper frame with the model rotation at the observed pose (reconstructed by IK on the TCP)."""
        r = self.ik(obs["tcp"], obs.get("axis"), obs["roll"], seed=obs.get("q"))
        R = self.fk(list(r.q) + [obs["roll"]])[2]
        return R.T @ (np.asarray(obs["grasp"], float) - np.asarray(obs["tcp"], float)) / 1000.0

    @property
    def grasp_tool(self):
        """Grasp center in the gripper frame (m): config grasp_tool_mm > kin_model.json grasp_tool > GRASP_OBS."""
        cfg = config().get("grasp_tool_mm")
        if cfg is not None:
            return np.asarray(cfg, float) / 1000.0
        if self._grasp_tool is None:
            self._grasp_tool = self.grasp_tool_from_obs()
        return self._grasp_tool


_MODEL = None


def model(reload=False):
    """The model from $DRAWBOT_STATE/kin_model.json (cached)."""
    global _MODEL
    if _MODEL is None or reload:
        _MODEL = KinModel.load()
    return _MODEL


def fk(ticks, tool=None):
    return model().fk(ticks, tool)


def ik(target, axis=None, roll=ROLL_HOME, seed=None, tool=None, w_axis=60.0, iters=80):
    return model().ik(target, axis, roll, seed, tool, w_axis, iters)


def goto(target, **kw):
    return model().goto(target, **kw)


def tool_arg(spec, m=None):
    """CLI tool spec -> gripper-frame point (m) or None: None/[] -> TCP, ['marker'] or bare --tool -> fitted
    marker point, ['grasp'] -> grasp center, [x, y, z] -> mm in the gripper frame."""
    if spec is None:
        return None
    m = m or model()
    if len(spec) == 0 or spec == ["marker"]:
        return m.TOOL
    if spec == ["grasp"]:
        return m.grasp_tool
    if spec == ["tcp"]:
        return None
    if len(spec) == 3:
        return np.array([float(v) for v in spec]) / 1000.0
    raise ValueError(f"bad --tool {spec}: use marker | grasp | tcp | X Y Z (mm, gripper frame)")
