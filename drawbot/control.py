"""Closed-loop pen control over raw joints: u = (pan ID1, elbow ID3, wrist ID4); shoulder ID2 held.

Output y = tracked marker barrel (x, y, h) in the paper-plane frame (mm). y ~= y0 + J du with a
central-difference Jacobian (optionally Broyden-refined). State (J, contact_h) in draw2_state.json.
"""
import numpy as np

from . import arm, config, load, save, track

STATE = "draw2_state.json"


class Ctl2:
    blend = 0.0     # weight of the model prediction in the measurement (smooths depth noise)
    bounds = None   # optional per-joint (lo, hi) around the drawing site; tighter than lim
    last_y = None
    rejects = 0

    def __init__(self, fr=None, cfg=None, measure_fn=None, move_fn=None, joints_fn=None, state=STATE):
        cfg = cfg or config()
        self.ids = tuple(int(i) for i in cfg["ids"])
        self.lim = {int(k): tuple(v) for k, v in cfg["lim"].items()}
        self.speed = cfg.get("speed", 4)
        self.state = state
        self._move = move_fn or arm.move
        self._joints = joints_fn or arm.joints
        if measure_fn is None:
            from .frame import Frame
            fr = fr or Frame.load()
            mode = cfg.get("track")
            measure_fn = lambda **kw: track.measure(fr, mode=mode, **kw)
        self._measure = measure_fn
        t = self._joints()[0]
        self.a2 = t[2]
        self.u = np.array([t[i] for i in self.ids], float)
        st = load(state, {}) if state else {}
        self.J = np.array(st["J"]) if st.get("J") is not None else None
        self.contact_h = st.get("contact_h")

    def save(self):
        if self.state:
            save(self.state, {**load(self.state, {}), "J": None if self.J is None else self.J.tolist(),
                              "contact_h": self.contact_h})

    def clip(self, u):
        u = np.round(np.asarray(u, float))
        for k, i in enumerate(self.ids):
            u[k] = np.clip(u[k], *self.lim[i])
            if self.bounds is not None:
                u[k] = np.clip(u[k], *self.bounds[k])
        return u

    def command(self, u, speed=None):
        u = self.clip(u)
        self._move({i: u[k] for k, i in enumerate(self.ids)}, speed=speed or self.speed, settle=0.3)
        self.u = u

    def measure(self, **kw):
        return self._measure(**kw)

    def y(self, avg=3, prior=None):
        prior = prior if prior is not None else self.last_y
        y = np.asarray(self._measure(avg=avg, prior=prior)["xyh"], float)
        self.last_y = y
        return y

    def solve(self, dy):
        if self.J is None:
            raise RuntimeError("no Jacobian: run `drawbot_cli.py jacobian --yes` first")
        return np.linalg.lstsq(self.J, np.asarray(dy, float), rcond=None)[0]

    def estimate(self, steps=(35, 40, 40)):
        """Central-difference Jacobian (mm per tick) around the current u; returns to u afterwards."""
        u0 = self.u.copy()
        J = np.zeros((3, len(self.ids)))
        for j, step in enumerate(steps):
            ys = []
            for sgn in (+1, -1):
                du = np.zeros(len(self.ids)); du[j] = sgn * step
                self.command(u0 + du)
                ys.append(np.mean([self.y() for _ in range(2)], 0))
            J[:, j] = (ys[0] - ys[1]) / (2 * step)
            self.command(u0)
        self.J = J
        self.save()
        return J

    def y_checked(self, y_pred, max_jump=20.0):
        """Measurement with outlier rejection against the model prediction (xy distance)."""
        for _ in range(3):
            y = self.y(prior=y_pred)
            if y_pred is None or np.linalg.norm((y - y_pred)[:2]) < max_jump:
                return y, True
        return np.asarray(y_pred, float), False

    def servo(self, target, tol=1.5, iters=6, freeze_h=False, gain=0.6, max_step=60, verbose=False, learn=False):
        """Drive the barrel to target (x, y, h). freeze_h: pen in contact, ignore h error (and h learning)."""
        target = np.asarray(target, float)
        y, _ = self.y_checked(self.last_y, max_jump=40.0)
        for k in range(iters):
            err = target - y
            if freeze_h:
                err[2] = 0.0
            if np.linalg.norm(err) < tol:
                break
            du = np.clip(self.solve(gain * err), -max_step, max_step)
            u_prev, y_prev = self.u.copy(), y
            self.command(self.u + du)
            dun = self.u - u_prev
            y_pred = y_prev + self.J @ dun
            y, ok = self.y_checked(y_pred)
            y = self.blend * y_pred + (1 - self.blend) * y
            if not ok:
                self.rejects += 1
                if self.rejects > 5:
                    raise RuntimeError("too many rejected measurements")
            dy = y - y_prev
            if learn and ok and np.linalg.norm(dun) > 6:
                if freeze_h:
                    dy[2] = self.J[2] @ dun
                self.J += 0.5 * np.outer(dy - self.J @ dun, dun) / (dun @ dun)
            if verbose:
                print(f"    it {k} y {y.round(1).tolist()} err {np.linalg.norm((target - y)[:2 if freeze_h else 3]):.1f}"
                      f" u {self.u.astype(int).tolist()}", flush=True)
        self.last_y = y
        self.save()
        return y
