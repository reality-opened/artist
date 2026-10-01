"""drawbot kinematic model / calibration / manipulation tests. Synthetic data and mocks only: nothing is sent."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from drawbot import arm

try:
    from drawbot import calib, kinmodel, manip
    from kinematics import URDF
    HAVE_URDF = Path(URDF).exists()
except Exception:  # pragma: no cover - URDF lives outside the repo
    HAVE_URDF = False

FIELD_X = [-0.00928798806943167, 0.029294339838674188, 0.17266429279754375, -288.5961386153333, -19.09082721756577,
           3.3559952078085487, -1.7737701354305246, 1.533162093755808, -0.17624551135427705, 0.025793508730381276,
           -0.011173642318299593, -0.019180028470718202]
FIELD_FRAME = {"ctr": [-44.7, 67.23, 426.82], "e1": [-0.84033, 0.100997, -0.532582],
               "e2": [0.521402, 0.419329, -0.743171], "n": [0.148269, -0.902199, -0.405036], "d": -240.16,
               "base": [-242.58, -11.69], "pan_ang": {"1717": 22.454, "1967": 0.6689, "2217": -20.456}}
K = {"fx": 686.65, "fy": 686.52, "cx": 641.32, "cy": 359.03}
quiet = lambda *a, **k: None


def field_model():
    return kinmodel.KinModel([1, 1, 1], FIELD_X, mid=kinmodel.MID_DEFAULT, base=FIELD_FRAME["base"])


def polar(m, r, ang_deg, h):
    a = np.radians(ang_deg)
    return [m.base[0] + r * np.cos(a), m.base[1] + r * np.sin(a), h]


class SagArm:
    """Joints that sag: measured pos = commanded - sag (per joint); ID5 roll with load/lag behavior."""

    def __init__(self, q0=(1952, 3250, 1550, 2330, 3746), sag=None, roll_load=None, roll_stop=None):
        self.cmd = {i + 1: int(v) for i, v in enumerate(q0)}
        self.sag = sag or {}
        self.roll_load = roll_load or (lambda r: 20)
        self.roll_stop = roll_stop
        self.moves = []

    def pos(self, i):
        v = self.cmd[i] - self.sag.get(i, 0)
        if i == 5 and self.roll_stop is not None:
            v = max(v, self.roll_stop)
        return v

    def move(self, goal, speed=12, settle=0.5):
        arm.check_goal(goal)
        self.moves.append(dict(goal))
        self.cmd.update({int(k): int(v) for k, v in goal.items()})

    def joints(self):
        t = dict(self.cmd)
        p = {i: self.pos(i) for i in range(1, 6)}
        m = {str(i): dict(target=t[i], pos=p[i], load=0) for i in range(1, 6)}
        m["5"]["load"] = self.roll_load(p[5])
        return t, p, m


@unittest.skipUnless(HAVE_URDF, "SO101 URDF not available")
class KinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DRAWBOT_STATE"] = self.tmp.name
        arm.CTL = Path(self.tmp.name)
        arm.enable(False)
        kinmodel._MODEL = None
        self.m = field_model()

    def tearDown(self):
        kinmodel._MODEL = None
        self.tmp.cleanup()

    def test_chain_matches_urdf_fk(self):
        from kinematics import Kinematics
        kin = Kinematics()
        Q = np.random.default_rng(0).uniform(-2, 2, (10, 5))
        T = kinmodel.chain().fk(Q)
        for i in range(10):
            np.testing.assert_allclose(T[i], kin.fk(Q[i]), atol=1e-12)
        np.testing.assert_allclose(kinmodel.chain().close_local, [1, 0, 0], atol=1e-6)  # jaws close along x

    def synthetic(self, true, rolls, noise, seed=3):
        rng = np.random.default_rng(seed)
        ticks = []
        for k, (_, g) in enumerate(calib.poses()[::2]):  # 30 of the field poses, with joint jitter
            ticks.append([g[1], g[2], g[3], g[4], rolls[k % len(rolls)]] + np.r_[rng.integers(-15, 16, 4), 0])
        Q = np.array(ticks, float)
        P = self.observe(true, Q) + rng.normal(0, noise, (len(Q), 3)) if noise else self.observe(true, Q)
        return Q, P

    @staticmethod
    def observe(x, Q):
        return calib.points(np.asarray(Q, float), np.ones(3), x, kinmodel.MID_DEFAULT) @ kinmodel.rodrigues(x[:3]).T + x[3:6]

    def true_x(self):
        true = np.array(FIELD_X)
        true[6:9] += [0.05, -0.04, 0.03]             # offsets away from the field values
        true[9:12] = [0.020, -0.008, -0.025]          # a different tool point (m)
        return true

    def test_kin_fit_recovers_synthetic_model(self):
        """Two wrist rolls make every parameter observable: exact recovery from clean URDF samples."""
        true = self.true_x()
        Q, P = self.synthetic(true, (3746, 1698), noise=0.0)
        S = [dict(q=q.tolist(), cam=p.tolist()) for q, p in zip(Q, P)]
        res = calib.run(S, mid=kinmodel.MID_DEFAULT, save_model=True, out=quiet)
        x = np.array(res["x"])
        self.assertLess(res["rms"], 0.01)
        np.testing.assert_allclose(x[6:9], true[6:9], atol=np.radians(0.05))
        np.testing.assert_allclose(x[9:12] * 1000, true[9:12] * 1000, atol=0.1)
        np.testing.assert_allclose(x[3:6], true[3:6], atol=0.1)
        m = kinmodel.model(reload=True)             # saved into $DRAWBOT_STATE and loadable
        np.testing.assert_allclose(m.x, x)
        self.assertEqual(json.loads(Path(self.tmp.name, "kin_model.json").read_text())["n"], len(S))

    def test_kin_fit_noisy_single_roll_with_outliers(self):
        """Field-like set (one roll, 0.7 mm noise, 2 tracking outliers): outliers dropped, predictions recovered."""
        true = self.true_x()
        Q, P = self.synthetic(true, (3746,), noise=0.7)
        P[[4, 17]] += [40.0, -30.0, 25.0]
        S = [dict(q=q.tolist(), cam=p.tolist()) for q, p in zip(Q, P)]
        lines = []
        res = calib.run(S, mid=kinmodel.MID_DEFAULT, save_model=False, out=lines.append)
        self.assertEqual(res["dropped"], 2)
        self.assertLess(res["rms"], 1.5)
        self.assertTrue(any("degenerate" in str(l) for l in lines))
        x = np.array(res["x"])
        rng = np.random.default_rng(9)                # held-out poses near the calibrated set, same roll
        Qt = np.array([[g[1], g[2], g[3], g[4], 3746] + np.r_[rng.integers(-60, 61, 4), 0] for _, g in calib.poses()[1::2]])
        err = np.linalg.norm(self.observe(x, Qt) - self.observe(true, Qt), axis=1)
        self.assertLess(np.sqrt(np.mean(err ** 2)), 1.5)
        self.assertLess(err.max(), 3.0)

    def test_ik_round_trip_and_unreachable(self):
        m = self.m
        for r in (150, 190, 225):
            for ang in (-20, 0, 25):
                for h in (15, 35):
                    tgt = polar(m, r, ang, h)
                    for axis in (None, kinmodel.DOWN):
                        sol = m.ik(tgt, axis)
                        self.assertLess(m.residual(tgt, sol.q, kinmodel.ROLL_HOME), 1.0, (tgt, axis))
                        self.assertTrue(sol.ok())
                        self.assertTrue(all(kinmodel.WIN[i][0] <= sol.q[i - 1] <= kinmodel.WIN[i][1] for i in range(1, 5)))
        far = m.ik(polar(m, 420, 0, 20))
        self.assertGreater(far.res, 4.0)
        self.assertFalse(far.ok())
        down_far = m.ik(polar(m, 300, 0, 5), kinmodel.DOWN)  # jaws down ends ~250 at table height
        self.assertFalse(down_far.ok())

    def test_goto_corrects_sag_from_measured_joints(self):
        m = self.m
        sim = SagArm(sag={1: 4, 2: 40, 3: -25, 4: 12})
        tgt = polar(m, 200, 10, 40)
        got, q = m.goto(tgt, axis=kinmodel.DOWN, roll=3746, move_fn=sim.move, joints_fn=sim.joints, verbose=False)
        self.assertGreaterEqual(len(sim.moves), 2)
        self.assertEqual(sim.moves[0], {i: int(q[i - 1]) for i in range(1, 5)})
        self.assertTrue(all(abs(sim.pos(i) - q[i - 1]) <= 6 for i in range(1, 5)))
        self.assertGreater(sim.moves[-1][2], q[1])       # shoulder commanded above the solution to cancel sag
        self.assertLess(np.linalg.norm(got - tgt), 3.0)
        sim2 = SagArm()
        with self.assertRaises(RuntimeError):            # unreachable: raises before any command
            m.goto(polar(m, 420, 0, 20), move_fn=sim2.move, joints_fn=sim2.joints, verbose=False)
        self.assertEqual(sim2.moves, [])

    def test_grasp_tool_from_observation_and_override(self):
        m = self.m
        g = m.grasp_tool
        obs = kinmodel.GRASP_OBS
        r = m.ik(obs["tcp"], obs["axis"], obs["roll"], seed=obs["q"])
        np.testing.assert_allclose(m.fk(list(r.q) + [obs["roll"]], g)[0], obs["grasp"], atol=1.0)
        self.assertLess(abs(r.q[0] - 2164), 15)          # reconstructed pan matches the observation
        Path(self.tmp.name, "config.json").write_text(json.dumps({"grasp_tool_mm": [1, 2, 3]}))
        np.testing.assert_allclose(field_model().grasp_tool, [0.001, 0.002, 0.003])

    def test_pick_plan_modes(self):
        m = self.m
        pose = lambda r, ang, ax2=(0.0, 1.0): dict(c=polar(m, r, ang, 12.0), axis2=list(ax2))
        near = manip.pick_marker(pose(200, 5), m)
        self.assertTrue(near["ok"], near)
        self.assertEqual(near["mode"], "down")
        self.assertGreater(near["approach"]["xyh"][2], near["grasp"]["xyh"][2] + 15)
        self.assertGreater(near["lift"]["xyh"][2], near["grasp"]["xyh"][2] + 15)
        self.assertEqual(near["gripper"]["close"], 794 - 38)
        cd = np.array(near["close_dir"][:2])
        self.assertLess(abs(cd @ [0, 1]) / np.linalg.norm(cd), 0.1)   # jaws close across the marker
        self.assertTrue(manip.ROLL_WIN[0] <= near["roll"] <= manip.ROLL_WIN[1])
        self.assertGreaterEqual(near["tcp_h"], 0.0)
        far = manip.pick_marker(pose(275, 0), m)
        self.assertTrue(far["ok"], far)
        self.assertEqual(far["mode"], "tilted")
        self.assertTrue(35 <= far["tilt_deg"] <= 50)
        ax = np.array(far["axis"])
        radial = np.r_[(np.array(far["grasp"]["xyh"][:2]) - m.base), 0]
        self.assertGreater(ax @ radial, 0)                              # tilted outward, away from the base
        for leg in ("approach", "grasp", "lift"):
            q5 = [far[leg]["joints"][i] for i in range(1, 6)]
            self.assertLess(np.linalg.norm(m.fk(q5, np.array(far["tool_mm"]) / 1000)[0] - far[leg]["xyh"]), 4.0)
        out = manip.pick_marker(pose(330, 0), m)
        self.assertFalse(out["ok"])

    def test_execute_pick_sequence_on_mock_arm(self):
        m = self.m
        plan = manip.pick_marker(dict(c=polar(m, 200, 5, 12.0), axis2=[0.0, 1.0]), m)
        sim = SagArm(sag={2: 30, 3: -20})
        sent = []
        got = manip.execute_pick(plan, m, move_fn=sim.move, joints_fn=sim.joints, send_fn=sent.append, out=quiet)
        self.assertEqual(sent, [{"gripper": 1000}, {"gripper": 756}])
        self.assertEqual(sim.cmd[5], plan["roll"])
        self.assertLess(np.linalg.norm(got["lift"] - plan["lift"]["xyh"]), 4.0)
        self.assertLess(np.linalg.norm(got["grasp"] - plan["grasp"]["xyh"]), 4.0)
        self.assertFalse(Path(self.tmp.name, "cmds.jsonl").exists())
        with self.assertRaises(RuntimeError):
            manip.execute_pick(dict(ok=False, reason="x"), m, move_fn=sim.move, joints_fn=sim.joints)

    def test_cli_dry_runs_send_nothing(self):
        import drawbot_cli
        from drawbot import save
        save("kin_model.json", dict(signs=[1, 1, 1], x=FIELD_X, mid=kinmodel.MID_DEFAULT.tolist()))
        save("frame.json", FIELD_FRAME)
        st = {"t": 0, "moving": False, "motors": {str(i): dict(pos=p, target=p, load=0, temp=30)
                                                   for i, p in zip(range(1, 7), (1952, 3250, 1550, 2330, 3746, 760))}}
        Path(self.tmp.name, "status.json").write_text(json.dumps(st))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(drawbot_cli.main(["ik", "-60", "0", "20", "--down", "--tool", "grasp"]), 0)
            self.assertEqual(drawbot_cli.main(["ik", "200", "0", "10", "--down"]), 2)
            drawbot_cli.main(["goto", "-60", "0", "40", "--down"])
            drawbot_cli.main(["flip"])
            drawbot_cli.main(["poke", "-60", "0", "-60", "30", "20"])
            drawbot_cli.main(["fk"])
        self.assertIn("-> 1698", buf.getvalue())
        self.assertFalse(Path(self.tmp.name, "cmds.jsonl").exists())


class ManipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DRAWBOT_STATE"] = self.tmp.name
        arm.CTL = Path(self.tmp.name)
        arm.enable(False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_heightmap_on_synthetic_points(self):
        from drawbot import manip
        rng = np.random.default_rng(0)
        xy = rng.uniform([-60, -30], [60, 30], (40000, 2))
        h = rng.normal(0, 0.3, len(xy))
        box = (np.abs(xy[:, 0] - 20) < 12) & (np.abs(xy[:, 1]) < 12)
        h[box] += 30.0
        Q = np.c_[xy, h]
        Q = np.vstack([Q, [[0.0, 0.0, 500.0]]])          # above hmax: ignored
        xs, ys, g = manip.heightmap_points(Q, -60, 60, -30, 42, cell=6)
        self.assertEqual(g.shape, (12, 20))
        iy, ix = int((0 - -30) // 6), int((20 - -60) // 6)
        self.assertAlmostEqual(g[iy, ix], 30.0, delta=1.0)  # box top
        self.assertLess(abs(g[0, 0]), 1.0)                  # table
        self.assertTrue(np.isnan(g[-1]).all())              # y >= 30: no points
        self.assertIn("  30", manip.show_heightmap(xs, ys, g))

    def test_flip_guard_stops_on_load_and_lag(self):
        from drawbot import manip
        self.assertEqual(manip.flip_target(3746), 1698)
        self.assertEqual(manip.flip_target(1698), 3746)
        with self.assertRaises(ValueError):
            manip.flip_target(2500)
        self.assertEqual(manip.load_mag(1024 + 56), 56)
        # cable load rises below 2000 (raw register with the direction bit set)
        sim = SagArm(roll_load=lambda r: 1024 + int(max(0, 2000 - r) * 1.2))
        r = manip.flip_roll(1698, move_fn=sim.move, joints_fn=sim.joints, out=quiet)
        self.assertFalse(r["ok"])
        self.assertIn("load", r["reason"])
        trip = r["trace"][-1]["cmd"]
        self.assertEqual(trip, 1746)                       # 3746 - 20 * 100: load 1.2 * 254 > 300
        self.assertEqual(sim.moves[-1], {5: trip + 150})    # backed off toward the start
        self.assertEqual(r["roll"], trip + 150)
        self.assertTrue(all(abs(a[5] - b[5]) == 100 for a, b in zip(sim.moves[:-2], sim.moves[1:-1])))
        # mechanical stop: position stops following -> lag guard
        sim = SagArm(roll_stop=1800)
        r = manip.flip_roll(1698, move_fn=sim.move, joints_fn=sim.joints, out=quiet)
        self.assertFalse(r["ok"])
        self.assertIn("lag", r["reason"])
        # free roll: reaches the target exactly
        sim = SagArm()
        r = manip.flip_roll(1698, move_fn=sim.move, joints_fn=sim.joints, out=quiet)
        self.assertTrue(r["ok"])
        self.assertEqual(sim.cmd[5], 1698)
        with self.assertRaises(ValueError):
            manip.flip_roll(1200, move_fn=sim.move, joints_fn=sim.joints, out=quiet)

    def test_scene_scan_synthetic_frame(self):
        from drawbot import scene
        from drawbot.frame import Frame
        from drawbot.rgbd import project
        fr = Frame(**FIELD_FRAME)
        H, W = 720, 1280
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1)
        depth = fr.D / (rays @ fr.N)                      # table plane h = 0
        img = np.full((H, W, 3), 200, np.uint8)
        c = np.array([-60.0, -100.0, 10.0])               # a green marker lying 10 mm up, along plane y
        pts = fr.to_cam(np.array([[c[0] + dx, c[1] + dy, 10.0] for dx in np.linspace(-6, 6, 25)
                                  for dy in np.linspace(-25, 25, 100)]))
        px = np.round(project(pts, K)).astype(int)
        img[px[:, 1], px[:, 0]] = (40, 200, 40)           # BGR green
        depth[px[:, 1], px[:, 0]] = pts[:, 2]
        depth = depth.astype(np.float32)
        sc = scene.scan(fr, frame=(img, depth, K), dedupe_px=12.0)
        self.assertIn("green", sc["markers"])
        g = sc["markers"]["green"]
        np.testing.assert_allclose(g["c"][:2], c[:2], atol=4.0)
        self.assertAlmostEqual(g["c"][2], 10.0, delta=2.0)
        self.assertGreater(abs(g["axis2"][1]), 0.95)
        r = np.hypot(*(c[:2] - fr.BASE))
        self.assertAlmostEqual(g["r"], r, delta=4.0)
        self.assertEqual(g["reachable"], r <= 300)
        self.assertFalse(sc["dark"])
        self.assertEqual(scene.reach_class(240), "down")
        self.assertEqual(scene.reach_class(280), "tilted")
        self.assertEqual(scene.reach_class(305), "out")
        dark = scene.scan(fr, frame=((img * 0.1).astype(np.uint8), depth, K))
        self.assertTrue(dark["dark"])


if __name__ == "__main__":
    unittest.main()
