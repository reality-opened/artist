"""drawbot marker picking / sliding tests. Synthetic depth points and mocked motion only: nothing is sent."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from drawbot import arm, pick
from drawbot.frame import Frame
from drawbot.rgbd import project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_drawbot_kin import FIELD_FRAME, FIELD_X, HAVE_URDF, K, quiet

if HAVE_URDF:
    from drawbot import kinmodel, slide


def cylinder_points(c, ax, L=130.0, rad=7.0, step=0.5, rng=None):
    """Upper half of a marker lying on the table (axis in plane xy, center height = rad)."""
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    perp = np.array([-ax[1], ax[0]])
    s = np.arange(-L / 2, L / 2, step)
    th = np.linspace(0, np.pi, 40)
    S, T = np.meshgrid(s, th)
    xy = np.asarray(c, float) + S.reshape(-1, 1) * ax + (rad * np.cos(T)).reshape(-1, 1) * perp
    h = rad + rad * np.sin(T).reshape(-1)
    Q = np.c_[xy, h]
    if rng is not None:
        Q = Q + rng.normal(0, 0.5, Q.shape)
    return Q


class Arm6:
    """Mock arm with a gripper: closing stops at `stall` when a marker is between the jaws."""

    def __init__(self, q0=(1952, 3250, 1550, 2330, 3746, 1000), stall=825, sag=None, slip_on_lift=False):
        self.cmd = {i + 1: int(v) for i, v in enumerate(q0)}
        self.stall, self.sag, self.slip = stall, sag or {}, slip_on_lift
        self.moves, self.sent, self.lifted = [], [], False

    def move(self, goal, speed=12, settle=0.5):
        arm.check_goal(goal)
        self.moves.append(dict(goal))
        self.cmd.update({int(k): int(v) for k, v in goal.items()})

    def send(self, cmd):
        self.sent.append(cmd)
        if "gripper" in cmd:
            self.cmd[6] = int(cmd["gripper"])

    def joints(self):
        t = dict(self.cmd)
        p = {i: self.cmd[i] - self.sag.get(i, 0) for i in range(1, 6)}
        held = self.stall is not None and not (self.slip and self.lifted)
        p[6] = max(self.cmd[6], self.stall) if held else self.cmd[6]
        m = {str(i): dict(target=t[i], pos=p[i], load=1024 + 200 if i == 6 and p[6] > t[6] else 20) for i in p}
        return t, p, m


class LocateTests(unittest.TestCase):
    def test_locate_synthetic_cylinder_with_clutter(self):
        rng = np.random.default_rng(1)
        c, ax = np.array([-150.0, 60.0]), np.array([0.6, 0.8])
        table = np.c_[rng.uniform(-260, -40, (30000, 2)), rng.normal(0, 0.6, 30000)]
        box = np.c_[rng.uniform([-100, 110], [-80, 130], (800, 2)), rng.uniform(40, 80, 800)]   # tall object
        speck = np.c_[rng.uniform([-200, 20], [-196, 24], (40, 2)), np.full(40, 10.0)]         # small clutter
        Q = np.vstack([table, cylinder_points(c, ax, rng=rng), box, speck])
        base = np.array(FIELD_FRAME["base"])
        r = pick.locate_points(Q, c + [15, -10], base=base, radius=90)
        self.assertIsNotNone(r)
        np.testing.assert_allclose(r["center"], c, atol=1.5)
        self.assertGreater(abs(r["axis"] @ ax), np.cos(np.radians(2)))
        self.assertAlmostEqual(r["length"], 130 * 0.96, delta=6)     # 2..98 percentile extent
        self.assertAlmostEqual(r["top"], 14.0, delta=1.5)
        self.assertGreater(r["tall_nearby"], 100)
        self.assertLess(np.linalg.norm(r["near_end"] - base), np.linalg.norm(r["far_end"] - base))
        self.assertIsNone(pick.locate_points(table, c, base=base))    # flat table: nothing

    def test_find_marker_dark_room_synthetic_frame(self):
        fr = Frame(**FIELD_FRAME)
        H, W = 720, 1280
        v, u = np.mgrid[0:H, 0:W]
        rays = np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1)
        depth = (fr.D / (rays @ fr.N)).astype(np.float32)
        img = np.full((H, W, 3), 22, np.uint8)                          # dark room: mean V ~ 22
        c, ax = np.array([-60.0, -40.0]), np.array([0.0, 1.0])
        pts = fr.to_cam(cylinder_points(c, ax, step=0.3))
        px = np.round(project(pts, K)).astype(int)
        order = np.argsort(-pts[:, 2])                                  # far first, near overwrites
        img[px[order, 1], px[order, 0]] = (8, 45, 8)                    # dim green
        depth[px[order, 1], px[order, 0]] = pts[order, 2]
        f = pick.find_marker("green", fr=fr, frame=(img, depth, K))
        self.assertIsNotNone(f)
        self.assertEqual(f["pose"]["source"], "shape")
        np.testing.assert_allclose(f["pose"]["center"], c, atol=2.5)
        self.assertGreater(abs(f["pose"]["axis"][1]), 0.99)
        self.assertIsNone(pick.find_marker("blue", fr=fr, frame=(img, depth, K)))


@unittest.skipUnless(HAVE_URDF, "SO101 URDF not available")
class PickPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = kinmodel.KinModel([1, 1, 1], FIELD_X, mid=kinmodel.MID_DEFAULT, base=FIELD_FRAME["base"])
        cls.C_G = pick.close_dir_gripper(cls.m)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DRAWBOT_STATE"] = self.tmp.name
        arm.CTL = Path(self.tmp.name)
        arm.enable(False)
        kinmodel._MODEL = None

    def tearDown(self):
        kinmodel._MODEL = None
        self.tmp.cleanup()

    def test_anchor_reproduces_field_roll(self):
        a = pick.PICK_DEFAULTS["pick_anchor"]
        roll, err, _ = pick.best_roll(a["p"], a["tilt"], [-0.48, 0.88], self.C_G, self.m)
        self.assertLess(abs(roll - a["roll"]), 30)
        self.assertLess(err, 2.0)
        self.assertAlmostEqual(np.linalg.norm(self.C_G), 1.0)

    def test_best_roll_perpendicular(self):
        m = self.m
        for ang in (0, 35, 90, 140):
            ma = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            p = [m.base[0] + 190, m.base[1] + 40, 60.0]
            roll, err, q = pick.best_roll(p, 20, ma, self.C_G, m)
            self.assertTrue(pick.PICK_ROLL[0] <= roll <= pick.PICK_ROLL[1])
            u = pick.in_plane_dir(m, q, roll, self.C_G)
            self.assertLess(abs(u @ ma), np.sin(np.radians(3)), (ang, roll, err))

    def test_offset_sign_puts_marker_between_fingers(self):
        m = self.m
        center, axis = [-210.0, 143.0], [-0.48, 0.88]
        plan = pick.plan_pick(center, axis, m, self.C_G)
        self.assertTrue(plan["ok"], plan)
        g = plan["waypoints"][-1]
        self.assertEqual(g["name"], "grasp")
        q5 = [g["joints"][i] for i in range(1, 6)]
        tcp = m.fk(q5)[0]
        u = pick.in_plane_dir(m, q5[:4], q5[4], self.C_G)
        # TCP (fixed-jaw tip) sits 25 mm on the -u side; grasp center = TCP + 25 u = the marker
        np.testing.assert_allclose(tcp[:2] + 25 * u, center, atol=2.5)
        self.assertGreater((np.asarray(center) - tcp[:2]) @ u, 20)
        self.assertAlmostEqual(tcp[2], 9.0, delta=2.5)
        self.assertLess(abs(u @ pick.unit2(axis)), 0.1)

    def test_plan_within_joint_windows_and_vertical(self):
        m = self.m
        rng = np.random.default_rng(3)
        for _ in range(4):
            r, ang = rng.uniform(150, 230), rng.uniform(-20, 40)
            c = m.base + r * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            plan = pick.plan_pick(c, rng.normal(size=2), m, self.C_G)
            self.assertTrue(plan["ok"], plan)
            wps = plan["waypoints"] + [plan["lift"]]
            for w in wps:
                self.assertTrue(pick.in_window(w["joints"]), w)
                self.assertLessEqual(w["res"], 2.5)
                arm.check_goal(w["joints"])
            xy = np.array([w["xyh"][:2] for w in wps])
            self.assertLess(np.ptp(xy, axis=0).max(), 1e-6)                    # vertical descent / lift
            hs = [w["xyh"][2] for w in plan["waypoints"]]
            self.assertEqual(hs, sorted(hs, reverse=True))
            self.assertEqual({w["joints"][5] for w in wps}, {plan["roll"]})
        far = pick.plan_pick(m.base + [330, 0], [0, 1], m, self.C_G)
        self.assertFalse(far["ok"])
        self.assertIn("out of reach", far["reason"])

    def test_execute_pick_mock_arm_and_verify(self):
        m = self.m
        plan = pick.plan_pick([-60.0, 20.0], [0.2, 1.0], m, self.C_G)
        self.assertTrue(plan["ok"], plan)
        sim = Arm6(sag={2: 25, 3: -15})
        r = pick.execute_pick(plan, move_fn=sim.move, joints_fn=sim.joints, send_fn=sim.send, out=quiet)
        self.assertEqual(sim.sent, [{"gripper": 1000}, {"gripper": 730}])        # no easing after the close
        self.assertTrue(r["grasp"]["held"])
        self.assertEqual(r["grasp"]["pos"], 825)
        self.assertTrue(r["held"])
        self.assertEqual(sim.cmd[5], plan["roll"])
        lift = plan["lift"]["joints"]
        self.assertTrue(all(abs(sim.joints()[1][i] - lift[i]) <= 5 for i in range(1, 5)))  # sag corrected
        self.assertFalse(Path(self.tmp.name, "cmds.jsonl").exists())
        empty = Arm6(stall=None)
        r = pick.execute_pick(plan, move_fn=empty.move, joints_fn=empty.joints, send_fn=empty.send, out=quiet)
        self.assertFalse(r["grasp"]["held"])
        self.assertIn("empty", r["grasp"]["reason"])
        with self.assertRaises(RuntimeError):
            pick.execute_pick(dict(ok=False, reason="x"), move_fn=sim.move, joints_fn=sim.joints, send_fn=sim.send)

    def test_slide_plan_and_mock_execute(self):
        m = self.m
        c = m.base + np.array([335.0, 30.0])
        mk = dict(center=c, axis=[0.99, 0.1], length=130.0, top=17.0)
        plan = slide.plan_slide(mk, m, dist=80, step=10)
        self.assertTrue(plan["ok"], plan)
        self.assertAlmostEqual(plan["press_h"], 10.0)
        self.assertLess(plan["r1"], plan["r0"] - 70)                          # pulled toward the robot
        d = np.array(plan["dir"])
        self.assertGreater(abs(d @ pick.unit2(mk["axis"])), 0.999)             # along the marker axis
        presses = [w for w in plan["waypoints"] if w["name"] not in ("hover", "lift")]
        self.assertEqual(len(presses), 9)
        for w in plan["waypoints"]:
            q5 = [w["joints"][i] for i in range(1, 6)]
            self.assertLess(np.linalg.norm(m.fk(q5)[0] - w["xyh"]), 2.5)
        sim = Arm6(stall=None)
        tr = slide.execute_slide(plan, move_fn=sim.move, joints_fn=sim.joints, send_fn=sim.send, out=quiet)
        self.assertEqual(sim.sent, [{"gripper": slide.SLIDE_GRIPPER}])
        self.assertEqual([t["name"] for t in tr], [w["name"] for w in plan["waypoints"]])
        self.assertFalse(slide.plan_slide(dict(center=m.base + [600, 0], axis=[0, 1], length=130, top=17), m)["ok"])

    def test_cli_pick2_dry_sends_nothing(self):
        import drawbot_cli
        from drawbot import save
        save("kin_model.json", dict(signs=[1, 1, 1], x=FIELD_X, mid=kinmodel.MID_DEFAULT.tolist()))
        save("frame.json", FIELD_FRAME)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(drawbot_cli.main(["pick2", "-210", "143", "-0.48", "0.88"]), 0)
            self.assertEqual(drawbot_cli.main(["pick2", "-210", "143", "-0.48", "0.88", "--dry", "--yes"]), 0)
        arm.enable(False)
        self.assertIn("plan only", buf.getvalue())
        self.assertIn("grasp", buf.getvalue())
        self.assertFalse(Path(self.tmp.name, "cmds.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
