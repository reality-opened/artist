import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from drawbot import arm, rgbd, skills, track
from drawbot.control import Ctl2
from drawbot.frame import Frame, base_from_pan

K = {"fx": 686.65, "fy": 686.52, "cx": 641.32, "cy": 359.03}
FIELD_FRAME = {"ctr": [-44.7, 67.23, 426.82], "e1": [-0.84033, 0.100997, -0.532582],
               "e2": [0.521402, 0.419329, -0.743171], "n": [0.148269, -0.902199, -0.405036], "d": -240.16,
               "base": [-242.58, -11.69], "pan_ang": {"1717": 22.454, "1967": 0.6689, "2217": -20.456}}
CFG = dict(ids=[1, 3, 4], lim={"1": [700, 3400], "3": [700, 3300], "4": [1735, 3775]}, speed=4, track="orange",
           pen_L=28.0, pen_up=12.0, press=2.5, step=2.0, site_bounds=[[-200, 200], [-350, 350], [-220, 210]],
           relocate_bounds=[[1450, 2150], [900, 2000], [1750, 2450]])


class Plant:
    """Linear arm+camera: y = y0 + J (u - u0); the barrel cannot go below `floor` (paper contact)."""

    def __init__(self, J, u0=(1800, 1500, 2030), y0=(20.0, 60.0, 40.0), floor=-1e9):
        self.J, self.u0, self.y0, self.floor = np.array(J, float), np.array(u0, float), np.array(y0, float), floor
        self.u = self.u0.copy()
        self.moves = []

    def move(self, goal, speed=12, settle=0.5):
        self.moves.append((dict(goal), speed))
        self.u = np.array([goal[i] for i in (1, 3, 4)], float)

    def joints(self):
        t = {1: self.u[0], 2: 2500, 3: self.u[1], 4: self.u[2], 5: 2000, 6: 1500}
        return t, dict(t), None

    def y(self):
        y = self.y0 + self.J @ (self.u - self.u0)
        y[2] = max(y[2], self.floor)
        return y

    def measure(self, avg=3, prior=None, gate=40.0):
        return dict(xyh=self.y())

    def status(self):
        t = self.joints()[0]
        return {"motors": {str(i): dict(target=int(t[i]), pos=int(t[i]), load=0) for i in range(1, 7)}}


J_TRUE = [[0.07, -0.10, 0.0], [-0.26, -0.02, -0.01], [-0.01, -0.12, -0.15]]


class DrawbotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DRAWBOT_STATE"] = self.tmp.name
        arm.CTL = Path(self.tmp.name)
        arm.enable(False)

    def tearDown(self):
        self.tmp.cleanup()

    def ctl(self, plant, J=None):
        c = Ctl2(cfg=CFG, measure_fn=plant.measure, move_fn=plant.move, joints_fn=plant.joints)
        c.J = np.array(J if J is not None else J_TRUE, float)
        return c

    def test_plane_fit_large_cloud(self):
        rng = np.random.default_rng(1)
        n = np.array([0.15, -0.9, -0.41]); n /= np.linalg.norm(n)
        d = -240.0
        a = np.cross(n, [1, 0, 0]); a /= np.linalg.norm(a)
        b = np.cross(n, a)
        s, t = rng.uniform(-150, 150, (2, 150_000))
        pts = d * n + s[:, None] * a + t[:, None] * b + rng.normal(0, 0.5, (150_000, 1)) * n
        pts[:5000] += rng.uniform(-60, 60, (5000, 3))  # clutter
        nf, df, inl = rgbd.fit_plane(pts)
        self.assertLess(nf[2], 0)
        self.assertGreater(abs(nf @ n), 0.9999)
        self.assertAlmostEqual(df * np.sign(nf @ n), d, delta=0.5)
        self.assertGreater(inl.mean(), 0.9)

    def test_backproject_project_roundtrip(self):
        u, v, z = np.array([10.0, 640, 1200]), np.array([5.0, 360, 700]), np.array([300.0, 450, 800])
        p = rgbd.backproject(u, v, z, K)
        np.testing.assert_allclose(rgbd.project(p, K), np.stack([u, v], 1), atol=1e-9)
        np.testing.assert_allclose(p[:, 2], z)

    def test_frame_roundtrip_and_pan(self):
        fr = Frame(**FIELD_FRAME)
        xyh = np.array([[10.0, -20, 30], [-100, 50, 0]])
        np.testing.assert_allclose(fr.plane(fr.to_cam(xyh)), xyh, atol=0.05)
        self.assertEqual(fr.pan_for_angle(0.6689), 1977)
        self.assertEqual(fr.pan_for_angle(0.6689 + 21.45), 1727)

    def test_base_from_pan_circle(self):
        c, r = np.array([-240.0, -12.0]), 265.0
        samples = [(t, c + r * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]))
                   for t, a in [(1717, 22.0), (1967, 0.5), (2217, -21.0)]]
        fit = base_from_pan(samples)
        np.testing.assert_allclose(fit["base"], c, atol=1e-6)
        self.assertAlmostEqual(fit["ticks_per_deg"], 500 / 43.0, places=6)
        fr = Frame(**{**FIELD_FRAME, **fit})
        self.assertEqual(fr.pan_for_angle(0.5), 1967)

    def test_blob_gating_and_prior(self):
        blob = lambda area: dict(area=area)
        cands = [(blob(5000), np.array([300.0, 0, 50])),   # outside |x| < 250
                 (blob(4000), np.array([0.0, 0, 260])),    # too high
                 (blob(3000), np.array([10.0, 5, -30])),   # below min_h
                 (blob(100), np.array([0.0, 0, 50])),      # too small
                 (blob(2000), np.array([50.0, 10, 40])),
                 (blob(900), np.array([12.0, 3, 41]))]
        self.assertEqual(track.select(cands)[1][0], 50.0)  # largest valid
        self.assertEqual(track.select(cands, prior=[10, 0, 40])[1][0], 12.0)  # nearest to the prior
        self.assertIsNone(track.select(cands, prior=[-150, 0, 40], gate=40.0))
        self.assertIsNone(track.select(cands[:4], min_h=15.0))
        self.assertEqual(track.mode_spec("blue")["hue"], (100, 125))
        self.assertEqual(track.mode_spec("hue:40-80:90")["min_sat"], 90)

    def test_resample(self):
        pts = [[0, 0], [10, 0], [10, 3]]
        r = skills.resample(pts, 2.0)
        np.testing.assert_allclose(r[0], [0, 0]); np.testing.assert_allclose(r[-1], [10, 3])
        self.assertTrue((np.linalg.norm(np.diff(r, axis=0), axis=1) <= 2.0 + 1e-9).all())
        self.assertEqual(len(r), 1 + 5 + 2)

    def test_face_to_plane_orientation(self):
        ctr = [16.6, 74.8]
        np.testing.assert_allclose(skills.face_to_plane([0, 0], ctr, 0.5), ctr)
        np.testing.assert_allclose(skills.face_to_plane([10, 0], ctr, 0.5), [11.6, 74.8])   # face +x -> plane -x
        np.testing.assert_allclose(skills.face_to_plane([0, 10], ctr, 0.5), [16.6, 69.8])   # face +y -> plane -y
        site = {"tip_contact": [16.6, 74.8], "r_face": 20.0, "hover_u": [1812, 1505, 2032]}
        np.testing.assert_allclose(skills.face_center(site), [39.6, 74.8])
        top = skills.layer_strokes("orange", site)[0][0]  # face (0, 35) = top of the circle
        np.testing.assert_allclose(top, [39.6, 54.8], atol=1e-9)

    def test_tilt_compensation_sign(self):
        ref = {"id3": 1587, "id4": 1898}
        self.assertEqual(skills.tilt([1800, 1587, 1898], ref), 0.0)
        th = skills.tilt([1800, 1587 + 100, 1898 + 100], ref)
        self.assertAlmostEqual(th, 200 * 2 * np.pi / 4096)
        b = skills.tip_to_barrel([10.0, 5.0], th, 28.0)
        self.assertGreater(b[0], 10.0)  # + tilt = pen bottom toward the base (-x): barrel sits at larger x
        self.assertEqual(b[1], 5.0)
        np.testing.assert_allclose(skills.barrel_to_tip(b, th, 28.0), [10.0, 5.0])

    def test_level_and_down(self):
        self.assertEqual(arm.level4(2149, 2047), 2934)
        self.assertEqual(arm.level4(2249, 2000), 2934 - 100 + 47)
        self.assertEqual(arm.down4(2149, 2047), 2934 + 1024)

    def test_send_disabled_by_default(self):
        with self.assertRaises(RuntimeError):
            arm.send({"joints": {"1": 2000}})
        self.assertFalse((arm.CTL / "cmds.jsonl").exists())
        with self.assertRaises(ValueError):
            arm.check_goal({2: 2000})
        with self.assertRaises(ValueError):
            arm.check_goal({4: 3800})

    def test_command_clips_to_limits_and_bounds(self):
        move = Mock()
        joints = Mock(return_value=({1: 1800, 2: 2500, 3: 1500, 4: 2030}, None, None))
        c = Ctl2(cfg=CFG, measure_fn=Mock(), move_fn=move, joints_fn=joints)
        c.command([600, 3400.4, 1700])
        move.assert_called_with({1: 700.0, 3: 3300.0, 4: 1735.0}, speed=4, settle=0.3)
        c.bounds = [(1600, 2000), (1155, 1855), (1812, 2242)]
        c.command([2100, 1000, 2000.6])
        move.assert_called_with({1: 2000.0, 3: 1155.0, 4: 2001.0}, speed=4, settle=0.3)
        np.testing.assert_array_equal(c.u, [2000, 1155, 2001])

    def test_y_checked_rejects_outliers(self):
        good, bad = np.array([20.0, 60, 40]), np.array([80.0, 60, 40])
        m = Mock(side_effect=[dict(xyh=bad), dict(xyh=good)])
        c = Ctl2(cfg=CFG, measure_fn=m, move_fn=Mock(), joints_fn=Plant(J_TRUE).joints)
        y, ok = c.y_checked(np.array([21.0, 61, 40]))
        self.assertTrue(ok); np.testing.assert_array_equal(y, good)
        m.side_effect = [dict(xyh=bad)] * 3
        y, ok = c.y_checked(np.array([21.0, 61, 40]))
        self.assertFalse(ok); np.testing.assert_array_equal(y, [21.0, 61, 40])
        self.assertEqual(m.call_args.kwargs["prior"].tolist(), [21.0, 61, 40])

    def test_servo_converges_on_linear_plant(self):
        p = Plant(J_TRUE)
        c = self.ctl(p, np.array(J_TRUE) * 1.2)  # 20% model error
        y = c.servo([30.0, 50.0, 35.0], tol=0.5, iters=20)
        self.assertLess(np.linalg.norm(y - [30, 50, 35]), 0.8)
        self.assertTrue(Path(self.tmp.name, "draw2_state.json").exists())

    def test_probe_contact_detects_floor(self):
        p = Plant(J_TRUE, floor=30.0)
        c = self.ctl(p)
        r = skills.probe_contact(c, step=1.5, status_fn=p.status, out=lambda *_: None)
        self.assertAlmostEqual(r["contact_h"], 30.0, delta=0.5)
        self.assertGreaterEqual(sum(t["dh"] < -1 for t in r["trace"]), 2)
        self.assertGreater(p.y()[2], 38.0)  # lifted afterwards

    def test_draw_layer_on_plant(self):
        p = Plant(J_TRUE, y0=(40.0, 75.0, 40.0), floor=25.5)
        c = self.ctl(p)
        site = {"tip_contact": [16.6, 74.8], "contact_h": 25.5, "r_face": 20.0, "hover_u": [1800, 1500, 2030]}
        pen = skills.Pen(c, site, {"id3": 1500, "id4": 2030}, CFG)
        log = Path(self.tmp.name, "layer.log")
        summary = skills.draw_layer(pen, "pink", log=log, out=lambda *_: None)
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(mean < 1.5 for mean, _ in summary))
        self.assertGreater(len(log.read_text().splitlines()), 10)
        for goal, _ in p.moves:  # every command stayed inside the site bounds
            self.assertTrue(1600 <= goal[1] <= 2000 and 1150 <= goal[3] <= 1850 and 1810 <= goal[4] <= 2240)

    def test_probe_contact_needs_verified_motion(self):
        p = Plant(J_TRUE, floor=40.0)  # already touching: no free steps first
        c = self.ctl(p)
        with self.assertRaises(RuntimeError):
            skills.probe_contact(c, status_fn=p.status, out=lambda *_: None)
        p = Plant(J_TRUE)
        c = self.ctl(p)
        stale = lambda: {"motors": {str(i): dict(target=0, pos=0, load=0) for i in range(1, 7)}}
        with self.assertRaises(RuntimeError):  # teleop not following our targets
            skills.probe_contact(c, status_fn=stale, out=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
