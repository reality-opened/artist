import unittest
from unittest.mock import Mock

from joint_step import execute


class JointStepTests(unittest.TestCase):
    def setup_motion(self):
        values = {i: dict(position=2000, max_torque_limit=1000,
                          voltage_min=40, voltage_max=80) for i in range(1, 7)}
        servos = {i: Mock() for i in values}
        plan = dict(deltas={2:24, 3:-24}, start=[2000]*5,
                    target=[2000,2024,1976,2000,2000], torque_limit=300)
        return values, servos, plan

    def test_shift_causes_no_writes(self):
        values, servos, plan = self.setup_motion()
        servos[1].read.side_effect = [0, 2020]
        with self.assertRaisesRegex(RuntimeError, "shifted"):
            execute(servos, values, plan, lambda _: None)
        for s in servos.values():
            s.write.assert_not_called()

    def test_setup_failure_releases_every_selected_joint(self):
        values, servos, plan = self.setup_motion()
        for s in servos.values():
            s.read.side_effect = lambda address, size=1: 0 if address == 40 else 2000
        servos[3].write.side_effect = lambda address, *args: (_ for _ in ()).throw(RuntimeError("injected")) if address == 41 else None
        with self.assertRaisesRegex(RuntimeError, "injected"):
            execute(servos, values, plan, lambda _: None)
        for i in (2, 3):
            self.assertEqual(servos[i].write.call_args.args, (40, 0))
        for i in (1, 4, 5, 6):
            servos[i].write.assert_not_called()
