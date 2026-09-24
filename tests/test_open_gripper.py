import unittest
from unittest.mock import Mock

from open_gripper import plan_open, open_once


class OpenGripperTests(unittest.TestCase):
    def values(self):
        return dict(model=777, mode=0, torque=0, position=1676, low=1679,
                    high=3130, voltage=53, voltage_min=40, voltage_max=80,
                    temperature=27, max_torque_limit=500)

    def test_boundary_recovery_only_toward_valid_interval(self):
        p = plan_open(self.values(), 24, 150)
        self.assertEqual((p["staged"], p["target"]), (1679, 1700))
        for update in [dict(position=1674), dict(torque=1), dict(position=3123),
                       dict(mode=1), dict(voltage=39), dict(temperature=50)]:
            with self.assertRaises(ValueError):
                plan_open(self.values() | update, 24, 150)
        for ticks in [-24, 8, 115]:
            with self.assertRaises(ValueError):
                plan_open(self.values(), ticks, 150)

    def test_shift_refuses_all_writes(self):
        s = Mock()
        s.read.side_effect = [0, 1680]
        with self.assertRaises(RuntimeError):
            open_once(s, plan_open(self.values(), 24, 150), lambda _: None)
        s.write.assert_not_called()

    def test_fault_releases_and_staging_precedes_torque(self):
        s = Mock()
        s.read.side_effect = [0, 1676, 1676, 1679, 53, 27, 1, 0, 0, 1700, 1, 1679]
        with self.assertRaisesRegex(RuntimeError, "Motor fault"):
            open_once(s, plan_open(self.values(), 24, 150), lambda _: None)
        writes = [c.args for c in s.write.call_args_list]
        self.assertLess(writes.index((42, 1679, 2)), writes.index((40, 1)))
        self.assertEqual(writes[-1], (40, 0))
