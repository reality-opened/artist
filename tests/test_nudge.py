import unittest
from unittest.mock import Mock

from nudge import check_plan, execute


class NudgeTests(unittest.TestCase):
    def values(self):
        return dict(model=777, mode=0, torque=0, temperature=28, voltage=53,
                    voltage_min=40, voltage_max=84, position=2000, low=0, high=4095,
                    torque_limit=1000, max_torque_limit=1000)

    def test_rejects_unsafe_plans(self):
        for update in [dict(position=4090), dict(position=3), dict(torque=1),
                       dict(mode=1), dict(model=1284), dict(temperature=55),
                       dict(voltage=20), dict(torque_limit=0), dict(low=4095),
                       dict(max_torque_limit=0)]:
            with self.assertRaises(ValueError):
                check_plan(5, 24, self.values() | update)
        for motor, delta in [(1, 24), (5, 0), (5, 115), (5, -115), (6, 25)]:
            with self.assertRaises(ValueError):
                check_plan(motor, delta, self.values())

    def test_no_writes_if_position_shifted(self):
        servo = Mock()
        servo.read.side_effect = [0, 2050]
        with self.assertRaises(RuntimeError):
            execute(servo, check_plan(5, 24, self.values()), lambda _: None)
        servo.write.assert_not_called()

    def test_torque_cap_and_proven_acceleration(self):
        self.assertEqual(check_plan(5, 24, self.values(), 300)["torque_limit"], 300)
        # A prior reduced SRAM setting must not silently cap a later request.
        self.assertEqual(check_plan(5, 24, self.values() | dict(torque_limit=150), 500)["torque_limit"], 500)
        self.assertEqual(check_plan(5, 24, self.values() | dict(max_torque_limit=250), 500)["torque_limit"], 250)
        self.assertEqual(check_plan(5, 24, self.values())["acceleration_register"], 254)
        for limit in [0, 501]:
            with self.assertRaises(ValueError):
                check_plan(5, 24, self.values(), limit)

    def test_stale_goal_replaced_before_enable_and_release_on_fault(self):
        servo = Mock()
        servo.read.side_effect = [0, 2000, 2000, 2100, 53, 0, 0, 1, 2024, 0, 28, 0]
        with self.assertRaises(RuntimeError):
            execute(servo, check_plan(5, 24, self.values()), lambda _: None)
        writes = [c.args for c in servo.write.call_args_list]
        self.assertLess(writes.index((42, 2000, 2)), writes.index((40, 1)))
        self.assertEqual(writes[-1], (40, 0))


if __name__ == "__main__":
    unittest.main()
