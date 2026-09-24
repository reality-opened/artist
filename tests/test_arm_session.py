import unittest
from unittest.mock import Mock

from arm_session import Session


class SessionTests(unittest.TestCase):
    def session(self):
        s = Session.__new__(Session)
        s.start = {i:2000 for i in range(1,7)}
        s.goals = s.start.copy()
        s.values = {i:dict(max_torque_limit=500, voltage_min=40, voltage_max=80,
                          low=0,high=4095) for i in s.start}
        s.servos = {i:Mock() for i in s.start}
        s.torque_limit, s.enabled, s.record = 300, False, Mock()
        return s

    def test_stage_every_motor_before_any_torque_enable(self):
        s = self.session()
        events = []
        for i,servo in s.servos.items():
            servo.read.side_effect = lambda a,n=1: 0 if a==40 else 2000
            servo.write.side_effect = lambda a,v,n=1,i=i: events.append((i,a,v))
        s.enable()
        staged = [events.index((i,42,2000)) for i in s.servos]
        enabled = [events.index((i,40,1)) for i in s.servos]
        self.assertLess(max(staged),min(enabled))

    def test_shift_blocks_all_writes(self):
        s = self.session()
        s.servos[1].read.side_effect = [0,2010]
        with self.assertRaisesRegex(RuntimeError,"changed"):
            s.enable()
        self.assertFalse(s.enabled)
        for servo in s.servos.values():
            servo.write.assert_not_called()

    def test_release_continues_after_unreachable_joint(self):
        s = self.session()
        s.enabled = True
        s.servos[2].write.side_effect = RuntimeError("unreachable")
        with self.assertRaisesRegex(RuntimeError,"UNCONFIRMED"):
            s.release()
        for i in [1,3,4,5,6]:
            s.servos[i].write.assert_called_with(40,0)
