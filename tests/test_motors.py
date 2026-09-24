import argparse
import unittest
from unittest.mock import Mock

from arm import motor_ids, read_motors


class MotorInspectionTests(unittest.TestCase):
    def test_reject_broadcast_and_duplicate_ids(self):
        for value in ["254", "0", "1,1", "-1", "abc", ""]:
            with self.assertRaises(argparse.ArgumentTypeError):
                motor_ids(value)

    def test_explicit_ids_are_not_assigned_joint_names(self):
        self.assertEqual(motor_ids("1,2,3,4,5,6"), list(range(1, 7)))

    def test_faults_and_unknown_models_are_not_decoded(self):
        for ping in [(777, -6, 0), (777, 0, 4), (1284, 0, 0)]:
            packet = Mock()
            packet.ping.return_value = ping
            self.assertIn("note", read_motors(None, packet, [1])[0])
            self.assertEqual([c[0] for c in packet.mock_calls], ["ping"])

    def test_only_reads_and_signed_position(self):
        packet = Mock()
        packet.ping.return_value = (777, 0, 0)
        packet.read2ByteTxRx.return_value = (0x8005, 0, 0)
        packet.read1ByteTxRx.return_value = (1, 0, 0)
        self.assertEqual(read_motors(None, packet, [1])[0]["position_ticks"], -5)
        self.assertTrue(all(c[0] in {"ping", "read1ByteTxRx", "read2ByteTxRx"}
                            for c in packet.mock_calls))


if __name__ == "__main__":
    unittest.main()
