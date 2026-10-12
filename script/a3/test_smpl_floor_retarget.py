"""Verify floor constraints and contact timing without policy computation."""

import unittest
import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers
configure_helpers()
from script.a3.smpl_floor_retarget import FloorA3Retargeter, supported_interval_slip
from script.a3 import test_smpl_contact_retarget as contact_tests


class FloorChecks(unittest.TestCase):
    def test_floor_limits_are_not_empty(self):
        r = FloorA3Retargeter()
        self.assertGreater(r.floor_collision_pairs, 0)

    def test_root_correction_rate_is_bounded(self):
        r = FloorA3Retargeter()
        r.set_human_source(np.zeros((300,24,3)))
        source = contact_tests.ContactTargets().source(r)
        r.targets(source, 1.)
        speed = np.linalg.norm(np.diff(r.root_offset, axis=0)*30, axis=-1)
        self.assertLessEqual(float(speed.max()), .5+1e-12)

    def test_noncontact_gap_movement_excluded(self):
        joints = np.zeros((30,24,3))
        joints[10:20,[10,11],1] = .2
        actual = np.zeros((30,2,3))
        actual[10:20,:,0] = 1.
        slip = supported_interval_slip(joints, actual, np.zeros((30,2)), ['left_ankle','right_ankle'])
        self.assertGreater(min(slip['supported_intervals_per_foot']), 5)
        self.assertEqual(slip['robot_contact_ankle_xy_speed_p95_m_s'], [0.,0.])

    def test_true_supported_slip_retained(self):
        joints = np.zeros((30,24,3))
        actual = np.zeros((30,2,3))
        actual[:,:,0] = np.arange(30)[:,None]/30
        slip = supported_interval_slip(joints, actual, np.zeros((30,2)), ['left_ankle','right_ankle'])
        np.testing.assert_allclose(slip['robot_contact_ankle_xy_speed_p95_m_s'], [1.,1.])


if __name__ == '__main__':
    unittest.main()
