"""Focused tests for rejecting inconsistent labels and contact artifacts."""

import unittest

import numpy as np

from script.a3.smpl_corpus_qualify import source_metrics, contact_metrics
from script.a3.smpl_encoder import PARENTS


def neutral_labels():
    offsets = np.asarray([
        [.09, -.08, 0], [-.09, -.08, 0], [0, .12, 0],
        [0, -.4, 0], [0, -.4, 0], [0, .12, 0],
        [0, -.4, 0], [0, -.4, 0], [0, .12, 0],
        [0, -.06, .12], [0, -.06, .12], [0, .18, 0],
        [.12, .08, 0], [-.12, .08, 0], [0, .18, 0],
        [.15, 0, 0], [-.15, 0, 0], [.25, 0, 0], [-.25, 0, 0],
        [.25, 0, 0], [-.25, 0, 0], [.08, 0, 0], [-.08, 0, 0]])
    rest = np.zeros((24, 3))
    for joint, parent in enumerate(PARENTS[1:], 1):
        rest[joint] = rest[parent] + offsets[joint - 1]
    pose = np.zeros((300, 24, 3))
    joints = np.tile(rest, (300, 1, 1))
    joints[:, :, 1] += .94
    return pose, joints


class CorpusChecks(unittest.TestCase):
    def test_coherent_skeleton_and_translation_pass(self):
        pose, joints = neutral_labels()
        joints[:, :, 2] += np.arange(300)[:, None] / 300
        metrics, _, _ = source_metrics(pose, joints)
        self.assertTrue(metrics['structural_quality_passed'])
        self.assertLess(metrics['source_fk_max_m'], 1e-12)

    def test_pose_motion_inconsistent_with_static_joints(self):
        pose, joints = neutral_labels()
        pose[:, 1, 0] = .2 * np.sin(np.linspace(0, 2 * np.pi, 300))
        metrics, _, _ = source_metrics(pose, joints)
        self.assertFalse(metrics['checks']['source_fk'])

    def test_one_frame_bone_corruption_rejected(self):
        pose, joints = neutral_labels()
        joints[100, 20, 0] += .01
        metrics, _, _ = source_metrics(pose, joints)
        self.assertFalse(metrics['checks']['bone_lengths'])
        self.assertFalse(metrics['structural_quality_passed'])

    def test_millimetres_rejected(self):
        pose, joints = neutral_labels()
        metrics, _, _ = source_metrics(pose, joints * 1000)
        self.assertFalse(metrics['checks']['metre_scale'])

    def test_contact_slip_is_measured_during_source_support(self):
        _, joints = neutral_labels()
        names = ['left_ankle', 'right_ankle']
        actual = np.zeros((300, 2, 3))
        actual[:, :, 0] = np.arange(300)[:, None] / 30
        contact = contact_metrics(joints, actual, np.zeros((300, 2)), names)
        self.assertEqual(contact['contact_frames_per_foot'], [300, 300])
        np.testing.assert_allclose(contact['robot_contact_ankle_xy_speed_p95_m_s'], [1, 1])
        self.assertEqual(contact['penetration_frames'], 0)

    def test_actual_foot_geometry_rejects_penetration(self):
        _, joints = neutral_labels()
        contact = contact_metrics(joints, np.zeros((300, 2, 3)),
                                  np.full((300, 2), -.01), ['left_ankle', 'right_ankle'])
        self.assertEqual(contact['penetration_frames'], 300)


if __name__ == '__main__':
    unittest.main()
