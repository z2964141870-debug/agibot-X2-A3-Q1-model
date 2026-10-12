"""Coordinate, timing, sample eligibility, and frozen-controller contracts."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from script.a3.smpl_encoder import (FrozenA3Bridge, SMPLStudent, feature_frames,
    smpl_window, pack_teacher, require_training_pairs, distillation_step, SMPL_TO_ROBOT)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_input_dim, self.actor_obs_dim = 640, 930
        self.encoder = nn.Linear(640, 64)
        self.decoder = nn.Linear(994, 29)

    def fsq(self, latent):
        return torch.round(torch.tanh(latent) * 16) / 16


class SMPLContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.times = np.arange(31) / 50
        self.joints = np.random.default_rng(0).normal(size=(31, 24, 3))
        self.pose = np.broadcast_to(np.eye(3), (31, 24, 3, 3)).copy()

    def test_global_coordinate_change_preserves_features(self):
        rotation = Rotation.from_rotvec([.2, -.3, .5]).as_matrix()
        root_robot = SMPL_TO_ROBOT @ rotation @ SMPL_TO_ROBOT.T
        original = feature_frames(self.joints, self.pose, np.eye(3))
        transformed_pose = self.pose.copy()
        transformed_pose[:, 0] = rotation
        transformed_joints = self.joints @ rotation.T + [7, -3, 2]
        transformed = feature_frames(transformed_joints, transformed_pose, root_robot)
        np.testing.assert_allclose(original, transformed, atol=1e-6)

    def test_wrist_twist_changes_feature_with_identical_joint_positions(self):
        twisted = self.pose.copy()
        twisted[:, 20] = Rotation.from_rotvec([.4, 0, 0]).as_matrix()
        base = feature_frames(self.joints, self.pose, np.eye(3))
        new = feature_frames(self.joints, twisted, np.eye(3))
        np.testing.assert_array_equal(base[:, :78], new[:, :78])
        self.assertGreater(np.max(np.abs(base[:, 78:] - new[:, 78:])), .1)

    def test_history_cannot_read_interpolation_sample_after_arrival(self):
        sparse = self.times[::2]
        with self.assertRaisesRegex(ValueError, 'not yet arrived'):
            smpl_window(sparse, self.joints[::2], self.pose[::2], np.eye(3),
                        .21, mode='causal_history', arrival_time=.21)

    def test_changing_future_does_not_change_causal_input(self):
        first, _ = smpl_window(self.times, self.joints, self.pose, np.eye(3),
                              .2, mode='causal_history', arrival_time=.2)
        changed = self.joints.copy()
        changed[self.times > .2] += 100
        second, query = smpl_window(self.times, changed, self.pose, np.eye(3),
                                   .2, mode='causal_history', arrival_time=.2)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(query.max(), .2 + 1e-10)

    def test_unavailable_pose_values_cannot_affect_causal_window(self):
        expected, _ = smpl_window(self.times, self.joints, self.pose, np.eye(3),
                                 .2, mode='causal_history', arrival_time=.2)
        joints, pose = self.joints.copy(), self.pose.copy()
        joints[self.times > .2] = np.nan
        pose[self.times > .2] = np.nan
        actual, _ = smpl_window(self.times, joints, pose, np.eye(3),
                               .2, mode='causal_history', arrival_time=.2)
        np.testing.assert_array_equal(expected, actual)

    def test_reflection_gap_and_extrapolation_rejected(self):
        reflected = self.pose.copy()
        reflected[:, 0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'SO\(3\)'):
            feature_frames(self.joints, reflected, np.eye(3))
        with self.assertRaisesRegex(ValueError, 'extrapolate'):
            smpl_window(self.times, self.joints, self.pose, np.eye(3), .5)
        with self.assertRaisesRegex(ValueError, 'gap'):
            smpl_window([0., .3], self.joints[:2], self.pose[:2], np.eye(3), 0.)

    def test_teacher_groups_position_then_velocity(self):
        q = np.arange(290).reshape(10, 29)
        v = q + 1000
        result = pack_teacher(q, v, np.zeros((10, 6))).reshape(10, 64)
        np.testing.assert_array_equal(result[:, :58].flatten(), np.concatenate((q.flatten(), v.flatten())))
        np.testing.assert_array_equal(result[0, :58], q[:2].flatten())

    def metadata(self):
        return dict(usage='TRAINING_PAIRED_SMPL_A3', training_allowed=True,
            quality_verified=True, source_use_verified=True, pair_alignment_verified=True,
            input_mode='offline_future', train_sessions=['synthetic_train'],
            validation_sessions=['synthetic_validation'], test_sessions=['synthetic_test'])

    def test_test_only_missing_checks_and_session_leakage_rejected(self):
        meta = self.metadata()
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY'):
            require_training_pairs(dict(meta, usage='TEST_ONLY', training_allowed=False))
        with self.assertRaisesRegex(ValueError, 'quality_verified'):
            require_training_pairs(dict(meta, quality_verified=False))
        with self.assertRaisesRegex(ValueError, 'separate'):
            require_training_pairs(dict(meta, validation_sessions=['synthetic_train']))

    def test_synthetic_student_update_preserves_every_teacher_tensor(self):
        student = SMPLStudent(hidden=(8,))
        bridge = FrozenA3Bridge(TinyPolicy())
        before = {name: value.clone() for name, value in bridge.state_dict().items()}
        human = torch.randn(3, 10, 90)
        command = torch.randn(3, 640)
        student_before = student.module[0].weight.clone()
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        result = distillation_step(student, optimizer, human, command, bridge, self.metadata())
        self.assertTrue(np.isfinite(result['loss']))
        self.assertFalse(torch.equal(student_before, student.module[0].weight))
        for name, value in bridge.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))
        self.assertTrue(all(value.grad is None for value in bridge.parameters()))
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY'):
            distillation_step(student, optimizer, human, command, bridge,
                              dict(self.metadata(), usage='TEST_ONLY', training_allowed=False))

    def test_normalization_constant_feature_and_train_only_gate(self):
        student = SMPLStudent(hidden=(8,))
        inputs = torch.ones(3, 10, 90)
        inputs[:, :, 1] = torch.arange(3).float()[:, None]
        with self.assertRaisesRegex(ValueError, 'TEST_ONLY'):
            student.fit_normalization(inputs,
                dict(self.metadata(), usage='TEST_ONLY', training_allowed=False))
        student.fit_normalization(inputs, self.metadata())
        self.assertTrue(torch.equal(student.input_std.reshape(10, 90)[:, 0], torch.ones(10)))
        self.assertTrue(torch.isfinite(student(inputs + .01)).all())

    def test_optimizer_cannot_include_teacher_and_student_reloads(self):
        student = SMPLStudent(hidden=(8,))
        bridge = FrozenA3Bridge(TinyPolicy())
        bad = torch.optim.Adam([*student.parameters(), *bridge.parameters()], lr=1e-4)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            distillation_step(student, bad, torch.ones(2, 10, 90), torch.ones(2, 640), bridge, self.metadata())
        replica = SMPLStudent(hidden=(8,))
        replica.load_state_dict(student.state_dict(), strict=True)
        inputs = torch.randn(2, 10, 90)
        self.assertTrue(torch.equal(student(inputs), replica(inputs)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
