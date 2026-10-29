"""Verify static saved-pose substitutions without physics or policy imports."""

import unittest

import numpy as np

from analyze_q1_saved_target_pose import describe, joint_outside, relative_other, substitute_joints


class SavedTargetPoseTests(unittest.TestCase):
    def test_joint_substitution_holds_root_velocities_and_input_fixed(self):
        states = np.arange(3 * 57, dtype=np.float64).reshape(3, 57)
        original = states.copy()
        joints = np.full((3, 22), .15)
        value = substitute_joints(states, joints)
        np.testing.assert_array_equal(states, original)
        np.testing.assert_array_equal(value[:, :13], states[:, :13])
        np.testing.assert_array_equal(value[:, 35:], states[:, 35:])
        np.testing.assert_array_equal(value[:, 13:35], joints)

    def test_substitution_rejects_misaligned_or_nonfinite_postures(self):
        for states, joints in ((np.zeros((0, 57)), np.zeros((0, 22))),
                               (np.zeros((2, 57)), np.zeros((1, 22))),
                               (np.zeros((2, 57)), np.full((2, 22), np.nan))):
            with self.assertRaises(ValueError):
                substitute_joints(states, joints)

    def test_signed_side_height_removes_common_root_translation(self):
        heights = np.array([[.02, .005], [-.1, -.07]])
        relative = relative_other(heights)
        np.testing.assert_allclose(relative, [[.015, -.015], [-.03, .03]])
        np.testing.assert_allclose(relative_other(heights + .73), relative)
        with self.assertRaises(ValueError):
            relative_other(np.zeros((3, 1)))

    def test_residual_geometry_delta_keeps_side_sign_and_exact_decomposition(self):
        pre = np.array([[.02, 0.], [.01, .005]])
        nominal = pre + [[.01, .001], [-.001, .002]]
        target = nominal + [[.001, .003], [.002, -.001]]
        for convert in (lambda value: value, relative_other):
            np.testing.assert_allclose(convert(target) - convert(pre),
                (convert(nominal) - convert(pre)) + (convert(target) - convert(nominal)))
        self.assertGreater((relative_other(target) - relative_other(nominal))[0, 1], 0)

    def test_joint_limits_preserve_outside_targets_without_clamping(self):
        joints = np.zeros((3, 22))
        joints[0, 6], joints[1, 9], joints[2, 0] = 1.2, -1.3, 1.
        outside = joint_outside(joints, np.full(22, -1.), np.ones(22))
        self.assertEqual(int(outside.sum()), 2)
        self.assertTrue(outside[0, 6])
        self.assertTrue(outside[1, 9])
        self.assertEqual(joints[0, 6], 1.2)
        with self.assertRaises(ValueError):
            joint_outside(joints, np.ones(22), np.ones(22))

    def test_empty_condition_keeps_null_and_finite_statistics(self):
        values = np.array([-.003, .004])
        self.assertEqual(describe(values, np.array([False, False])),
                         dict(mean=None, rms=None, abs_p95=None, abs_max=None))
        summary = describe(values, np.array([True, True]))
        self.assertAlmostEqual(summary["mean"], .0005)
        self.assertAlmostEqual(summary["rms"], np.sqrt(.0000125))
        with self.assertRaises(ValueError):
            describe(values, np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()
