"""Static projection counterexamples, without a native environment or inference."""

import unittest

import numpy as np

from analyze_q1_saved_root_projection import foot_support, projection_parts, tilt_coordinates, up_in_root


class RootProjectionTests(unittest.TestCase):
    def test_world_yaw_preserves_root_up_projection(self):
        angle, yaw = .17, 1.1
        roll = np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)], [0., np.sin(angle), np.cos(angle)]])
        heading = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        np.testing.assert_allclose(up_in_root(roll[None]), up_in_root((heading @ roll)[None]))
        np.testing.assert_allclose(tilt_coordinates(up_in_root(roll[None])), [[angle, 0.]])

    def test_lateral_projection_sign_matches_bilateral_offsets(self):
        old = np.array([[0., 0., 1.]])
        new = np.array([[0., np.sin(.1), np.cos(.1)]])
        points = np.array([[[0., .1, -.5], [0., -.1, -.5]]])
        radii = np.zeros((1, 2))
        result = projection_parts(old, new, points, radii, points, radii)
        self.assertAlmostEqual(result["root_effect_m"][0], -.2 * np.sin(.1))
        np.testing.assert_array_equal(result["projection_x_m"], 0.)
        np.testing.assert_array_equal(result["support_switch_m"], 0.)

    def test_forward_and_vertical_projection_terms_remain_separate(self):
        old = np.array([[0., 0., 1.]])
        new = np.array([[-np.sin(.2), 0., np.cos(.2)]])
        points = np.array([[[0., .1, -.5], [.1, -.1, -.48]]])
        radii = np.zeros((1, 2))
        result = projection_parts(old, new, points, radii, points, radii)
        self.assertAlmostEqual(result["projection_x_m"][0], -.1 * np.sin(.2))
        self.assertAlmostEqual(result["projection_z_m"][0], .02 * (np.cos(.2) - 1))

    def test_support_selection_respects_sphere_radii_and_switching(self):
        points = np.array([[-.1, 0., 0.], [.1, 0., .001]])
        radii = np.array([0., .004])
        position, rotation = np.zeros((1, 3)), np.eye(3)[None]
        old, new = np.array([[0., 0., 1.]]), np.array([[np.sin(.2), 0., np.cos(.2)]])
        first = foot_support(position, rotation, points, radii, old)
        second = foot_support(position, rotation, points, radii, new)
        self.assertEqual(first[3][0], 1)
        self.assertEqual(second[3][0], 0)
        self.assertAlmostEqual(first[0][0], -.003)
        old_points = np.stack((np.zeros((1, 3)), first[1]), axis=1)
        new_points = np.stack((np.zeros((1, 3)), second[1]), axis=1)
        result = projection_parts(old, new, old_points, np.stack((np.zeros(1), first[2]), axis=1),
                                  new_points, np.stack((np.zeros(1), second[2]), axis=1))
        self.assertLess(result["support_switch_m"][0], 0.)
        self.assertAlmostEqual(result["root_effect_m"][0], second[0][0] - first[0][0])

    def test_batched_support_matches_explicit_material_points(self):
        rng = np.random.default_rng(54)
        points = rng.normal(size=(13, 3))
        radii = rng.uniform(0, .1, 13)
        positions = rng.normal(size=(135, 3))
        rotations = np.repeat(np.eye(3)[None], 135, axis=0)
        normals = rng.normal(size=(135, 3))
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        actual = foot_support(positions, rotations, points, radii, normals)[0]
        expected = np.min(np.sum(normals[:, None] * (positions[:, None] + points), axis=-1) - radii, axis=-1)
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_identical_normals_and_points_have_zero_effect(self):
        normals = np.array([[0., 0., 1.]])
        points = np.array([[[.1, .2, -.5], [-.1, -.2, -.4]]])
        radii = np.array([[.01, .02]])
        result = projection_parts(normals, normals, points, radii, points, radii)
        for key in ("root_effect_m", "projection_x_m", "projection_y_m", "projection_z_m", "support_switch_m"):
            np.testing.assert_array_equal(result[key], 0.)

    def test_foot_rotation_is_applied_before_support_selection(self):
        points = np.array([[.1, 0., 0.], [-.2, 0., -.1]])
        rotation = np.array([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]])
        actual = foot_support(np.array([[0., 0., -.4]]), rotation, points, np.zeros(2), np.array([[0., 0., 1.]]))
        self.assertEqual(actual[3][0], 0)
        self.assertAlmostEqual(actual[0][0], -.5)
        np.testing.assert_allclose(actual[1], [[0., 0., -.5]])

    def test_invalid_geometry_is_rejected(self):
        for matrix in (np.zeros((1, 3, 3)), np.diag([1., 1., -1.])[None], np.full((1, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                up_in_root(matrix)
        with self.assertRaises(ValueError):
            foot_support(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)), np.zeros(1), np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            tilt_coordinates(np.array([[0., 0., 2.]]))


if __name__ == "__main__":
    unittest.main()
