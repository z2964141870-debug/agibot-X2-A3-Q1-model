"""Canonical masks, root-relative metrics, and actual shape support geometry."""

import unittest

import numpy as np

from native_metrics import support_point, tracking


class TrackingTests(unittest.TestCase):
    def test_masks_absent_joints_excludes_reset_and_ignores_root_translation(self):
        reference = np.zeros((4, 269))
        actual = reference.copy()
        actual[:, 58:100] += np.tile([10., 20., 30.], 14)
        actual[0, :29] = 100.
        actual[1:, 1] = 99.
        actual[1:, 0] = [2., 4., 8.]
        present = np.zeros(29, dtype=bool)
        present[0] = True
        metrics = tracking(actual, reference, present, 2)
        self.assertAlmostEqual(metrics["joint_rmse_rad"], np.sqrt(10.))
        self.assertEqual(metrics["body_relative_rmse_m"], 0.)
        actual[1:3, 61] += 3.
        self.assertAlmostEqual(tracking(actual, reference, present, 2)["body_relative_rmse_m"],
                               3. / np.sqrt(39.))

    def test_missing_prefix_and_nonfinite_data_cannot_publish_errors(self):
        states = np.zeros((3, 269))
        mask = np.ones(29, dtype=bool)
        for steps in (0, 3, True):
            with self.subTest(steps=steps), self.assertRaises(RuntimeError):
                tracking(states, states, mask, steps)
        states[1, 0] = np.nan
        with self.assertRaises(RuntimeError):
            tracking(states, states, mask, 1)


class ShapeSupportTests(unittest.TestCase):
    def test_matches_analytic_support_in_an_oblique_direction(self):
        size, direction = np.array([2., 3., 4.]), np.array([2., -3., -6.]) / 7.
        expected = dict(sphere=2., box=float(size @ np.abs(direction)),
            ellipsoid=float(np.linalg.norm(size * direction)), capsule=2. + 3. * abs(direction[2]),
            cylinder=2. * np.linalg.norm(direction[:2]) + 3. * abs(direction[2]))
        for kind, distance in expected.items():
            with self.subTest(kind=kind):
                point = support_point(kind, size, direction)
                self.assertAlmostEqual(float(point @ direction), distance)
        vertices = np.array([[0., 0., 0.], [2., -3., -4.], [-2., 3., 4.]])
        np.testing.assert_equal(support_point("mesh", size, direction, vertices), vertices[1])

    def test_rejects_invalid_geometry_and_direction(self):
        for kind, size, direction, vertices in (
            ("mesh", [1., 1., 1.], [0., 0., -1.], [[np.nan, 0., 0.]]),
            ("ellipsoid", [0., 0., 0.], [0., 0., -1.], None),
            ("sphere", [1., 0., 0.], [0., 0., 0.], None),
            ("box", [-1., 1., 1.], [0., 0., -1.], None)):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                support_point(kind, size, direction, vertices)


if __name__ == "__main__":
    unittest.main()
