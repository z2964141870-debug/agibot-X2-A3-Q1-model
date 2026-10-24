import unittest

import numpy as np

from diagnose_amass_mapping import rigid_pair_bound


class RigidPairBoundTests(unittest.TestCase):
    def test_midpoint_attains_the_two_target_lower_bound(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.]])
        second = np.array([[.2, 0., 0.], [1., .4, 0.]])
        midpoint = (first + second) / 2
        bound, observed = rigid_pair_bound(first, second, midpoint, midpoint, 0.)
        np.testing.assert_allclose(bound, [.02, .08], atol=1e-15)
        np.testing.assert_allclose(observed, bound, atol=1e-15)

    def test_departure_from_midpoint_adds_twice_squared_distance(self):
        first, second = np.array([[0., 0., 0.]]), np.array([[.2, 0., 0.]])
        offset = np.array([[0., .3, 0.]])
        actual = (first + second) / 2 + offset
        bound, observed = rigid_pair_bound(first, second, actual, actual, 0.)
        np.testing.assert_allclose(observed - bound, 2 * np.sum(offset ** 2, axis=-1), atol=1e-15)

    def test_fixed_link_can_fit_targets_at_its_actual_distance(self):
        first, second = np.array([[0., 0., 0.]]), np.array([[.13, 0., 0.]])
        bound, observed = rigid_pair_bound(first, second, first, second, .13)
        np.testing.assert_array_equal(bound, 0.)
        np.testing.assert_array_equal(observed, 0.)

    def test_rigid_pair_attains_bound_when_centered_on_stretched_targets(self):
        first, second = np.array([[0., 0., 0.]]), np.array([[.2, 0., 0.]])
        bound, observed = rigid_pair_bound(first, second, [[.05, 0., 0.]], [[.15, 0., 0.]], .1)
        np.testing.assert_allclose(bound, [.005], atol=1e-15)
        np.testing.assert_allclose(observed, bound, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
