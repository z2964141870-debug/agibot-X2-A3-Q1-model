import unittest

import numpy as np

from diagnose_amass_mapping import shared_frame_bound


class SharedFrameBoundTests(unittest.TestCase):
    def test_midpoint_attains_the_two_target_lower_bound(self):
        first = np.array([[0., 0., 0.], [1., 0., 0.]])
        second = np.array([[.2, 0., 0.], [1., .4, 0.]])
        bound, observed = shared_frame_bound(first, second, (first + second) / 2)
        np.testing.assert_allclose(bound, [.02, .08], atol=1e-15)
        np.testing.assert_allclose(observed, bound, atol=1e-15)

    def test_departure_from_midpoint_adds_twice_squared_distance(self):
        first, second = np.array([[0., 0., 0.]]), np.array([[.2, 0., 0.]])
        offset = np.array([[0., .3, 0.]])
        bound, observed = shared_frame_bound(first, second, (first + second) / 2 + offset)
        np.testing.assert_allclose(observed - bound, 2 * np.sum(offset ** 2, axis=-1), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
