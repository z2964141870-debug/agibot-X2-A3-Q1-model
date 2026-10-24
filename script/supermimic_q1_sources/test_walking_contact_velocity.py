"""Reject low-height swing motion while preserving fixed, unmodified train targets."""

import unittest

import numpy as np

from probe_walking_contact_velocity import refine_stance


class VelocityStanceTests(unittest.TestCase):
    def test_low_feet_moving_fast_are_not_stance(self):
        positions = np.zeros((21, 2, 3))
        positions[:, :, 0] = np.arange(21)[:, None] * .02
        self.assertFalse(refine_stance(positions, np.ones((21, 2), bool)).any())

    def test_stationary_contact_preserved_and_inputs_unchanged(self):
        positions = np.zeros((21, 2, 3))
        positions[:, :, 0] = np.arange(21)[:, None] * .001
        mask = np.ones((21, 2), bool)
        mask[:, 1] = False
        before, before_mask = positions.copy(), mask.copy()
        result = refine_stance(positions, mask)
        np.testing.assert_array_equal(result, mask)
        np.testing.assert_array_equal(positions, before)
        np.testing.assert_array_equal(mask, before_mask)

    def test_no_label_forced_for_missing_contact_and_bad_shape_rejected(self):
        self.assertFalse(refine_stance(np.zeros((12, 2, 3)), np.zeros((12, 2), bool)).any())
        for value in (np.full((12, 2, 3), np.nan), np.zeros((3, 2, 3)), np.zeros((12, 3, 3))):
            with self.assertRaises(ValueError):
                refine_stance(value, np.ones((12, 2), bool))


if __name__ == "__main__":
    unittest.main()
