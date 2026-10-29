"""Exact phase boundaries and scale-saturation counterexamples."""

import unittest
import numpy as np

from calibrate_saved import merge_review_rows, phase_labels, scale_statistics


class CalibrationTests(unittest.TestCase):
    def test_shared_baseline_rows_must_be_identical_across_tables(self):
        row = dict(label="baseline", repeat="0", clip="walk", steps="20")
        merged, copies = merge_review_rows([[row], [dict(row)]])
        self.assertEqual(len(merged), 1)
        self.assertEqual(copies, 1)
        for tables in (([row, dict(row)],), ([row], [dict(row, steps="19")])):
            with self.assertRaises(ValueError):
                merge_review_rows(tables)

    def test_contact_transition_is_neutral_not_swing(self):
        source = np.array([[True, False], [False, True], [False, True]])
        stance, swing = phase_labels(source, np.arange(4))
        np.testing.assert_array_equal(stance[0], [True, False])
        np.testing.assert_array_equal(swing[0], [False, True])
        self.assertFalse(stance[1].any())
        self.assertFalse(swing[1].any())
        np.testing.assert_array_equal(stance[2], [False, True])
        np.testing.assert_array_equal(swing[2], [True, False])

    def test_static_or_empty_phase_has_no_fake_gain(self):
        value = scale_statistics(np.zeros((3, 2)), np.zeros((3, 2), dtype=bool), .02)
        self.assertEqual(value["foot_frames"], 0)
        self.assertIsNone(value["perfect_gain_on_active_frames"])

    def test_active_feet_are_normalized_per_frame(self):
        error = np.array([[.02, 100.], [.02, .02], [100., 100.]])
        mask = np.array([[True, False], [True, True], [False, False]])
        value = scale_statistics(error, mask, .02)
        self.assertAlmostEqual(value["score_on_active_frames"], np.exp(-1))
        self.assertAlmostEqual(value["perfect_gain_on_active_frames"], 1 - np.exp(-1))
        self.assertAlmostEqual(value["rmse_m"], .02)

    def test_too_narrow_scale_saturates_large_error(self):
        error = np.full((2, 2), .04)
        mask = np.ones_like(error, dtype=bool)
        narrow = scale_statistics(error, mask, .01)
        broad = scale_statistics(error, mask, .02)
        self.assertLess(narrow["absolute_gradient_per_m"], broad["absolute_gradient_per_m"])
        self.assertEqual(narrow["gradient_below_1_per_m_fraction"], 1.)

    def test_invalid_metadata_and_nonfinite_values(self):
        for contact in (np.ones((2, 2)), np.ones((2, 1), dtype=bool)):
            with self.assertRaises(ValueError):
                phase_labels(contact, np.arange(2))
        with self.assertRaises(ValueError):
            phase_labels(np.zeros((2, 2), dtype=bool), np.array([.5]))
        with self.assertRaises(ValueError):
            scale_statistics(np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool), .02)


if __name__ == "__main__":
    unittest.main()
