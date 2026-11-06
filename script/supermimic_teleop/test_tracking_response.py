"""Known lag sign, identical support, amplitude and nonidentifiability checks."""

import unittest

import numpy as np

from tracking_response import analyze_signal, summarize


class TrackingResponseTests(unittest.TestCase):
    def test_known_positive_delay_uses_identical_support_and_preserves_raw_error(self):
        reference = np.random.default_rng(42).normal(0., .1, 400)
        actual = np.concatenate((np.zeros(3), reference[:-3]))
        row = analyze_signal(reference, actual)
        self.assertEqual(row["apparent_correlation_lag_ms"], 60)
        self.assertEqual(row["minimum_rmse_lag_ms"], 60)
        self.assertAlmostEqual(row["minimum_shifted_diagnostic_rmse_rad"], 0.)
        self.assertGreater(row["unshifted_full_rmse_rad"], .05)
        self.assertEqual(row["common_samples"], 370)
        self.assertEqual(row["reference_support_first_frame"], 15)
        self.assertEqual(row["reference_support_last_frame"], 384)
        self.assertIsNone(row["measured_end_to_end_latency_ms"])

    def test_negative_lag_is_a_lead_and_amplitude_offset_stay_visible(self):
        reference = np.random.default_rng(43).normal(0., .1, 400)
        actual = np.concatenate((1.7*reference[2:] + .2, np.zeros(2)))
        row = analyze_signal(reference, actual)
        self.assertEqual(row["apparent_correlation_lag_ms"], -40)
        self.assertAlmostEqual(row["descriptive_affine_gain"], 1.7)
        self.assertAlmostEqual(row["descriptive_affine_offset_rad"], .2)
        self.assertLess(row["affine_residual_rmse_rad"], 1e-14)
        self.assertGreater(row["unshifted_full_rmse_rad"], .1)

    def test_constant_low_excitation_and_short_traces_are_not_delay_measurements(self):
        constant = analyze_signal(np.ones(100), np.ones(100)*2)
        self.assertIn("LOW_EXCITATION", constant["flags"])
        self.assertIsNone(constant["apparent_correlation_lag_ms"])
        self.assertFalse(constant["candidate_is_narrow_high_correlation"])
        short = analyze_signal(np.arange(35)*.1, np.arange(35)*.1)
        self.assertIn("INSUFFICIENT_COMMON_SUPPORT", short["flags"])
        self.assertIsNone(short["apparent_correlation_lag_ms"])

    def test_periodic_ambiguity_and_boundary_peaks_are_explicit(self):
        reference = np.sin(np.arange(400)*2*np.pi/10)
        ambiguous = analyze_signal(reference, reference)
        self.assertIn("BROAD_OR_MULTIPLE_CORRELATION_PEAKS", ambiguous["flags"])
        self.assertFalse(ambiguous["candidate_is_narrow_high_correlation"])
        self.assertIn(-200, ambiguous["near_peak_lags_ms"])
        self.assertIn(200, ambiguous["near_peak_lags_ms"])
        reference = np.random.default_rng(44).normal(0., .1, 400)
        actual = np.concatenate((np.zeros(15), reference[:-15]))
        row = analyze_signal(reference, actual)
        self.assertEqual(row["apparent_correlation_lag_ms"], 300)
        self.assertIn("CORRELATION_PEAK_AT_SEARCH_BOUNDARY", row["flags"])

    def test_invalid_traces_rejected_and_no_missing_value_becomes_zero_lag(self):
        for reference, actual in (([0., 1.], [0.]), ([0., np.nan], [0., 1.]),
                                   (np.ones((3, 2)), np.ones((3, 2)))):
            with self.assertRaises(ValueError):
                analyze_signal(reference, actual)
        weak = analyze_signal(np.ones(100), np.ones(100))
        summary = summarize([dict(joints=[weak])])
        self.assertEqual(summary["narrow_candidates"], 0)
        self.assertIsNone(summary["narrow_candidate_lag_ms"])
        self.assertIsNone(summary["all_observable_correlation_lag_ms"])


if __name__ == "__main__":
    unittest.main()
