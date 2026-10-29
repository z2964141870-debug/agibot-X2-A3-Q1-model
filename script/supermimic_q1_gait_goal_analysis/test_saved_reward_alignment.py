"""Preserve meaningful denominators and undefined gradient comparisons."""

import unittest

import numpy as np

from saved_reward_alignment import alignment, finite_mean, median_over_repeats, summarize_alignment


class RewardAlignmentTests(unittest.TestCase):
    def test_sign_and_scale_in_common_coordinates(self):
        first = np.zeros((3, 12))
        first[:, 0] = 2.
        second = first * np.array([3., -1., 1.])[:, None]
        second[2] = 0.
        second[2, 1] = 1.
        cosine, valid, _, _ = alignment(first, second)
        np.testing.assert_allclose(cosine, [1., -1., 0.])
        self.assertTrue(valid.all())
        summary = summarize_alignment(first, second, np.ones(3, bool))
        self.assertAlmostEqual(summary["opposing_fraction"], 1/3)
        self.assertEqual(summary["cosine_median"], 0.)

    def test_zero_gradients_are_undefined(self):
        first, second = np.zeros((3, 6)), np.ones((3, 6))
        second[1] = 0.
        first[2] = 1e-12
        cosine, valid, _, _ = alignment(first, second)
        self.assertTrue(np.isnan(cosine).all())
        self.assertFalse(valid.any())
        summary = summarize_alignment(first, second, np.ones(3, bool))
        self.assertEqual(summary["first_zero_frames"], 3)
        self.assertEqual(summary["second_zero_frames"], 1)
        self.assertIsNone(summary["opposing_fraction"])
        self.assertIsNone(summary["cosine_median"])

    def test_conditions_keep_their_own_denominator(self):
        first, second = np.ones((4, 12)), np.ones((4, 12))
        second[1:] *= -1.
        summary = summarize_alignment(first, second, np.array([True, False, False, False]))
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(summary["opposing_fraction"], 0.)
        empty = summarize_alignment(first, second, np.zeros(4, bool))
        self.assertEqual(empty["valid_gradient_pairs"], 0)
        self.assertIsNone(empty["first_to_second_norm_median"])
        self.assertIsNone(finite_mean(np.ones(4), np.zeros(4, bool)))

    def test_nonfinite_and_mixed_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            alignment(np.ones((2, 3)), np.ones((2, 3)))
        with self.assertRaises(ValueError):
            alignment(np.full((2, 12), np.nan), np.ones((2, 12)))
        with self.assertRaises(ValueError):
            summarize_alignment(np.ones((2, 12)), np.ones((2, 12)), np.ones(2))

    def test_plot_repeat_aggregation_keeps_gaps(self):
        actual = median_over_repeats([np.array([np.nan, .1, .3]), np.array([np.nan, .3, np.nan])])
        self.assertTrue(np.isnan(actual[0]))
        np.testing.assert_allclose(actual[1:], [.2, .3])


if __name__ == "__main__":
    unittest.main()
