"""Check phase boundaries, conditional denominators and paired saved comparisons."""

import unittest

import numpy as np

from saved_phase_tradeoff import average, foot_conditions, frame_metrics, paired_regions, phases, regions


class SavedPhaseTests(unittest.TestCase):
    def sample(self, count=7):
        return dict(height=np.zeros((count, 2)), reference_height=np.full((count, 2), .02),
            force=np.tile([6., 5.], (count, 1)), slip=np.tile([.1, 100.], (count, 1)),
            tilt=np.arange(1, count+1, dtype=float) / 100,
            frames=np.arange(count), swing=np.ones((count, 2), bool), stance=np.zeros((count, 2), bool))

    def test_mixed_contact_is_neutral(self):
        stance, swing = phases(np.array([[True, False], [False, True], [False, True]]), 4)
        np.testing.assert_array_equal(stance[0], [True, False])
        self.assertFalse(stance[1].any() or swing[1].any())
        np.testing.assert_array_equal(swing[2], [True, False])
        with self.assertRaises(ValueError):
            phases([[1, 0], [0, 1]], 3)

    def test_empty_condition_is_missing(self):
        self.assertIsNone(average(np.ones(3), np.zeros(3, bool)))
        with self.assertRaises(ValueError):
            average(np.array([np.nan]), np.ones(1, bool))

    def test_native_threshold_and_per_foot_denominators(self):
        value = self.sample()
        metrics = frame_metrics(value, np.ones(7, bool))
        self.assertAlmostEqual(metrics["loaded_slip_mps"], .1)
        self.assertEqual(metrics["loaded_foot_frames"], 7)
        rows = foot_conditions(value)
        right = next(row for row in rows if row["side"] == "right" and row["condition"] == "native_loaded")
        self.assertEqual(right["foot_frames"], 0)
        self.assertIsNone(right["height_rmse_m"])

    def test_quarters_partition_uneven_lengths(self):
        value = self.sample()
        quarters = regions(7)[1:]
        np.testing.assert_array_equal(sum(mask.astype(int) for _, mask in quarters), np.ones(7, int))
        weighted = sum(frame_metrics(value, mask)["tilt_rms_rad"] ** 2 * mask.sum() for _, mask in quarters)
        self.assertAlmostEqual(np.sqrt(weighted / 7), frame_metrics(value, np.ones(7, bool))["tilt_rms_rad"])

    def test_pair_requires_identical_reference(self):
        before, after = self.sample(), self.sample()
        after["slip"][:, 0] *= .8
        after["tilt"] *= 1.3
        self.assertTrue(paired_regions(before, after)[0]["slip_down_and_tilt_up"])
        after["force"][0, 0] = 0.
        paired = paired_regions(before, after)[0]
        self.assertEqual(paired["common_loaded_foot_frames"], 6)
        self.assertAlmostEqual(paired["common_loaded_slip_initial"], .1)
        self.assertAlmostEqual(paired["common_loaded_slip_candidate"], .08)
        self.assertAlmostEqual(paired["initial_load_lost_fraction"], 1/7)
        after["frames"][0] += 1
        with self.assertRaises(AssertionError):
            paired_regions(before, after)


if __name__ == "__main__":
    unittest.main()
