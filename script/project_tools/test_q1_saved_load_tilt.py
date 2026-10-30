"""Saved force/pose clock and unloading counterexamples without runtime imports."""

import csv
import math
from pathlib import Path
import tempfile
import unittest

from analyze_q1_saved_load_tilt import analyze, paired_indices, read_root


def samples(force, other=None, height=None, reference=None):
    count = len(force)
    other = other if other is not None else [20.] * count
    height = height if height is not None else [.003] * count
    reference = reference if reference is not None else [True] * count
    feet = {side: [dict(native_force_N=f, sole_z_m=h, reference_sole_z_m=.02,
        intent_swing=ref, training_seconds=0.) for f, h, ref in zip(values, height, reference)]
        for side, values in (("right", force), ("left", other))}
    roots = [dict(local_y_tilt_delta_rad=.05, projection_y_m=-.01) for _ in force]
    return feet, roots


class SavedLoadTiltTests(unittest.TestCase):
    def evaluate(self, force, other=None, height=None, reference=None, shift=0):
        feet, roots = samples(force, other, height, reference)
        return analyze(("ppo", "stage_02", 0, "synthetic"), feet, roots, "right", 5., shift)

    def test_low_force_runs_can_remain_near_ground(self):
        row, episodes = self.evaluate([0.] * 5)
        self.assertEqual(row["low_runs_ge3"], 1)
        self.assertEqual(row["low_runs_ge3_with_high_ge3"], 0)
        self.assertEqual(row["low_other_loaded_near5_fraction"], 1.)
        self.assertAlmostEqual(episodes[0]["endpoint_span_s"], .08)

    def test_strict_force_height_and_opposite_support_boundaries(self):
        row, _ = self.evaluate([5., 5.00001, 0., 0.], other=[5.00001, 20., 5., 20.], height=[.005, .005, .02, .010])
        self.assertEqual(row["low_other_loaded_frames"], 2)
        self.assertEqual(row["loaded_frames"], 1)
        self.assertEqual(row["both_low_frames"], 1)
        self.assertEqual(row["low_other_loaded_near5_frames"], 1)
        self.assertEqual(row["low_other_loaded_high10_frames"], 0)

    def test_reference_gaps_and_missing_load_split_low_runs(self):
        row, episodes = self.evaluate([0.] * 7, other=[20., 20., 20., 0., 20., 20., 20.])
        self.assertEqual([r["samples"] for r in episodes], [3, 3])
        self.assertEqual(row["low_runs_ge3"], 2)
        row, _ = self.evaluate([0.] * 7, reference=[True, True, True, False, True, True, True])
        self.assertEqual(row["low_runs_ge3"], 2)

    def test_clearance_requires_three_samples_within_one_low_run(self):
        row, _ = self.evaluate([0.] * 6, height=[.02, .02, .003, .02, .02, .003])
        self.assertEqual(row["low_runs_ge3_with_high_ge3"], 0)
        row, _ = self.evaluate([0.] * 6, height=[.02, .02, .02, .003, .003, .003])
        self.assertEqual(row["low_runs_ge3_with_high_ge3"], 1)

    def test_shifted_pairs_do_not_wrap_and_keep_force_window_anchor(self):
        self.assertEqual(paired_indices(3, -1), [(1, 0), (2, 1)])
        self.assertEqual(paired_indices(3, 1), [(0, 1), (1, 2)])
        row, _ = self.evaluate([0.] * 4, height=[.003, .02, .02, .02], reference=[True, True, True, False], shift=1)
        self.assertEqual(row["eligible_reference_frames"], 3)
        self.assertEqual(row["low_other_loaded_high10_frames"], 3)
        self.assertAlmostEqual(row["pose_minus_force_sample_s"], .025)
        self.assertAlmostEqual(row["root_minus_force_sample_s"], .005)

    def test_empty_reference_or_no_low_force_preserves_null(self):
        row, episodes = self.evaluate([20.] * 4, reference=[False] * 4)
        self.assertIsNone(row["loaded_fraction_window"])
        self.assertIsNone(row["low_other_loaded_near5_fraction"])
        self.assertEqual(episodes, [])
        row, _ = self.evaluate([20.] * 4)
        self.assertEqual(row["loaded_fraction_window"], 1.)
        self.assertIsNone(row["local_y_tilt_delta_low_mean_deg"])

    def test_invalid_pairing_and_force_threshold_are_rejected(self):
        for count, shift in ((0, 0), (3, 2), (True, 0), (3, False)):
            with self.assertRaises(ValueError):
                paired_indices(count, shift)
        feet, roots = samples([0.] * 3)
        with self.assertRaises(ValueError):
            analyze(("base", "initial", 0, "synthetic"), feet, roots, "right", math.nan, 0)

    def test_root_csv_rejects_unknown_or_incomplete_clip_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "root.csv"
            path.write_text("line,label,repeat,clip\n")
            with self.assertRaises(ValueError):
                read_root(path, {("base", "initial", 0, "synthetic"): {}})

    def test_root_join_validates_frame_clock_budget_and_reference_flag(self):
        feet, _ = samples([0.] * 3)
        key = "base", "initial", 0, "synthetic"
        for side in ("left", "right"):
            for index, row in enumerate(feet[side]):
                row["global_reference_frame"] = 100 + index
        rows = [dict(line=key[0], label=key[1], repeat=key[2], clip=key[3], sample=i,
            control_start_s=i * .02, reference_sample_end_s=(i + 1) * .02,
            training_seconds=0., local_y_tilt_delta_rad=.05, projection_y_m=-.01,
            global_reference_frame=100 + i, raised_reference_left=True,
            raised_reference_right=True) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "root.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(values)
            write(rows)
            self.assertEqual(len(read_root(path, {key: feet})[key]), 3)
            for field, bad in (("sample", 0), ("global_reference_frame", 999),
                               ("control_start_s", .025), ("training_seconds", 1.),
                               ("raised_reference_right", False), ("local_y_tilt_delta_rad", math.nan)):
                with self.subTest(field=field):
                    altered = [dict(row) for row in rows]
                    altered[1][field] = bad
                    write(altered)
                    with self.assertRaises(ValueError):
                        read_root(path, {key: feet})


if __name__ == "__main__":
    unittest.main()
