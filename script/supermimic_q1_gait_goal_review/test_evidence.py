"""Failure-prefix, zero-baseline and safe-ranking counterexamples."""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "supermimic_q1_gait_goal"))
from test_physical_gate import rows
from evidence import candidate_rank, forward_metrics, physical_decision, review_tables


class EvidenceTests(unittest.TestCase):
    def test_failed_prefix_keeps_full_reference_direction(self):
        full = [[0., 0.], [0., 1.], [2., 1.], [2., 0.]]
        actual = [.2, .5]
        result = forward_metrics(full, actual, [0., 0.])
        self.assertEqual(result["reference_displacement_m"], 2.)
        self.assertAlmostEqual(result["forward_displacement_m"], .2)
        self.assertEqual(full, [[0., 0.], [0., 1.], [2., 1.], [2., 0.]])

    def test_early_fall_is_a_saved_severe_failure(self):
        candidate = rows(.75)
        candidate[0].update(steps=7, failed=True)
        result = physical_decision(rows(), candidate)
        self.assertFalse(result["fixed_numeric_passed"])
        self.assertTrue(result["physical_regression_stop"])

    def test_missing_loaded_or_swing_observation_stops_without_fabrication(self):
        for key in ("loaded_slip_mps", "swing_sole_z_rmse_m"):
            candidate = rows(.75)
            candidate[0][key] = None
            result = physical_decision(rows(), candidate)
            self.assertTrue(result["physical_regression_stop"])
            self.assertIsNone(candidate[0][key])
            self.assertIn("metric_error", result)

    def test_zero_guard_baseline_cannot_escape_severe_stop(self):
        baseline, candidate = rows(), rows(.75)
        baseline[0]["upper_rmse"] = 0.
        candidate[0]["upper_rmse"] = .001
        result = physical_decision(baseline, candidate)
        self.assertIsNone(result["per_repeat"][0]["guards"]["upper_rmse"]["ratio"])
        self.assertTrue(result["physical_regression_stop"])

    def test_safe_baseline_outranks_guard_regression_despite_large_gains(self):
        baseline, candidate = rows(), rows(.1)
        candidate[0]["upper_rmse"] *= 1.11
        initial = dict(label="initial", training_seconds=0., decision=physical_decision(baseline, baseline))
        trained = dict(label="stage_01", training_seconds=301., decision=physical_decision(baseline, candidate))
        self.assertLess(candidate_rank(initial), candidate_rank(trained))

    def test_valid_gain_outranks_initial_without_settling_goal(self):
        initial = dict(label="initial", training_seconds=0., decision=physical_decision(rows(), rows()))
        trained = dict(label="stage_01", training_seconds=301., decision=physical_decision(rows(), rows(.75)))
        self.assertLess(candidate_rank(trained), candidate_rank(initial))
        self.assertFalse(trained["decision"]["overall_goal_achieved"])

    def test_fall_cannot_win_a_ranking(self):
        baseline, candidate = rows(), rows(.1)
        candidate[0].update(steps=1, failed=True)
        initial = dict(label="initial", training_seconds=0., decision=physical_decision(baseline, baseline))
        trained = dict(label="stage_01", training_seconds=301., decision=physical_decision(baseline, candidate))
        before = copy.deepcopy(candidate)
        self.assertLess(candidate_rank(initial), candidate_rank(trained))
        self.assertEqual(before, candidate)

    def test_full_library_tables_preserve_failures_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory)
            (trial / "run").mkdir()
            (trial / "evaluation").mkdir()
            (trial / "run/checkpoint_selection.json").write_text(json.dumps(dict(candidates=[dict(label="initial")])))
            for repeat in range(3):
                per_clip = [dict(name="clip" + str(i), split="train" if i < 300 else "validation",
                    group="walking", failed=i == 0, completion=.5 if i == 0 else 1.,
                    root_xy_rmse=.1, upper_rmse=.2, leg_rmse=.3, wrist_relative_rmse=.04,
                    loaded_slip=None if i == 0 else .1, target_oscillation=.05) for i in range(330)]
                (trial / f"evaluation/initial_repeat{repeat}.json").write_text(json.dumps(dict(
                    training_seconds=0., per_clip=per_clip, goal_fixed=[row for row in rows() if row["repeat"] == repeat])))
            result = review_tables(trial)
            self.assertEqual(result["per_clip_rows"], 990)
            self.assertEqual(result["fixed_rows"], 9)
            with (trial / "review/per_clip.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["loaded_slip"], "")
            self.assertEqual(saved[0]["failed"], "True")
            with (trial / "review/split_group.csv").open() as stream:
                groups = list(csv.DictReader(stream))
            training = next(row for row in groups if row["split"] == "train")
            self.assertEqual(training["failures"], "1")
            self.assertEqual(training["loaded_slip_observations"], "299")


if __name__ == "__main__":
    unittest.main()
