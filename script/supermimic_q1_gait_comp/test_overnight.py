"""Counterexamples for deadlines, bounded rates and per-walk continuations."""

import copy
from datetime import datetime, timezone
import unittest

from overnight import (OPTIMIZATION_DEADLINE, DELIVERY_DEADLINE, bounded_learning_rate,
                       continuing_evidence, next_check)


def evidence():
    return {"walking_ratios": [dict(walking_slip=.85, swing_error=1.01, walking_tilt=1.01, target_change=.99)
                               for _ in range(2)],
            "ratios": dict(upper_rmse=1.01, wrist_rmse=1.01, raised_peak_error=1.01),
            "stationary_drift_m": .09, "baseline_stationary_drift_m": .08,
            "validation_failures": 0, "validation_completion": 1.,
            "fixed_demos_complete": True, "walking_progress_preserved": True}


class OvernightTests(unittest.TestCase):
    def test_exact_deadlines_and_two_hour_delivery_reserve(self):
        self.assertEqual(datetime.fromtimestamp(OPTIMIZATION_DEADLINE, timezone.utc).isoformat(), "2026-10-28T23:00:00+00:00")
        self.assertEqual(DELIVERY_DEADLINE-OPTIMIZATION_DEADLINE, 7200)

    def test_rsl_adaptive_increase_is_capped(self):
        self.assertEqual(bounded_learning_rate(.01), 5e-5)
        self.assertEqual(bounded_learning_rate(1e-7), 1e-5)
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bounded_learning_rate(value)

    def test_checks_do_not_skip_early_physics(self):
        self.assertEqual(next_check(0.), 1200.)
        self.assertEqual(next_check(1201.), 2400.)
        self.assertEqual(next_check(3501.), 7100.)
        self.assertIsNone(next_check(14301.))

    def test_improve_either_foot_objective_on_both_walks(self):
        self.assertTrue(continuing_evidence([evidence() for _ in range(3)])["continue_authorized"])

    def test_one_walk_cannot_hide_the_other(self):
        repeats = [evidence() for _ in range(3)]
        for row in repeats:
            row["walking_ratios"][1]["swing_error"] = 1.11
        self.assertFalse(continuing_evidence(repeats)["continue_authorized"])

    def test_one_failed_demo_blocks_all_extension(self):
        repeats = [evidence() for _ in range(3)]
        repeats[2]["fixed_demos_complete"] = False
        self.assertFalse(continuing_evidence(repeats)["continue_authorized"])

    def test_nan_and_missing_are_not_improvement(self):
        for value in (None, float("nan"), float("inf")):
            repeats = [evidence() for _ in range(3)]
            for row in repeats:
                row["walking_ratios"][0]["walking_slip"] = value
            self.assertFalse(continuing_evidence(repeats)["continue_authorized"])

    def test_raise_drift_and_upper_regression_block(self):
        for field in ("stationary_drift_m", "upper_rmse"):
            repeats = [evidence() for _ in range(3)]
            for row in repeats:
                if field == "stationary_drift_m":
                    row[field] = .2
                else:
                    row["ratios"][field] = 1.16
            self.assertFalse(continuing_evidence(repeats)["continue_authorized"])


if __name__ == "__main__":
    unittest.main()
