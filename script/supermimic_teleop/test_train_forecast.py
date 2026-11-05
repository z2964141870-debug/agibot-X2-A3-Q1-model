"""Admission and budget checks for the independent forecast trainer."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from train_forecast import DEFAULT_BOOT, FORMAL_TRAINING, ForecastBudget, gpu_lock, windows


class TrainForecastTests(unittest.TestCase):
    def test_window_enumeration_is_source_local_and_excludes_future_padding(self):
        plan = dict(history_frames=25, future_frames=10, clips=[
            dict(relative="a/a.npz", split="train", frames=35),
            dict(relative="b/b.npz", split="validation", frames=60),
        ])
        train = list(windows(plan, "train"))
        self.assertEqual(train, [("a/a.npz", 24)])
        validation = list(windows(plan, "validation"))
        self.assertEqual(validation[0], ("b/b.npz", 24))
        self.assertEqual(validation[-1], ("b/b.npz", 49))

    def test_formal_training_root_is_separate_from_independent_outputs(self):
        self.assertIn("MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104", str(FORMAL_TRAINING))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "forecast"
            self.assertFalse(output.is_relative_to(FORMAL_TRAINING))

    def test_whole_experiment_deadline_applies_during_preparation_and_validation(self):
        clock = [0.]
        with tempfile.TemporaryDirectory() as tmp, \
                patch("train_forecast.time.monotonic", side_effect=lambda: clock[0]), \
                patch("train_forecast.resources", return_value={}) as sample, \
                patch("train_forecast.Path.read_text", return_value=DEFAULT_BOOT):
            budget = ForecastBudget(tmp, max_seconds=10.)
            budget.check(start=True)
            clock[0] = 3.
            budget.check()
            clock[0] = 11.
            with self.assertRaisesRegex(RuntimeError, "whole-experiment wall"):
                budget.check()
            self.assertEqual(sample.call_count, 2)

    def test_cpu_diagnostics_allow_original_compute_but_training_requires_idle_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("train_forecast.resources", return_value={}) as sample, \
                patch("train_forecast.Path.read_text", return_value=DEFAULT_BOOT):
            ForecastBudget(tmp, max_seconds=10., require_idle=False).check(start=True)
            self.assertFalse(sample.call_args.kwargs["require_idle"])
            ForecastBudget(tmp, max_seconds=10.).check(start=True)
            self.assertTrue(sample.call_args.kwargs["require_idle"])
        for value in (0, -1, 1801, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                ForecastBudget("unused", max_seconds=value)

    def test_gpu_lock_denies_a_competing_experiment_without_disturbing_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            with gpu_lock(tmp):
                with self.assertRaisesRegex(RuntimeError, "owns the Q1 lock"):
                    with gpu_lock(tmp):
                        self.fail("Competing experiment entered the lock")
            with gpu_lock(tmp):
                pass


if __name__ == "__main__":
    unittest.main()
