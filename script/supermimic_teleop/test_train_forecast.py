"""Admission and budget checks for the independent forecast trainer."""

import tempfile
from pathlib import Path
import unittest

from train_forecast import FORMAL_TRAINING, windows


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


if __name__ == "__main__":
    unittest.main()
