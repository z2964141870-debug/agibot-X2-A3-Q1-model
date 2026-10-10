"""Protect trial isolation and the predeclared conservative update variables."""

import unittest
from unittest.mock import patch

from script.a3 import corrected_trial as trial


class SearchTrialTests(unittest.TestCase):
    def tearDown(self):
        trial.configure_trial("R06")

    def config(self, identity):
        trial.configure_trial(identity)
        with patch.object(trial, "checkpoint_metadata", return_value={"sha256": "official"}):
            return trial.configuration()

    def test_epoch_trial_retains_lr_and_safety(self):
        control = self.config("R07")
        candidate = self.config("R08")
        changed = {key for key in control if candidate[key] != control[key]}
        self.assertEqual(changed, {"num_learning_epochs", "evaluation_steps"})
        self.assertEqual(candidate["num_learning_epochs"], 1)
        self.assertIsNone(candidate["max_attempts"])
        self.assertIsNone(candidate["expires_at"])
        self.assertEqual(candidate["max_no_progress"], 2)
        self.assertEqual(candidate["cooldown_seconds"], 300)

    def test_lr_trial_is_one_scale_factor(self):
        control = self.config("R08")
        candidate = self.config("R09")
        changed = {key for key in control if candidate[key] != control[key]}
        self.assertEqual(changed, {"lr_scale"})
        self.assertEqual(candidate["lr_scale"], 0.1)
        self.assertEqual(candidate["num_learning_epochs"] * candidate["num_mini_batches"], 4)

    def test_paths_and_defaults_reset_between_trials(self):
        paths = []
        for identity in ("R06", "R07", "R08", "R09"):
            self.config(identity)
            paths.append(trial.OUTPUT)
            self.assertEqual(trial.TRIAL, identity)
        self.assertEqual(len(set(paths)), 4)
        self.config("R07")
        self.assertEqual(trial.EPOCHS, 5)
        self.assertEqual(trial.LR_SCALE, 1)
        self.assertEqual(trial.EVALUATED_STEPS, (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
