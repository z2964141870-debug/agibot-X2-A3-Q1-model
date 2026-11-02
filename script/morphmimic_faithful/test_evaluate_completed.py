"""Reject incomplete jobs before any deployment or physics call."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_completed import completed_training


class CompletionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "run").mkdir()
        self.states = {
            "status.json": {"status": "COMPLETED"},
            "guard.json": {"status": "WORKER_COMPLETED", "cleanup": {
                "worker_reaped": True, "worker_return_code": 0}},
            "run/progress.json": {"status": "TRAINING_COMPLETED", "iteration": 160000,
                "nonfinite_steps": 0, "optimizer_steps": {"wm_opt": 1600000,
                    "policy_opt": 158000, "critic_opt": 3160000,
                    "kp_opt": 158000, "critic_kp_opt": 3160000}},
        }

    def check(self):
        for name, data in self.states.items():
            (self.root / name).write_text(json.dumps(data))
        return completed_training(self.root)

    def test_clean_full_run_is_accepted(self):
        self.assertEqual(self.check()[1]["iteration"], 160000)

    def test_completed_label_cannot_hide_short_run(self):
        self.states["run/progress.json"]["iteration"] = 40000
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.check()

    def test_completed_label_cannot_hide_nonzero_exit(self):
        self.states["guard.json"]["cleanup"]["worker_return_code"] = 1
        with self.assertRaisesRegex(ValueError, "not completed"):
            self.check()

    def test_missing_keypoint_updates_are_rejected(self):
        del self.states["run/progress.json"]["optimizer_steps"]["kp_opt"]
        with self.assertRaisesRegex(ValueError, "five optimizer"):
            self.check()

    def test_still_present_original_pid_is_rejected(self):
        with patch("evaluate_completed.Path.exists", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "PID is still present"):
                self.check()


if __name__ == "__main__":
    unittest.main()
