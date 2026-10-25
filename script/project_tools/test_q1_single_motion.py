import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_q1_single_motion import CLIP, ENVS, STEPS, completed_training, matched_comparison


class SingleMotionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_stopped_training_cannot_be_evaluated_as_finished(self):
        admission = {"train_clips": [{"path": "/references/" + CLIP + ".npz"}]}
        finished = {"status": "TRAINING_FINISHED_PENDING_EVALUATION", "step": STEPS,
                    "env_steps": STEPS * ENVS, "error": None}
        self.write("admission.json", admission)
        self.write("receipt.json", finished)
        self.assertEqual(completed_training(self.root), finished)
        for key, value in (("status", "STOPPED_SIGNAL"), ("step", 1000),
                           ("env_steps", 64000), ("error", "worker failed")):
            with self.subTest(key=key):
                partial = {**finished, key: value}
                self.write("receipt.json", partial)
                with self.assertRaises(ValueError):
                    completed_training(self.root)

    def test_actual_multiclip_admission_is_rejected(self):
        self.write("receipt.json", {"status": "TRAINING_FINISHED_PENDING_EVALUATION",
                   "step": STEPS, "env_steps": STEPS * ENVS, "error": None})
        self.write("admission.json", {"train_clips": [
            {"path": "/references/" + CLIP + ".npz"}, {"path": "/references/other.npz"}]})
        with self.assertRaises(ValueError):
            completed_training(self.root)

    def test_timed_receipt_requires_actual_duration_and_2048_transitions(self):
        self.write("admission.json", {"train_clips": [{"path": "/references/" + CLIP + ".npz"}]})
        receipt = {"status": "TRAINING_FINISHED_PENDING_EVALUATION", "step": 1000,
                   "env_steps": 1000 * 2048, "training_seconds": 7200.1, "error": None}
        self.write("receipt.json", receipt)
        self.assertEqual(completed_training(self.root, 2048, 12000, 7200), receipt)
        for changes in ({"training_seconds": 7199}, {"env_steps": 64000}, {"step": 999}):
            self.write("receipt.json", {**receipt, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                completed_training(self.root, 2048, 12000, 7200)

    def test_pair_preserves_failure_and_rejects_reference_mismatch(self):
        receipt = {"rows": [{"clip": CLIP, "failed": True,
                            "reference": {"path": "/reference.npz", "sha256": "a"}}],
                   "groups": {"train": {"failures": 1}}}
        for label in ("initial", "final"):
            self.write("evaluation_" + label + "/receipt.json", receipt)
        comparison = matched_comparison(self.root)
        self.assertTrue(comparison["pairs"][0]["final"]["failed"])
        self.assertFalse(comparison["performance_thresholds_used"])
        mismatched = copy.deepcopy(receipt)
        mismatched["rows"][0]["reference"]["sha256"] = "b"
        self.write("evaluation_final/receipt.json", mismatched)
        with self.assertRaises(ValueError):
            matched_comparison(self.root)


if __name__ == "__main__":
    unittest.main()
