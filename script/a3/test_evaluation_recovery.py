"""Recovery keeps validated motions and preserves invalid output evidence."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.a3.checkpoint_store import sha256
from script.a3.recover_evaluation import recover


class RecoveryTests(unittest.TestCase):
    def test_valid_motion_is_kept_and_changed_motion_is_quarantined(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data/eval"
            output.mkdir(parents=True)
            good = output / "good.metrics.json"
            good.write_text(json.dumps({"motions": [{"num_policy_steps": 1}]}))
            (output / "good.timeseries.json").write_text(json.dumps({"motions": [{}]}))
            bad = output / "bad.metrics.json"
            bad.write_text("interrupted")
            manifest = {"protocol": {}, "runs": {
                "good": {"status": "passed", "metrics_sha256": sha256(good)},
                "bad": {"status": "passed", "metrics_sha256": "invalid", "attempt": 1}}}
            (output / "explicit_manifest.json").write_text(json.dumps(manifest))
            with patch("script.a3.recover_evaluation.ROOT", root), patch(
                    "script.a3.recover_evaluation.subprocess.run", return_value=SimpleNamespace(returncode=1)), patch(
                    "script.a3.recover_evaluation.rollout_windows"):
                recover(output)
            saved = json.loads((output / "explicit_manifest.json").read_text())
            self.assertTrue(good.exists())
            self.assertEqual(saved["runs"]["good"]["status"], "passed")
            self.assertEqual(saved["runs"]["bad"]["status"], "interrupted_artifacts_unverified")
            kept = Path(saved["runs"]["bad"]["preserved_artifacts"])
            self.assertEqual((kept / bad.name).read_text(), "interrupted")
            self.assertEqual(saved["runs"]["bad"]["attempt"], 1)

    def test_active_evaluator_prevents_mutation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("script.a3.recover_evaluation.ROOT", root), patch(
                    "script.a3.recover_evaluation.subprocess.run", return_value=SimpleNamespace(returncode=0)):
                with self.assertRaises(RuntimeError):
                    recover(root / "data/eval")


if __name__ == "__main__":
    unittest.main()
