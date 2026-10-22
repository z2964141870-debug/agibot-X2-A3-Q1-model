"""Focused controls for the saved-trace policy gate and process identity checks."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import assess_policy
from complete_run import process_snapshot
from supermimic.a3_assets import identity


class CompletionControls(unittest.TestCase):
    def assess(self, directory, variation=None):
        directory = Path(directory)
        evaluation_rows, audit_rows = [], []
        for index in range(8):
            failed = variation == "early_failure" and index == 0
            count = 9 if failed else 10
            reward_value = .01 if variation == "low_return" else .1
            joint_value = .2 if variation == "bad_joint" and index == 0 else .05
            rewards, joints = np.full(count, reward_value), np.full(count, joint_value)
            path = directory / f"clip_{index}.npz"
            np.savez_compressed(path, reward=rewards, joint_rmse_rad=joints)
            row = {"clip": f"clip_{index}", "split": "training" if index < 4 else "heldout",
                   "expected_steps": 10, "executed_steps": count, "failed": failed,
                   "normalized_return": float(rewards.sum()), "full_clip_completion": count / 10,
                   "executed_joint_rmse_rad": float(np.sqrt(np.mean(joints ** 2))), "trace": identity(path)}
            evaluation_rows.append(row)
            audit_rows.append({"clip": row["clip"], "split": row["split"],
                               **{name: {"trained": row[name]} for name in
                                  ("normalized_return", "full_clip_completion", "failed", "executed_steps")}})
        evaluation_path = directory / "evaluation.json"
        evaluation_path.write_text(json.dumps({"status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED",
                                              "checkpoint": {"sha256": "synthetic-control"}, "rows": evaluation_rows}))
        audit_path = directory / "audit.json"
        audit_path.write_text(json.dumps({"status": "TRAINING_AND_MATCHED_EVALUATION_VERIFIED", "rows": audit_rows,
                                         "final_checkpoint": {"sha256": "synthetic-control"},
                                         "evaluation_receipts": [{}, identity(evaluation_path)]}))
        if variation == "tampered_trace":
            with (directory / "clip_0.npz").open("ab") as stream:
                stream.write(b"tamper")
        output = directory / "assessment.json"
        with patch.object(sys, "argv", ["assess_policy.py", "--audit", str(audit_path), "--output", str(output)]):
            with redirect_stdout(io.StringIO()):
                assess_policy.main()
        return json.loads(output.read_text())

    def test_all_thresholds_are_required(self):
        for variation in (None, "early_failure", "low_return", "bad_joint"):
            with self.subTest(variation=variation), tempfile.TemporaryDirectory() as directory:
                result = self.assess(directory, variation)
                self.assertEqual(result["passed_four_training_clip_gate"], variation is None)

    def test_changed_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "trace changed"):
                self.assess(directory, "tampered_trace")

    @unittest.skipUnless(Path("/proc/self/stat").exists(), "Linux process identity control")
    def test_process_identity_changes_are_rejected(self):
        launch = {"pid": os.getpid(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  "command": Path("/proc/self/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}
        snapshot = process_snapshot(launch)
        self.assertEqual(process_snapshot(launch, snapshot["start_ticks"])["pid"], os.getpid())
        with self.assertRaisesRegex(RuntimeError, "reused"):
            process_snapshot(launch, snapshot["start_ticks"] + 1)
        with self.assertRaisesRegex(RuntimeError, "command"):
            process_snapshot({**launch, "command": ["wrong-command"]}, snapshot["start_ticks"])
        with self.assertRaisesRegex(RuntimeError, "boot"):
            process_snapshot({**launch, "boot_id": "wrong-boot"}, snapshot["start_ticks"])


if __name__ == "__main__":
    unittest.main()
