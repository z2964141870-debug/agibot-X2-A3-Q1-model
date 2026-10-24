"""Controls for stationary-root failures and prefix-only early termination metrics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare import compare, identity


class ComparisonTests(unittest.TestCase):
    def fixture(self, root, early_failure=False):
        motions = root / "motions"
        motions.mkdir()
        rows = [{"clip": name, "split": split, "bucket": "locomotion",
                 "phuma_source": {"source_group": name}}
                for name, split in (("train", "train"), ("dev", "development"), ("test", "test"))]
        (motions / "manifest.json").write_text(json.dumps({"motions": rows}))
        (root / "spec.json").write_text(json.dumps({"motions": str(motions),
            "evaluation_clips": [row["clip"] for row in rows]}))
        for label in ("initial", "final"):
            destination = root / ("evaluation_" + label)
            destination.mkdir()
            evaluated = []
            for row in rows:
                reference = motions / (row["clip"] + ".npz")
                qpos = np.zeros((2, 29))
                qpos[:, 3] = 1
                qpos[1, 0] = 1
                if not reference.exists():
                    np.savez(reference, qpos=qpos)
                count = 1 if early_failure and label == "final" else 2
                state = np.zeros((count, 57))
                state[:, 3] = 1
                if label == "initial":
                    state[:, :29] = qpos[:count]
                trace = destination / (row["clip"] + ".npz")
                np.savez(trace, state=state)
                evaluated.append({"clip": row["clip"], "executed_steps": count, "expected_steps": 2,
                    "failed": count < 2, "normalized_return": .5, "full_clip_completion": count / 2,
                    "executed_joint_rmse_rad": 0., "reference": identity(reference), "trace": identity(trace)})
            receipt = destination / "receipt.json"
            receipt.write_text(json.dumps({"rows": evaluated}))
            (root / ("evaluation_audit_" + label + ".json")).write_text(json.dumps({
                "status": "EVALUATION_TRACE_AUDIT_PASSED", "receipt": identity(receipt)}))

    def test_perfect_joints_do_not_hide_stationary_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            compare(root)
            report = json.loads((root / "comparison.json").read_text())
            final = report["pairs"][0]["final"]
            self.assertEqual(final["executed_joint_rmse_rad"], 0.)
            self.assertEqual(final["policy_max_horizontal_displacement_m"], 0.)
            self.assertGreater(final["root_position_rmse_m"], .7)
            self.assertFalse(final["meets_registered_tracking_gates"])
            self.assertTrue(report["pairs"][0]["initial"]["meets_registered_tracking_gates"])

    def test_short_zero_error_prefix_is_not_full_clip_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, early_failure=True)
            compare(root)
            final = json.loads((root / "comparison.json").read_text())["pairs"][0]["final"]
            self.assertEqual(final["root_position_rmse_m"], 0.)
            self.assertEqual(final["full_clip_completion"], .5)
            self.assertIsNone(final["completed_root_endpoint_error_m"])
            self.assertFalse(final["meets_registered_tracking_gates"])


if __name__ == "__main__":
    unittest.main()
