"""Metadata admission controls; no checkpoint tensors or physics are loaded."""

import json
from pathlib import Path
import tempfile
import unittest

from diagnose_completed_run import identity, verified_evaluation


class EvaluationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("model.pt", "reference.npz", "trace.npz"):
            (self.root / name).write_bytes(b"metadata fixture")
        self.receipt = {
            "status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED",
            "checkpoint": identity(self.root / "model.pt"),
            "rows": [{"clip": "walk", "expected_steps": 8, "executed_steps": 8,
                      "failed": False, "reference": identity(self.root / "reference.npz"),
                      "trace": identity(self.root / "trace.npz")}],
        }

    def write(self, value, name):
        path = self.root / name
        path.write_text(json.dumps(value))
        return identity(path)

    def records(self, verified=True, executed=8):
        receipt = self.write(self.receipt, "receipt.json")
        audit = self.write({
            "status": "EVALUATION_TRACE_AUDIT_PASSED", "verified": verified,
            "receipt": receipt,
            "clips": [{"clip": "walk", "passed": True,
                       "recomputed": {"expected_steps": 8, "executed_steps": executed,
                                      "failed": False}}],
        }, "audit.json")
        return receipt, audit

    def test_matching_audit_accepts_metadata(self):
        self.assertEqual(verified_evaluation(*self.records()), self.receipt)

    def test_unverified_or_wrong_counts_rejected(self):
        with self.assertRaises(ValueError):
            verified_evaluation(*self.records(verified=False))
        with self.assertRaises(ValueError):
            verified_evaluation(*self.records(executed=7))

    def test_changed_saved_trace_rejected(self):
        records = self.records()
        (self.root / "trace.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verified_evaluation(*records)


if __name__ == "__main__":
    unittest.main()
