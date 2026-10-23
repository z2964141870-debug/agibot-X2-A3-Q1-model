from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import render_audited_trace as renderer


class AuditedRenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.core = self.root / "script/supermimic_for_q1"
        self.core.mkdir(parents=True)
        for name in ("render_trace.py", "contact_protocol.py", "q1_assets.py"):
            (self.core / name).write_text("# synthetic source\n")
        run = self.root / "data/supermimic_for_q1/run"
        run.mkdir(parents=True)
        checkpoint = run / "final.pt"
        checkpoint.write_bytes(b"synthetic checkpoint; no model loading")
        self.reference = self.root / "reference.npz"
        self.reference.write_bytes(b"synthetic reference")
        self.trace = self.root / "trace.npz"
        self.trace.write_bytes(b"synthetic trace")
        xml = self.root / "robot.xml"
        xml.write_text("synthetic XML")
        self.admission = run / "admission.json"
        self.manifest = {"assets": {"mjcf": renderer.identity(xml),
            "adapter": renderer.identity(self.core / "q1_assets.py"),
            "contact_protocol": {"version": "Q1_FOOT_PRIORITY_V2",
                "source": renderer.identity(self.core / "contact_protocol.py")}},
            "source_files": [renderer.identity(self.core / "render_trace.py")]}
        self.admission.write_text(json.dumps(self.manifest))
        self.row = {"clip": "walk", "expected_steps": 1653, "executed_steps": 1653,
                    "failed": False, "trace": renderer.identity(self.trace),
                    "reference": renderer.identity(self.reference)}
        self.receipt = self.root / "evaluation.json"
        self.evaluation = {"status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED",
                           "checkpoint": renderer.identity(checkpoint), "rows": [self.row]}
        self.audit = self.root / "audit.json"
        self.save_evaluation()

    def save_evaluation(self):
        self.receipt.write_text(json.dumps(self.evaluation))
        self.audit.write_text(json.dumps({"status": "EVALUATION_TRACE_AUDIT_PASSED",
            "verified": True, "receipt": renderer.identity(self.receipt),
            "clips": [{"clip": "walk", "passed": True, "recomputed": {
                key: self.row[key] for key in ("expected_steps", "executed_steps", "failed")}}]}))

    def prepare(self):
        return renderer.prepare(self.root, self.receipt, self.audit, "walk")

    def test_full_and_failed_prefix_frame_counts_are_preserved(self):
        result = self.prepare()
        self.assertEqual(result["rendered_frames_expected"], 827)
        self.assertGreater(result["max_seconds"], 1653 / 50)
        self.row.update(executed_steps=700, failed=True)
        self.save_evaluation()
        result = self.prepare()
        self.assertEqual(result["rendered_frames_expected"], 350)
        self.assertEqual(result["expected_steps"], 1653)
        self.assertTrue(result["failed"])
        self.row["failed"] = False
        self.save_evaluation()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_mismatched_audit_and_changed_trace_are_rejected(self):
        self.receipt.write_text(self.receipt.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.prepare()
        self.save_evaluation()
        self.trace.write_bytes(b"changed trace")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_old_protocol_and_changed_renderer_are_rejected(self):
        self.manifest["assets"]["contact_protocol"]["version"] = "OLD_PROTOCOL"
        self.admission.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self.prepare()
        self.manifest["assets"]["contact_protocol"]["version"] = "Q1_FOOT_PRIORITY_V2"
        self.admission.write_text(json.dumps(self.manifest))
        (self.core / "render_trace.py").write_text("# changed renderer\n")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_check_only_never_calls_renderer_or_creates_video_directory(self):
        output = self.root / "video"
        proof = self.root / "preflight.json"
        argv = ["render_audited_trace.py", "--root", str(self.root), "--evaluation", str(self.receipt),
                "--audit", str(self.audit), "--clip", "walk", "--output", str(output),
                "--check-only", "--preflight-output", str(proof)]
        with patch.object(renderer.sys, "argv", argv), patch.object(renderer, "render") as render, redirect_stdout(io.StringIO()):
            renderer.main()
            render.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(json.loads(proof.read_text())["new_physics_steps"], 0)
        with patch.object(renderer.sys, "argv", argv), self.assertRaises(FileExistsError):
            renderer.main()


if __name__ == "__main__":
    unittest.main()
