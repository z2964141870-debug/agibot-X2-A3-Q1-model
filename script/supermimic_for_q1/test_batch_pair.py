"""Controller failure-path checks; no physics, torch or optimizer execution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_batch_pair as runner


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "logs").mkdir()
        self.plan = {"root": str(self.root), "output": str(self.root), "logs": str(self.root / "logs"),
                     "baseline_health": {"oom_kill": 14}}
        self.state = {"status": "RUNNING"}
        self.verify = patch.object(runner, "verify_frozen").start()
        patch.object(runner, "health", return_value={"oom_kill": 14}).start()
        patch.object(runner, "reclaim_checkpoint_cache", return_value={}).start()
        patch.object(runner, "process_identity", side_effect=lambda pid: {"pid": pid}).start()
        self.addCleanup(patch.stopall)
        runner.STOP_REQUESTED = False

    def stage(self, name, script):
        result = self.root / (name + ".json")
        return {"name": name, "argv": [sys.executable, "-c", script, str(result)],
                "cpu": True, "result": str(result), "expected_status": "AUDIT_PASSED"}

    def test_native_failure_retains_exit_and_prevents_second_arm(self):
        self.plan["stages"] = [self.stage("A", "import sys; print('expected failure'); sys.exit(7)"),
                               self.stage("B", "raise AssertionError('must not run')")]
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            runner.run_pipeline(self.plan, self.state)
        event = json.loads((self.root / "events.jsonl").read_text())
        self.assertEqual(event["exit_code"], 7)
        self.assertIn("expected failure", (self.root / "logs/A.log").read_text())
        self.assertFalse((self.root / "logs/B.log").exists())

    def test_failed_usability_gate_does_not_abort_valid_audit(self):
        script = "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({'status':'AUDIT_PASSED','usability_gates':{'tracking':False}}))"
        self.plan["stages"] = [self.stage("A", script), self.stage("B", script)]
        runner.run_pipeline(self.plan, self.state)
        self.assertTrue((self.root / "B.json").exists())

    def test_unexpected_receipt_stops_pipeline_even_on_zero_exit(self):
        stage = self.stage("A", "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('{\"status\":\"FAILED\"}')")
        with self.assertRaisesRegex(RuntimeError, "Unexpected result"):
            runner.run_stage(self.plan, stage, self.state)

    def test_output_cannot_be_reused(self):
        stage = self.stage("A", "raise AssertionError('must not run')")
        (self.root / "logs/A.log").touch()
        with self.assertRaises(FileExistsError):
            runner.run_stage(self.plan, stage, self.state)

    def test_signal_prevents_next_stage(self):
        runner.STOP_REQUESTED = True
        with self.assertRaises(InterruptedError):
            runner.run_stage(self.plan, self.stage("B", "raise AssertionError('must not run')"), self.state)
        self.assertFalse((self.root / "logs/B.log").exists())


class CacheScopeTest(unittest.TestCase):
    def test_only_project_checkpoints_are_advised_and_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "data/supermimic_for_q1"
            base.mkdir(parents=True)
            checkpoint = base / "saved.pt"
            checkpoint.write_bytes(b"saved model contents")
            (base / "writing.tmp").write_bytes(b"in progress")
            external = root / "unrelated.pt"
            external.write_bytes(b"unrelated")
            (base / "external.pt").symlink_to(external)
            with patch.object(runner.os, "posix_fadvise", create=True) as advise:
                with patch.object(runner.os, "POSIX_FADV_DONTNEED", 4, create=True):
                    result = runner.reclaim_checkpoint_cache(root)
            self.assertEqual(advise.call_count, 1)
            self.assertEqual(result["files"], 1)
            self.assertEqual(checkpoint.read_bytes(), b"saved model contents")
            self.assertEqual(external.read_bytes(), b"unrelated")
            self.assertEqual(result["files_deleted"], 0)


if __name__ == "__main__":
    unittest.main()
