"""Continuation orchestration controls; no robot physics or optimizer updates."""

from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pipeline
from test_initialization import ParentFixture


class PipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "logs").mkdir()

    def test_release_rejects_unpublished_sibling_source(self):
        core = self.root / "script/supermimic_for_q1"
        extension = self.root / pipeline.EXTENSION
        core.mkdir(parents=True)
        extension.mkdir(parents=True)
        (core / "train.py").write_text("pass\n")
        (core / "improvement_plan_S01.json").write_text("{}\n")
        worker = extension / "train.py"
        worker.write_text("pass\n")
        for command in (["git", "init", "-q", "-b", "Q1"], ["git", "add", "script"],
                        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                         "commit", "-qm", "fixture"]):
            subprocess.run(command, cwd=self.root, check=True, capture_output=True)
        release_path = self.root / "release.json"
        pipeline.release(self.root, release_path)
        self.assertEqual(len(json.loads(release_path.read_text())["files"]), 3)
        worker.write_text("raise RuntimeError('unpublished')\n")
        rejected = self.root / "rejected.json"
        with self.assertRaisesRegex(ValueError, "Unpublished source"):
            pipeline.release(self.root, rejected)
        self.assertFalse(rejected.exists())

    def test_registered_eval_keeps_reused_development_clip_when_heldout_field_is_empty(self):
        names = ["walk", "squat", "punch"]
        parent = {"train_clips": [{"path": "walk.npz"}, {"path": "squat.npz"}],
                  "heldout_clips": [], "motion_manifest": {"path": str(self.root / "motions/manifest.json")}}
        stages = pipeline.make_stages(self.root, self.root / "extension", parent, names,
            self.root / "pair", self.root / "pair/A", 6000, 7, "0" * 40)
        evaluations = [stage for stage in stages if stage["name"] in ("eval_initial", "eval_final")]
        self.assertEqual(len(evaluations), 2)
        for stage in evaluations:
            self.assertEqual(stage["argv"][stage["argv"].index("--clips") + 1:], names)

    def test_unfinished_pair_is_rejected_before_output_is_created(self):
        fixture = ParentFixture(self.root)
        fixture.finish()
        fixture.write(fixture.pair / "receipt.json", {"status": "RUNNING"})
        output = self.root / "data/supermimic_for_q1/continuation"
        with self.assertRaisesRegex(ValueError, "not completed"):
            pipeline.register(self.root, output, self.root / "release.json", fixture.pair,
                              fixture.run, 6000, 7, "fixture")
        self.assertFalse(output.exists())

    def plan(self):
        output = self.root / "continuation"
        output.mkdir()
        logs = self.root / "logs/continuation"
        logs.mkdir()
        plan = {"root": str(self.root), "output": str(output), "logs": str(logs),
                "baseline_health": {"oom_kill": 14}, "stages": []}
        path = output / "plan.json"
        path.write_text(json.dumps(plan) + "\n")
        return plan, path

    def test_pair_lock_prevents_concurrent_extension_without_creating_started_state(self):
        plan, path = self.plan()
        with (self.root / "logs/.q1_batch_pair.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                pipeline.execute(path)
        self.assertFalse((Path(plan["output"]) / "started.json").exists())

    def test_child_failure_is_terminal_and_cannot_repeat_the_run(self):
        plan, path = self.plan()
        output = Path(plan["output"])
        plan["stages"] = [{"name": "train", "argv": [sys.executable, "-c", "import sys; sys.exit(7)"],
            "result": str(output / "train.json"), "expected_status": "SHOULD_NOT_SUCCEED", "cpu": True},
            {"name": "evaluation", "argv": [sys.executable, "-c", "raise AssertionError('must not run')"],
             "result": str(output / "eval.json"), "expected_status": "SHOULD_NOT_SUCCEED", "cpu": True}]
        path.write_text(json.dumps(plan) + "\n")
        runner = pipeline.runner
        runner.STOP_REQUESTED = False
        with patch.object(pipeline, "verify_plan"), patch.object(runner, "verify_frozen"), \
                patch.object(runner, "health", return_value={"oom_kill": 14}), \
                patch.object(runner, "reclaim_checkpoint_cache", return_value={}), \
                patch.object(runner, "process_identity", side_effect=lambda pid: {"pid": pid}), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                pipeline.execute(path)
            receipt = (output / "receipt.json").read_bytes()
            self.assertEqual(json.loads(receipt)["status"], "FAILED")
            self.assertEqual(json.loads((output / "events.jsonl").read_text())["exit_code"], 7)
            self.assertFalse((Path(plan["logs"]) / "evaluation.log").exists())
            with self.assertRaises(FileExistsError):
                pipeline.execute(path)
            self.assertEqual((output / "receipt.json").read_bytes(), receipt)


if __name__ == "__main__":
    unittest.main()
