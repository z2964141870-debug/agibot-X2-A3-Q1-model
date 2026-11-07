"""Recovery must preserve completed physics, its original source and total budget."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from student_review_resume import (ResumeBudget, child_command, copy_first, execute, phases_for,
                                   review_cases, spent_seconds)
from common import identity, save
from delivery import read


class ResumeTests(unittest.TestCase):
    def test_actual_media_branch_sets_the_video_module_path_before_rendering(self):
        args = SimpleNamespace(child="media", config=Path("config"), runtime_config=Path("runtime"),
            output=Path("out"), event_kind="nominal", variant="teacher", horizon=10, clip="clip")
        calls = []
        runtime = dict(primebot_python="/primebot/python")
        with patch("student_review_resume.read", side_effect=[{}, runtime]), \
                patch("student_review_resume.runtime_paths", side_effect=lambda r:calls.append(("paths", r))), \
                patch("student_review_resume.media", side_effect=lambda *a:calls.append(("media", a[2]))):
            execute(args)
        self.assertEqual(calls, [("paths", runtime), ("media", runtime)])

    def test_media_and_physics_use_the_locked_primebot_runtime_without_a_config_python_key(self):
        args = SimpleNamespace(config=Path("config"), runtime_config=Path("runtime"), output=Path("output"),
            training_root=Path("trained"), evaluation_commit="a"*40)
        runtime = {"primebot_python":"/primebot/python", "morph_python":"/morph/python"}
        owner = dict(pid=123, start_ticks="42")
        for phase in ("replay", "media"):
            command = child_command(args, runtime, owner, review_cases()[0], phase)
            self.assertEqual(command[0], runtime["primebot_python"])
            self.assertEqual(command[command.index("--child")+1], phase)

    def test_exactly_one_replay_is_reused_and_all_seven_media_phases_remain(self):
        phases = [(i, phase) for i, case in enumerate(review_cases()) for phase in phases_for(case)
                  if not (i == 0 and phase == "replay")]
        self.assertEqual(sum(p == "replay" for _, p in phases), 22)
        self.assertEqual(sum(p == "media" for _, p in phases), 7)

    def test_unrelated_failure_or_exhausted_budget_cannot_use_this_recovery(self):
        value = dict(status="FAILED", stage="R1_MATCHED_CLOSED_LOOP", error="KeyError: 'python'",
            started_utc="2026-11-07T03:22:50+00:00", ended_utc="2026-11-07T03:24:39+00:00")
        self.assertEqual(spent_seconds(value), 109)
        for bad in (dict(value, status="RUNNING"), dict(value, error="RuntimeError: worker failed"),
                    dict(value, ended_utc="2026-11-07T04:24:39+00:00")):
            with self.assertRaises(RuntimeError):
                spent_seconds(bad)

    def test_copy_rewrites_only_new_trace_and_keeps_original_artifacts_identical(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/"old/first", root/"new"
            directory = source/"review/clip_s0"
            directory.mkdir(parents=True)
            trace = directory/"rollout.npz"
            trace.write_bytes(b"preserved test bytes")
            timing = source/"stream_timing.jsonl"
            timing.write_text("{}\n")
            row = dict(trace=str(trace), complete=True)
            save(directory/"result.json", row)
            save(source/"review/result.json", dict(status="PHYSICS_PASSED", cases=[row],
                stream_telemetry=identity(timing)))
            originals = [identity(p) for p in source.rglob("*") if p.is_file()]
            copy_first(source, output)
            self.assertTrue(all(identity(r["path"]) == r for r in originals))
            result = read(output/"first/review/result.json")
            self.assertEqual(Path(result["cases"][0]["trace"]), output/"first/review/clip_s0/rollout.npz")
            self.assertEqual(result["stream_telemetry"], identity(output/"first/stream_timing.jsonl"))

    def test_continuation_cannot_reset_the_cumulative_review_budget(self):
        budget = object.__new__(ResumeBudget)
        budget.spent, budget.started = 109., 100.
        with patch("student_review_resume.time.monotonic", return_value=2692.):
            with self.assertRaisesRegex(RuntimeError, "Cumulative"):
                budget.check()


if __name__ == "__main__":
    unittest.main()
