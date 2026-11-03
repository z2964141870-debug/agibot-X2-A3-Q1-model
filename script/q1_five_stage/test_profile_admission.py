"""Profile admission and observation must run before the protected worker."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from guard import Budget, check_initial_cache, run
from guard_phuma_batch import MemoryBoundBudget, MemoryLimit
from memory_watch import GuardMemoryRecorder


class ProfileAdmissionTests(unittest.TestCase):
    def test_exact_cache_boundary_and_invalid_limits(self):
        maximum = 4 * 1024**3
        self.assertEqual(check_initial_cache(maximum, maximum)["cache_bytes"], maximum)
        with self.assertRaisesRegex(RuntimeError, "Initial cache exceeds"):
            check_initial_cache(maximum + 1, maximum)
        for limit in (0, True, 1.5, maximum + 1):
            with self.subTest(limit=limit), self.assertRaisesRegex(RuntimeError, "Profile cache limit"):
                check_initial_cache(0, limit)
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "measurement"):
                check_initial_cache(value, maximum)

    def test_guard_rechecks_after_inputs_before_any_worker_can_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = root / "data/supermimic_for_q1"
            data.mkdir(parents=True)
            source = root / "source"
            source.mkdir()
            output = data / "profile"
            read_text = Path.read_text

            def read(path, *args, **kwargs):
                if str(path) == "/sys/fs/cgroup/memory/memory.usage_in_bytes":
                    return "0"
                return read_text(path, *args, **kwargs)

            with patch("guard.ROOT", root), patch("guard.HERE", source), patch("guard.signal.signal"), \
                 patch("guard.subprocess.check_output", return_value=""), \
                 patch.object(Path, "read_text", read), patch.dict(os.environ), \
                 patch("guard.profile_cache_admission", side_effect=RuntimeError("Initial cache exceeds")) as admit, \
                 patch("guard.guarded_run") as worker, patch("guard.Budget") as budget:
                with self.assertRaisesRegex(RuntimeError, "Initial cache exceeds"):
                    run(dict(adapted_source=str(source)), output, ["must-not-start"], 60, 1024,
                        maximum_initial_cache_bytes=4 * 1024**3)
                admit.assert_called_once_with(4 * 1024**3)
                worker.assert_not_called()
                budget.assert_not_called()
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertFalse(status["automatic_retry"])

    def test_memory_sample_is_written_before_original_guard_rejects(self):
        with tempfile.TemporaryDirectory() as directory, patch("guard.StorageBudget"), \
             patch("guard.MemoryBoundBudget") as inner, \
             patch("memory_watch.GuardMemoryRecorder") as recorder:
            budget = Budget(dict(adapted_source=directory, record_memory_breakdown=True), Path(directory), 60, 1024)
            events = []
            recorder.return_value.record.side_effect = lambda: events.append("sample_written")

            def reject():
                events.append("original_memory_guard")
                raise RuntimeError("memory reserve reached")

            inner.return_value.check.side_effect = reject
            with self.assertRaisesRegex(RuntimeError, "memory reserve reached"):
                budget.check()
            self.assertEqual(events, ["sample_written", "original_memory_guard"])
            self.assertEqual(inner.call_args.args[1], 30 * 1024**3)

    def test_legacy_budget_does_not_create_an_observer(self):
        with tempfile.TemporaryDirectory() as directory, patch("guard.StorageBudget"), \
             patch("guard.MemoryBoundBudget"), patch("memory_watch.GuardMemoryRecorder") as recorder:
            budget = Budget(dict(adapted_source=directory), Path(directory), 60, 1024)
            self.assertIsNone(budget.recorder)
            recorder.assert_not_called()

    def test_real_memory_guard_failure_keeps_its_sample_and_observer_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, cgroup = root / "proc", root / "cgroup"
            handle = proc / str(os.getpid())
            handle.mkdir(parents=True)
            cgroup.mkdir()
            (handle / "stat").write_text(f"{os.getpid()} (guard) " + " ".join(
                ["S", "1"] + ["0"] * 17 + ["100"]))
            (handle / "status").write_text("VmRSS: 100 kB\nRssAnon: 60 kB\nRssFile: 40 kB\n")
            (cgroup / "memory.usage_in_bytes").write_text(str(30 * 1024**3))
            (cgroup / "memory.stat").write_text(
                "total_cache 8000\ntotal_rss 9000\ntotal_shmem 0\ntotal_dirty 0\ntotal_writeback 0\n")
            (cgroup / "memory.oom_control").write_text("under_oom 0\noom_kill 0\n")
            recorder = GuardMemoryRecorder(proc, cgroup, os.getpid(), root / "memory.jsonl")
            with patch("guard.StorageBudget"), patch("guard.MemoryBoundBudget", side_effect=lambda *args:
                    MemoryBoundBudget(*args, cgroup=cgroup)), \
                 patch("memory_watch.GuardMemoryRecorder", return_value=recorder):
                budget = Budget(dict(adapted_source=directory, record_memory_breakdown=True), root, 60, 1024)
                with self.assertRaisesRegex(MemoryLimit, "shared_bytes=32212254720"):
                    budget.check()
                budget.recorder.close()
            summary = json.loads((root / "memory.summary.json").read_text())
            self.assertEqual(summary["status"], "OBSERVATION_ENDED")
            self.assertEqual(summary["sample_at_maximum_total"]["memory"]["total_bytes"], 30 * 1024**3)
            self.assertEqual(summary["sample_at_maximum_total"]["memory"]["total_rss_bytes"], 9000)
            self.assertEqual(summary["sample_at_maximum_total"]["memory"]["oom_kill"], 0)


if __name__ == "__main__":
    unittest.main()
