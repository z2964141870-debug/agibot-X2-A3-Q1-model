"""Read-only sampling must distinguish exit, PID reuse and observation bounds."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memory_watch import GuardMemoryRecorder, memory, process, sample, watch


class MemoryWatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc, self.cgroup = self.root / "proc", self.root / "cgroup"
        self.proc.mkdir()
        self.cgroup.mkdir()
        (self.cgroup / "memory.usage_in_bytes").write_text("10000\n")
        (self.cgroup / "memory.stat").write_text(
            "total_cache 6000\ntotal_rss 3000\ntotal_shmem 0\ntotal_dirty 40\ntotal_writeback 10\n")
        (self.cgroup / "memory.oom_control").write_text("under_oom 0\noom_kill 0\n")
        self.make_process(12, 1, "100")
        self.make_process(13, 12, "101")
        self.make_process(99, 1, "102")

    def make_process(self, pid, parent, start):
        path = self.proc / str(pid)
        path.mkdir(exist_ok=True)
        values = ["S", str(parent)] + ["0"] * 17 + [start]
        (path / "stat").write_text(f"{pid} (worker (name)) " + " ".join(values))
        (path / "status").write_text("VmRSS: 100 kB\nRssAnon: 60 kB\nRssFile: 40 kB\n")

    def test_exact_handle_and_child_tree_without_foreign_processes(self):
        row = sample(self.proc, self.cgroup, 12, "100")
        self.assertEqual(row["status"], "OBSERVED_LIVE")
        self.assertEqual([p["pid"] for p in row["processes"]], [12, 13])
        self.assertEqual(row["processes"][0]["name"], "worker (name)")
        self.assertEqual(row["processes"][0]["VmRSS_bytes"], 102400)
        self.assertEqual(row["memory"]["total_bytes"], 10000)
        self.assertEqual(memory(self.cgroup)["total_cache_bytes"], 6000)

    def test_missing_and_reused_handles_cannot_begin_observing(self):
        self.assertIsNone(process(self.proc, 22))
        self.assertEqual(sample(self.proc, self.cgroup, 22, "100")["status"], "OBSERVED_PROCESS_EXIT")
        self.assertEqual(sample(self.proc, self.cgroup, 12, "999")["status"], "OBSERVED_PID_REUSE")
        output = self.root / "missing.jsonl"
        with self.assertRaisesRegex(RuntimeError, "verified live"):
            watch(self.proc, self.cgroup, 12, "999", output, 1, 1)
        self.assertFalse(output.exists())

    def test_live_then_exit_records_measurements_without_claiming_training_complete(self):
        first = sample(self.proc, self.cgroup, 12, "100")
        output = self.root / "exit.jsonl"
        with patch("memory_watch.sample", side_effect=[first, dict(status="OBSERVED_PROCESS_EXIT")]), \
             patch("memory_watch.time.sleep"):
            summary = watch(self.proc, self.cgroup, 12, "100", output, 1, 1)
        self.assertEqual(summary["status"], "OBSERVED_PROCESS_EXIT")
        self.assertEqual(summary["samples"], 1)
        self.assertFalse(summary["training_status_inferred"])
        self.assertEqual(summary["signals_sent"], 0)
        self.assertEqual(summary["maximum_anonymous_rss_bytes"], 3000)
        self.assertEqual(json.loads(output.read_text())["memory"]["total_cache_bytes"], 6000)
        with self.assertRaises(FileExistsError):
            watch(self.proc, self.cgroup, 12, "100", output, 1, 1)

    def test_reuse_during_process_tree_scan_discards_the_sample(self):
        initial = process(self.proc, 12)
        reused = dict(initial, start_ticks="999")
        with patch("memory_watch.subtree", return_value=[initial]), \
             patch("memory_watch.process", side_effect=[initial, reused]):
            self.assertEqual(sample(self.proc, self.cgroup, 12, "100")["status"], "OBSERVED_PID_REUSE")

    def test_observation_timeout_does_not_declare_process_terminal(self):
        output = self.root / "timeout.jsonl"
        with patch("memory_watch.time.monotonic", side_effect=[0, 2, 2]):
            summary = watch(self.proc, self.cgroup, 12, "100", output, 1, 1)
        self.assertEqual(summary["status"], "OBSERVATION_TIME_BOUND_REACHED")
        self.assertFalse(summary["training_status_inferred"])
        self.assertEqual(process(self.proc, 12)["start_ticks"], "100")

    def test_invalid_bounds_create_no_files(self):
        for seconds, interval in ((0, 1), (10801, 1), (1, .1), (1, 11)):
            with self.subTest(seconds=seconds, interval=interval), self.assertRaises(ValueError):
                watch(self.proc, self.cgroup, 12, "100", self.root / "invalid.jsonl", seconds, interval)
        self.assertFalse((self.root / "invalid.jsonl").exists())

    def test_guard_peak_breakdown_keeps_values_from_the_same_sample(self):
        output = self.root / "guard.jsonl"
        recorder = GuardMemoryRecorder(self.proc, self.cgroup, 12, output)
        recorder.record()
        (self.cgroup / "memory.usage_in_bytes").write_text("20000\n")
        (self.cgroup / "memory.stat").write_text(
            "total_cache 2000\ntotal_rss 17000\ntotal_shmem 0\ntotal_dirty 0\ntotal_writeback 0\n")
        recorder.record()
        recorder.close()
        summary = json.loads(output.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["maximum_cache_bytes"], 6000)
        self.assertEqual(summary["sample_at_maximum_total"]["memory"]["total_cache_bytes"], 2000)
        self.assertEqual(summary["sample_at_maximum_total"]["memory"]["total_rss_bytes"], 17000)
        self.assertEqual(summary["maximum_total_bytes"], 20000)
        self.assertEqual(summary["observed_samples"], 2)
        self.assertEqual(summary["status"], "OBSERVATION_ENDED")
        self.assertFalse(summary["training_status_inferred"])
        with self.assertRaisesRegex(RuntimeError, "has ended"):
            recorder.record()
        with self.assertRaises(FileExistsError):
            GuardMemoryRecorder(self.proc, self.cgroup, 12, output)

    def test_guard_history_limit_keeps_sampling_peak_without_growing_log(self):
        output = self.root / "bounded.jsonl"
        recorder = GuardMemoryRecorder(self.proc, self.cgroup, 12, output, maximum_bytes=1)
        recorder.record()
        (self.cgroup / "memory.usage_in_bytes").write_text("30000\n")
        recorder.record()
        recorder.close()
        summary = json.loads(output.with_suffix(".summary.json").read_text())
        self.assertEqual(output.stat().st_size, 0)
        self.assertEqual(summary["samples"], 0)
        self.assertEqual(summary["observed_samples"], 2)
        self.assertTrue(summary["history_size_bound_reached"])
        self.assertEqual(summary["sample_at_maximum_total"]["memory"]["total_bytes"], 30000)

    def test_guard_reused_handle_is_rejected_and_old_summary_preserved(self):
        output = self.root / "reused.jsonl"
        recorder = GuardMemoryRecorder(self.proc, self.cgroup, 12, output)
        recorder.record()
        before = output.read_bytes()
        self.make_process(12, 1, "999")
        with self.assertRaisesRegex(RuntimeError, "exact live handle"):
            recorder.record()
        self.assertEqual(output.read_bytes(), before)
        recorder.close()
        for limit in (0, True, 1.5, 16 * 1024**2 + 1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                GuardMemoryRecorder(self.proc, self.cgroup, 12, self.root / "invalid.jsonl", limit)


if __name__ == "__main__":
    unittest.main()
