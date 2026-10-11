"""Check observation classification without touching the host or starting jobs."""

import unittest
from pathlib import Path
import tempfile

from script.a3.idle_reboot_watch import compute_processes, read_rows, summarize


class IdleWatchTests(unittest.TestCase):
    def test_truncated_restart_record_keeps_valid_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text('{"boot_id":"a"}\n{"boot_id":\n{"boot_id":"b"}\n')
            rows, invalid = read_rows(path)
        self.assertEqual(rows, [{"boot_id": "a"}, {"boot_id": "b"}])
        self.assertEqual(invalid, [2])

    def test_detects_compute_but_excludes_monitor_and_shell(self):
        output = "\n".join([
            "1 python3 /usr/bin/python3 -m script.a3.controller_adapter_trial train",
            "2 python3 /usr/bin/python3 -m script.a3.sim_entry --capture input.npz",
            "3 python3 /usr/bin/python3 -m script.a3.idle_reboot_watch",
            "4 bash bash -c python3 -m script.a3.run_controller_adapter run",
        ])
        self.assertEqual([row["pid"] for row in compute_processes(output)], [1, 2])

    def test_boot_change_and_unknown_are_separate(self):
        job = dict(started_at_utc="start", expires_at_utc="end", expires_epoch=float("inf"))
        def row(boot, time, readable=True, compute=""):
            return dict(type="sample", boot_id=boot, at_utc=time, a3_compute_processes=[],
                        process_query_readable=True,
                        gpu_compute_query=dict(returncode=0 if readable else None, stdout=compute))
        result = summarize(job, [row("a", "1"), row("a", "2", False), row("b", "3", compute="8")])
        self.assertEqual(result["observed_reboot_count_lower_bound"], 1)
        self.assertEqual(result["unreadable_sample_times_utc"], ["2"])
        self.assertEqual(result["compute_seen_sample_times_utc"], ["3"])
        self.assertFalse(result["deadline_reached"])


if __name__ == "__main__":
    unittest.main()
