"""Keep stopped-run reports separate from formal completion and missing cases."""

import json
from pathlib import Path
import tempfile
import unittest

from event_partial_summary import CLIPS, WALL_ERROR, groups, stopped_gate
from event_summary import analyze


def stopped():
    result = dict(status="FAILED", error=WALL_ERROR, controller=dict(pid=100),
        started_utc="2026-11-06T14:24:05+00:00", ended_utc="2026-11-06T16:24:05+00:00")
    guard = dict(status="FAILED", error=WALL_ERROR, worker_pid=101,
                 cleanup=dict(worker_reaped=True, worker_return_code=-15))
    return result, guard


class PartialEventTests(unittest.TestCase):
    def test_only_clean_whole_time_stop_is_eligible(self):
        result, guard = stopped()
        stopped_gate(result, guard, lambda pid: False)
        with self.assertRaisesRegex(ValueError, "still present"):
            stopped_gate(result, guard, lambda pid: pid == 101)
        guard["cleanup"]["worker_reaped"] = False
        with self.assertRaisesRegex(ValueError, "cleanup"):
            stopped_gate(result, guard, lambda pid: False)

    def test_live_other_failure_and_phase_only_stop_are_rejected(self):
        result, guard = stopped()
        result["status"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "wall-time failure"):
            stopped_gate(result, guard, lambda pid: False)
        result["status"], result["error"] = "FAILED", "CUDA error"
        with self.assertRaisesRegex(ValueError, "wall-time failure"):
            stopped_gate(result, guard, lambda pid: False)
        result["error"], result["ended_utc"] = WALL_ERROR, "2026-11-06T14:54:05+00:00"
        with self.assertRaisesRegex(ValueError, "whole-campaign"):
            stopped_gate(result, guard, lambda pid: False)

    def test_missing_or_failed_clip_has_no_full_group_mean(self):
        rows = [dict(event="stop", method="masked", horizon_frames=10, clip=c,
                     complete=True) for c in CLIPS[:2]]
        value = groups(rows)[0]
        self.assertEqual(value["available_cases"], 2)
        self.assertIsNone(value["full_fixed_set_means"])
        rows.append(dict(event="stop", method="masked", horizon_frames=10,
                         clip=CLIPS[2], complete=False))
        self.assertIsNone(groups(rows)[0]["full_fixed_set_means"])

    def test_original_full_analysis_still_rejects_preserved_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve()/"source"
            source.mkdir()
            (source/"result.json").write_text(json.dumps(stopped()[0]))
            output = source.parent/"output"
            with self.assertRaisesRegex(ValueError, "not formally complete"):
                analyze(source, output)
            self.assertFalse(output.exists())

    def test_duplicate_clip_cannot_be_counted_as_complete_coverage(self):
        row = dict(event="stop", method="masked", horizon_frames=10,
                   clip=CLIPS[0], complete=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            groups([row, row.copy()])


if __name__ == "__main__":
    unittest.main()
