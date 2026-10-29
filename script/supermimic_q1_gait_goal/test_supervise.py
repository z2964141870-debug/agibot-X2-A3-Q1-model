"""Budget boundaries must hold while evaluation pauses the training clock."""

import unittest
from pathlib import Path
import tempfile

from supervise import active_training_seconds, memory_admission


class TrainingCapTests(unittest.TestCase):
    def clock(self, **values):
        return dict(absolute_maximum_seconds=1200, current_cap_seconds=1200,
                    inherited_training_seconds=0, accumulated_training_seconds=300,
                    active_started_monotonic=None, **values)

    def test_paused_evaluation_excluded(self):
        self.assertEqual(active_training_seconds(self.clock(), 5000), 300)

    def test_active_interval_counted(self):
        clock = self.clock()
        clock["active_started_monotonic"] = 100
        self.assertEqual(active_training_seconds(clock, 150), 350)

    def test_extended_inherited_or_invalid_clocks_rejected(self):
        for key, value in (("absolute_maximum_seconds", 1201), ("current_cap_seconds", 2400),
                           ("inherited_training_seconds", 1), ("accumulated_training_seconds", -1),
                           ("accumulated_training_seconds", float("nan")),
                           ("active_started_monotonic", 201), ("active_started_monotonic", float("inf"))):
            clock = self.clock()
            clock[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                active_training_seconds(clock, 200)

    def test_memory_admission_keeps_cache_in_original_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.limit_in_bytes").write_text(str(32 * 1024**3))
            (root / "memory.stat").write_text("total_cache 27909169152\ntotal_rss 13176832\ntotal_shmem 0\n")
            for gib, expected in ((27, False), (24, True)):
                (root / "memory.usage_in_bytes").write_text(str(gib * 1024**3))
                result = memory_admission(root)
                self.assertEqual(result["admitted"], expected)
                self.assertEqual(result["guard_maximum_bytes"], 30 * 1024**3)
                self.assertFalse(result["automatic_cache_eviction"])
                self.assertFalse(result["guard_threshold_changed"])


if __name__ == "__main__":
    unittest.main()
