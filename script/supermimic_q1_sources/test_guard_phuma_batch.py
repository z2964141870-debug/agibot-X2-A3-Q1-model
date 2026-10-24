import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from guard_phuma_batch import MemoryBoundBudget, MemoryLimit
from runtime_guard import guarded_run


class SharedMemoryTests(unittest.TestCase):
    def make_budget(self, directory, memory=50, oom=14):
        root = Path(directory)
        (root / "memory.usage_in_bytes").write_text(str(memory))
        (root / "memory.oom_control").write_text(f"under_oom 0\noom_kill {oom}\n")
        storage = Mock()
        storage.check.return_value = {"storage_checked": True}
        return MemoryBoundBudget(storage, 100, 14, root)

    def test_memory_limit_prevents_worker_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budget = self.make_budget(directory, memory=100)
            marker = root / "started"
            with self.assertRaises(MemoryLimit):
                guarded_run([sys.executable, "-c", "from pathlib import Path; Path('started').touch()"],
                            root, budget, root / "status.json", root / "stdout.log")
            self.assertFalse(marker.exists())
            status = json.loads((root / "status.json").read_text())
            self.assertFalse(status["cleanup"]["worker_started"])
            budget.storage.check.assert_not_called()

    def test_oom_counter_and_post_storage_memory_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = self.make_budget(directory, oom=15)
            with self.assertRaises(MemoryLimit):
                budget.check()
            budget = self.make_budget(directory)
            budget.storage.check.side_effect = lambda: (
                (Path(directory) / "memory.usage_in_bytes").write_text("100"), {})[1]
            with self.assertRaises(MemoryLimit):
                budget.check()

    def test_success_keeps_storage_and_memory_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = self.make_budget(directory)
            result = budget.check()
            self.assertTrue(result["storage_checked"])
            self.assertEqual(result["shared_memory"], {"shared_bytes": 50, "oom_kill": 14, "maximum_bytes": 100})


if __name__ == "__main__":
    unittest.main()
