"""Source admission stays bounded for both original and shared source trees."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from guard import Budget
from runtime_guard import StorageLimit


class SourceBudgetTests(unittest.TestCase):
    def budget(self, **config):
        return Budget(dict(adapted_source="/source", **config), Path("/experiment"), 60, 8 * 1024**3)

    def check_storage(self, budget, source_bytes=142503936):
        def tree(path):
            return dict(root=str(path), unique_allocated_bytes=(source_bytes if path == Path("/source") else 1024))

        with patch("runtime_guard.allocated_tree", side_effect=tree), \
             patch("runtime_guard.shutil.disk_usage", return_value=SimpleNamespace(free=500 * 1024**3)):
            return budget.inner.storage.check()

    def test_original_default_rejects_the_measured_larger_source(self):
        budget = self.budget()
        self.assertEqual(budget.inner.storage.maximum_source_bytes, 64 * 1024**2)
        with self.assertRaisesRegex(StorageLimit, "Source storage exceeds"):
            self.check_storage(budget)

    def test_explicit_shared_source_budget_admits_the_measured_tree(self):
        budget = self.budget(maximum_source_bytes=256 * 1024**2)
        self.assertEqual(self.check_storage(budget)["source"]["unique_allocated_bytes"], 142503936)
        self.assertEqual(budget.inner.maximum_bytes, 30 * 1024**3)
        self.assertEqual(budget.inner.expected_oom, 0)
        self.assertEqual(budget.inner.storage.minimum_free_bytes, 100 * 1024**3)
        self.assertEqual(budget.inner.storage.maximum_experiment_bytes, 8 * 1024**3)

    def test_shared_source_budget_still_rejects_an_oversized_tree(self):
        with self.assertRaisesRegex(StorageLimit, "Source storage exceeds"):
            self.check_storage(self.budget(maximum_source_bytes=256 * 1024**2), 256 * 1024**2 + 1)

    def test_invalid_or_unbounded_source_budgets_are_rejected(self):
        for value in (0, -1, True, "268435456", 268435456.0, 256 * 1024**2 + 1):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.budget(maximum_source_bytes=value)


if __name__ == "__main__":
    unittest.main()
