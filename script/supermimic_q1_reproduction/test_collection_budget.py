import unittest

from collection_budget import CollectionBudget


class CollectionBudgetTests(unittest.TestCase):
    def test_timed_budget_never_finishes_early_or_between_report_boundaries(self):
        budget = CollectionBudget(12000, 7200)
        self.assertFalse(budget.finished(1000, 7199.999))
        self.assertFalse(budget.finished(1001, 7201))
        self.assertTrue(budget.finished(1025, 7201))

    def test_fast_step_cap_cannot_masquerade_as_two_hours(self):
        with self.assertRaises(RuntimeError):
            CollectionBudget(12000, 7200).finished(12000, 4000)

    def test_original_step_budget_is_preserved(self):
        budget = CollectionBudget(12000)
        self.assertFalse(budget.finished(11999, 99999))
        self.assertTrue(budget.finished(12000, 1))

    def test_invalid_timed_budgets_are_rejected(self):
        for cap, duration in ((301, 7200), (12000, 0), (12000, float("nan"))):
            with self.subTest(cap=cap, duration=duration), self.assertRaises(ValueError):
                CollectionBudget(cap, duration)


if __name__ == "__main__":
    unittest.main()
