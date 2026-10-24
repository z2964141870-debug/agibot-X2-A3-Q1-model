"""Controls for held-out selection, early failure and saved window boundaries."""

import copy
import unittest

from diagnose_phuma_models import diagnostic_rows, full_denominator_terms, window_starts


class PhumaDiagnosticTests(unittest.TestCase):
    def fixture(self):
        splits = {"train": ["walk"], "development": ["dev"], "test": ["unseen"]}
        rows = [{"clip": name, "split": "training" if split == "train" else "heldout", "reference": {"sha256": name},
                 "expected_steps": 20} for split, names in splits.items() for name in names]
        return {"rows": rows}, {"rows": copy.deepcopy(rows)}, splits

    def test_only_registered_train_development_returned(self):
        rows = diagnostic_rows(*self.fixture())
        self.assertEqual([[row["clip"] for row in phase] for phase in rows],
                         [["walk", "dev"], ["walk", "dev"]])
        self.assertEqual([row["split"] for row in rows[0]], ["train", "development"])
        self.assertEqual([row["evaluation_split"] for row in rows[0]], ["training", "heldout"])

    def test_changed_pair_or_split_rejected(self):
        for field, value in (("reference", {"sha256": "changed"}),
                             ("expected_steps", 19), ("split", "test")):
            initial, final, splits = self.fixture()
            final["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic_rows(initial, final, splits)
        initial, final, splits = self.fixture()
        splits["test"].append("walk")
        with self.assertRaises(ValueError):
            diagnostic_rows(initial, final, splits)

    def test_early_failure_terms_use_complete_denominator(self):
        row = {"executed_steps": 2, "expected_steps": 8, "normalized_return": .15,
               "mean_step_rewards": {"root": 2., "body": 5., "penalty": -1., "total": 6.}}
        terms = full_denominator_terms(row, 10.)
        self.assertAlmostEqual(terms["root"], .05)
        self.assertAlmostEqual(terms["body"], .125)
        self.assertAlmostEqual(terms["penalty"], -.025)
        self.assertAlmostEqual(sum(terms.values()), .15)
        row["normalized_return"] = .6
        with self.assertRaises(ValueError):
            full_denominator_terms(row, 10.)

    def test_no_windows_extend_past_terminal_state(self):
        self.assertEqual(window_starts(38, 32), [0])
        self.assertEqual(window_starts(40, 32), [0, 8])
        self.assertEqual(window_starts(7, 8), [])
        self.assertEqual(window_starts(8, 8), [0])
        for values in ((38, 0), (38, 8, 0), (-1, 8), (True, 8)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                window_starts(*values)


if __name__ == "__main__":
    unittest.main()
