import copy
import unittest

from inspect_phuma_reference_dynamics import select_rows


class SelectionTests(unittest.TestCase):
    def fixture(self):
        splits = {"train": [f"train_{i}" for i in range(12)], "development": ["dev"], "test": ["test"]}
        motions, checks = [], []
        for split, names in splits.items():
            for name in names:
                reference = {"path": "/reference/" + name, "bytes": 1, "sha256": name}
                motions.append({"clip": name, "split": split, "bucket": "locomotion",
                                "qualified_for_training_pilot": True,
                                "phuma_source": {"source_group": name}, "output": reference})
                checks.append({"clip": name, "split": split, "bucket": "locomotion",
                               "source_group": name, "qualified_for_training_pilot": True,
                               "reference": reference, "gates": {str(i): True for i in range(19)}})
        return {"motions": motions}, {"clips": checks, "qualified_clips": sum(splits.values(), [])}, splits

    def test_fixed_selection_ignores_order_and_never_uses_heldout(self):
        manifest, checked, splits = self.fixture()
        original = copy.deepcopy((manifest, checked, splits))
        selected = [row["clip"] for row in select_rows(manifest, checked, splits)]
        self.assertEqual(len(selected), 8)
        self.assertLessEqual(set(selected), set(splits["train"]))
        self.assertEqual((manifest, checked, splits), original)
        manifest["motions"].reverse()
        checked["clips"].reverse()
        self.assertEqual(selected, [row["clip"] for row in select_rows(manifest, checked, splits)])

    def test_bad_admission_split_or_origin_is_rejected(self):
        for mutation in ("gate", "split", "reference", "origin", "duplicate"):
            manifest, checked, splits = self.fixture()
            if mutation == "gate":
                checked["clips"][0]["gates"]["0"] = False
            elif mutation == "split":
                checked["clips"][0]["split"] = "test"
            elif mutation == "reference":
                checked["clips"][0]["reference"] = {"path": "/wrong"}
            elif mutation == "origin":
                manifest["motions"][0]["phuma_source"]["source_group"] = "wrong"
            else:
                checked["clips"].append(copy.deepcopy(checked["clips"][0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select_rows(manifest, checked, splits)

    def test_disjoint_splits_and_enough_train_candidates_required(self):
        manifest, checked, splits = self.fixture()
        splits["test"].append(splits["train"][0])
        with self.assertRaises(ValueError):
            select_rows(manifest, checked, splits)
        manifest, checked, splits = self.fixture()
        with self.assertRaises(ValueError):
            select_rows(manifest, checked, splits, count=16)


if __name__ == "__main__":
    unittest.main()
