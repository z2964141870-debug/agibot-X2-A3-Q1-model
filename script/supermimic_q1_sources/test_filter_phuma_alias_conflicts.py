import copy
import unittest

from filter_phuma_alias_conflicts import exclusions


class AliasExclusionTests(unittest.TestCase):
    def fixture(self):
        splits = {"train": ["a", "b", "c", "d"], "development": ["dev"], "test": ["test"]}
        keys = {"a": ("CMU/1/walk", "CMU/1"), "b": ("CMU/2/other", "CMU/2"),
                "c": ("CMU/3/c", "CMU/3"), "d": ("CMU/4/d", "CMU/4"),
                "dev": ("CMU/1/walk", "CMU/1"), "test": ("CMU/2/test", "CMU/2")}
        motions, rows = [], []
        for split, names in splits.items():
            for name in names:
                source = "humanml/" + name
                motions.append({"clip": name, "phuma_source": {"source_group": source}})
                rows.append({"clip": name, "split": split, "source_group": source,
                             "original_motion_key": keys[name][0], "directory_subject_candidate": keys[name][1]})
        return motions, sum(splits.values(), []), splits, {"rows": rows}

    def test_excludes_exact_event_and_subject_candidates_from_train_only(self):
        inputs = self.fixture()
        original = copy.deepcopy(inputs)
        removed = exclusions(*inputs)
        self.assertEqual(set(removed), {"a", "b"})
        self.assertEqual(set(removed["a"]), {"original_motion_key", "directory_subject_candidate"})
        self.assertEqual(removed["b"], {"directory_subject_candidate": "CMU/2"})
        self.assertEqual(inputs, original)

    def test_missing_alias_or_wrong_split_or_origin_is_rejected(self):
        for mutation in ("missing", "split", "origin", "duplicate"):
            motions, qualified, splits, aliases = self.fixture()
            if mutation == "missing":
                aliases["rows"].pop()
            elif mutation == "split":
                aliases["rows"][0]["split"] = "test"
            elif mutation == "origin":
                aliases["rows"][0]["source_group"] = "humanml/wrong"
            else:
                aliases["rows"].append(copy.deepcopy(aliases["rows"][0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                exclusions(motions, qualified, splits, aliases)

    def test_unresolved_candidates_do_not_become_certified_or_drop_unrelated_clips(self):
        motions, qualified, splits, aliases = self.fixture()
        for row in aliases["rows"]:
            row["original_motion_key"] = None
            row["directory_subject_candidate"] = None
        self.assertEqual(exclusions(motions, qualified, splits, aliases), {})


if __name__ == "__main__":
    unittest.main()
