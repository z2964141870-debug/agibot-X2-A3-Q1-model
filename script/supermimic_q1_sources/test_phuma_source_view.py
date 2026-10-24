"""Reject provenance or geometry changes when admitting an inherited source view."""

import copy
import unittest

from phuma_source_view import ALIAS_STATUS, PARENT_STATUS, VIEW_STATUS, validate_documents


class SourceViewTests(unittest.TestCase):
    def fixture(self):
        split = {"train": ["a", "b", "c"], "development": ["dev"], "test": ["test"]}
        motions, checks, rows = [], [], []
        for group, names in split.items():
            for name in names:
                source = {"source_group": "humanml/" + name}
                motions.append({"clip": name, "phuma_source": source, "split": group,
                                "qualified_for_training_pilot": True, "output": {"path": name}})
                checks.append({"clip": name, "split": group, "qualified_for_training_pilot": True,
                               "duration_s": 2., "bucket": "locomotion", "reference": {"path": name},
                               "gates": {str(i): True for i in range(19)}})
                subject = "one" if name in ("a", "dev") else name
                rows.append({"clip": name, "split": group, "source_group": source["source_group"],
                             "directory_subject_candidate": subject, "original_motion_key": name})
        parent = {"status": PARENT_STATUS, "split_clips": split, "manifest": {"id": "parent_manifest"},
                  "reference_check": {"id": "parent_check"}, "scan_check": {"id": "scan"}}
        manifest = {"status": "original", "motions": motions, "qualified_clips": sum(split.values(), []),
                    "qualified_split_counts": {"train": 3, "development": 1, "test": 1}, "sources": []}
        checked = {"input_manifest": parent["manifest"], "clips": checks,
                   "qualified_clips": sum(split.values(), []),
                   "qualified_split_counts": {"train": 3, "development": 1, "test": 1}, "rejected_clips": []}
        old_aliases = {"status": ALIAS_STATUS, "inputs": [parent["manifest"], parent["reference_check"], {"id": "index"}],
                       "source": {"id": "auditor"}, "rows": rows,
                       "other_source_namespace_counts_unresolved": {}, "source_isolation_certified": False}
        provenance = {"mode": "TRAIN_ONLY_KNOWN_HELDOUT_ALIAS_EXCLUSION_VIEW",
                      "parent_manifest": parent["manifest"], "parent_reference_check": parent["reference_check"],
                      "source": {"id": "producer"}, "geometry_checks_rerun": False,
                      "source_isolation_certified": False,
                      "excluded_train_clips": {"a": {"directory_subject_candidate": "one"}}}
        completed = {"status": VIEW_STATUS, "source_exclusion": provenance, "source_isolation_certified": False,
                     "manifest": {"id": "view_manifest"}, "reference_check": {"id": "view_check"},
                     "scan_check": parent["scan_check"], "qualified_count": 4,
                     "split_clips": {"train": ["b", "c"], "development": ["dev"], "test": ["test"]},
                     "qualified_duration_s": {"train": 4., "development": 2., "test": 2.},
                     "qualified_locomotion_duration_s": {"train": 4., "development": 2., "test": 2.}}
        view_manifest, view_check = copy.deepcopy((manifest, checked))
        for records in (view_manifest["motions"], view_check["clips"]):
            records[0].update(qualified_for_training_pilot=False, qualification_before_source_exclusion=True,
                              source_exclusion_reasons={"directory_subject_candidate": "one"})
        for doc in (view_manifest, view_check):
            doc.update(qualified_clips=["b", "c", "dev", "test"],
                       qualified_split_counts={"train": 2, "development": 1, "test": 1}, source_exclusion=provenance)
        view_manifest.update(status="PHUMA_SOURCE_EXCLUSION_VIEW_WITH_INHERITED_GEOMETRY", sources=[provenance["source"]])
        view_check.update(input_manifest=completed["manifest"], rejected_clips=["a"])
        aliases = {**copy.deepcopy(old_aliases), "inputs": [completed["manifest"], completed["reference_check"], {"id": "index"}],
                   "rows": copy.deepcopy(rows[1:]), "accepted_reference_count": 4, "humanml_reference_count": 4,
                   "original_motion_cross_split": [], "directory_subject_candidate_cross_split": []}
        return [completed, view_manifest, view_check, parent, manifest, checked, old_aliases, aliases]

    def test_exact_exclusion_is_admitted_without_mutating_any_evidence(self):
        docs = self.fixture()
        original = copy.deepcopy(docs)
        self.assertEqual(validate_documents(*docs), docs[0]["split_clips"])
        self.assertEqual(docs, original)

    def test_hidden_gate_reference_or_split_change_is_rejected(self):
        for change in ("gate", "reference", "heldout", "duration"):
            docs = self.fixture()
            if change == "gate":
                docs[2]["clips"][1]["gates"]["0"] = False
            elif change == "reference":
                docs[1]["motions"][1]["output"]["path"] = "different"
            elif change == "heldout":
                docs[0]["split_clips"]["development"] = ["b"]
            else:
                docs[0]["qualified_duration_s"]["train"] = 6.
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_documents(*docs)

    def test_stale_missing_or_conflicted_source_reaudit_is_rejected(self):
        for change in ("parent_input", "new_index", "missing_row", "conflict"):
            docs = self.fixture()
            if change == "parent_input":
                docs[6]["inputs"][0] = {"id": "stale"}
            elif change == "new_index":
                docs[7]["inputs"][2] = {"id": "different_index"}
            elif change == "missing_row":
                docs[7]["rows"].pop()
            else:
                docs[6]["rows"][1]["directory_subject_candidate"] = "test"
                docs[7]["rows"][0]["directory_subject_candidate"] = "test"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_documents(*docs)

    def test_status_or_unresolved_identity_cannot_be_misrepresented(self):
        for change in ("status", "certified", "rerun", "exclusions"):
            docs = self.fixture()
            if change == "status":
                docs[0]["status"] = PARENT_STATUS
            elif change == "certified":
                docs[0]["source_isolation_certified"] = True
            elif change == "rerun":
                docs[0]["source_exclusion"]["geometry_checks_rerun"] = True
            else:
                docs[0]["source_exclusion"]["excluded_train_clips"]["b"] = {"made_up": True}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_documents(*docs)


if __name__ == "__main__":
    unittest.main()
