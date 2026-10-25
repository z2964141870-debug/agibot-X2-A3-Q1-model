"""Protect inherited geometry, held-out sources and bounded view construction."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from filter_phuma_directory_conflicts import check_reserved_sources, documents, exclusions, grab_directory, run


class DirectoryExclusionTests(unittest.TestCase):
    def fixture(self):
        specs = [("a", "train", "GRAB/s1/other"), ("b", "train", "GRAB/s10/other"),
                 ("c", "train", "GRAB/s2/other"), ("game", "train", "game_motion/subset_1/one"),
                 ("human", "train", "humanml/000001"), ("dev", "development", "GRAB/s1/heldout"),
                 ("test", "test", "GRAB/s10/heldout"), ("rejected", "train", "GRAB/s1/rejected")]
        motions, checks = [], []
        splits = {s: [] for s in ("train", "development", "test")}
        for name, split, group in specs:
            accepted = name != "rejected"
            source = {"source_group": group, "split": split, "sha256": name}
            ref = {"path": name + ".npz", "bytes": 1, "sha256": name}
            motions.append({"clip": name, "split": split, "phuma_source": source,
                            "qualified_for_training_pilot": accepted, "output": ref})
            checks.append({"clip": name, "split": split, "source_group": group,
                           "qualified_for_training_pilot": accepted, "reference": ref,
                           "duration_s": 2., "gates": {str(i): True for i in range(19)}})
            if accepted:
                splits[split].append(name)
        names = [row["clip"] for row in checks if row["qualified_for_training_pilot"]]
        manifest = {"status": "parent", "sources": [], "motions": motions, "qualified_clips": names,
                    "source_exclusion": {"preserved_humanml_history": True}}
        checked = {"status": "INDEPENDENT_REFERENCE_CHECK_COMPLETED", "clips": checks,
                   "qualified_clips": names, "source_exclusion": {"preserved_humanml_history": True}}
        parent = {"split_clips": splits}
        aliases = {"rows": [{"clip": "human", "source_group": "humanml/000001",
                             "original_motion_key": "BioMotionLab_NTroje/person1/walk"}]}
        holdout = {"status": "PHUMA_P10_SUPPLEMENTARY_WALKING_HOLDOUT_SELECTED",
                   "selected": [{"member": "reserved", "source_group": "humanml/000999", "sha256": "reserved"}],
                   "walking_holdout": {"role": "SUPPLEMENTARY_TEST_ONLY_NEVER_TRAIN", "source_split_preserved": True,
                       "selected_lineage": [{"member": "reserved", "original_motion_key": "KIT/person2/walk"}]}}
        return manifest, checked, parent, aliases, holdout

    def test_only_qualified_grab_directory_candidates_are_excluded(self):
        manifest, checked, parent, _, _ = self.fixture()
        before = copy.deepcopy((manifest, checked, parent))
        removed = exclusions(manifest, checked, parent["split_clips"])
        self.assertEqual(set(removed), {"a", "b"})
        self.assertEqual(removed["b"], {"grab_named_directory_candidate": "GRAB/s10"})
        self.assertEqual((manifest, checked, parent), before)
        for group in ("GRAB/s1extra/one", "game_motion/subset_1/one", "GRAB/s1/extra/one", "/GRAB/s1/one"):
            self.assertIsNone(grab_directory(group))

    def test_changes_to_parent_coverage_source_or_gates_are_rejected(self):
        for change in ("duplicate", "split", "reference", "gate", "extra", "same_bytes"):
            manifest, checked, parent, _, _ = self.fixture()
            if change == "duplicate":
                checked["clips"].append(copy.deepcopy(checked["clips"][0]))
            elif change == "split":
                checked["clips"][0]["split"] = "test"
            elif change == "reference":
                checked["clips"][0]["reference"] = {"path": "wrong"}
            elif change == "gate":
                checked["clips"][0]["gates"]["0"] = False
            elif change == "extra":
                parent["split_clips"]["train"].append("unknown")
            else:
                manifest["motions"][0]["phuma_source"]["sha256"] = "b"
            with self.subTest(change=change), self.assertRaises(ValueError):
                exclusions(manifest, checked, parent["split_clips"])

    def test_documents_preserve_reference_gates_history_and_heldout(self):
        manifest, checked, parent, _, _ = self.fixture()
        before = copy.deepcopy((manifest, checked, parent))
        provenance = {"mode": "TRAIN_ONLY_GRAB_NAMED_DIRECTORY_CANDIDATE_EXCLUSION",
                      "source_isolation_certified": False, "geometry_checks_rerun": False,
                      "source": {"id": "producer"}, "excluded_train_clips": exclusions(manifest, checked, parent["split_clips"])}
        view, receipt, splits = documents(manifest, checked, parent, provenance)
        self.assertEqual(splits, {"train": ["c", "game", "human"], "development": ["dev"], "test": ["test"]})
        self.assertEqual(view["source_exclusion"], manifest["source_exclusion"])
        for original, updated in zip(checked["clips"], receipt["clips"]):
            self.assertEqual(updated["gates"], original["gates"])
            self.assertEqual(updated["reference"], original["reference"])
            if original["clip"] not in ("a", "b"):
                self.assertEqual(updated, original)
        self.assertEqual((manifest, checked, parent), before)
        for change in ("extra", "missing", "certified", "rerun"):
            altered = copy.deepcopy(provenance)
            if change == "extra":
                altered["excluded_train_clips"]["c"] = {"made_up": True}
            elif change == "missing":
                altered["excluded_train_clips"].pop("a")
            elif change == "certified":
                altered["source_isolation_certified"] = True
            else:
                altered["geometry_checks_rerun"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                documents(manifest, checked, parent, altered)

    def test_reserved_walking_groups_bytes_events_and_directories_are_rejected(self):
        manifest, _, parent, aliases, holdout = self.fixture()
        proof = check_reserved_sources(manifest, parent["split_clips"], aliases, holdout)
        self.assertEqual(proof["parent_references_checked"], 7)
        for change in ("group", "bytes", "event", "directory", "aliases", "role", "coverage"):
            m, _, p, a, h = self.fixture()
            if change == "group":
                h["selected"][0]["source_group"] = "humanml/000001"
            elif change == "bytes":
                h["selected"][0]["sha256"] = "human"
            elif change in ("event", "directory"):
                suffix = "walk" if change == "event" else "other"
                h["walking_holdout"]["selected_lineage"][0]["original_motion_key"] = "BMLrub/person1/" + suffix
            elif change == "aliases":
                a["rows"] = []
            elif change == "role":
                h["walking_holdout"]["role"] = "train"
            else:
                h["walking_holdout"]["selected_lineage"] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_reserved_sources(m, p["split_clips"], a, h)

    def test_storage_rejection_precedes_parent_reads_or_output_creation(self):
        from runtime_guard import StorageLimit

        with TemporaryDirectory() as temp:
            root = Path(temp)
            library = root / "data/supermimic_for_q1/parent"
            library.mkdir(parents=True)
            output = library.parent / "new_view"
            args = SimpleNamespace(root=root, library=library, expected_oom_kill=14, preview=False, output=output)
            with mock.patch("guard_phuma_batch.MemoryBoundBudget.check", side_effect=StorageLimit("reserve")), \
                    mock.patch("filter_phuma_directory_conflicts.validated_view") as read_parent:
                with self.assertRaises(StorageLimit):
                    run(args)
                read_parent.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
