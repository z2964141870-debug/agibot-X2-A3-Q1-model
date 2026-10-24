"""Exercise leakage exclusions and immutable source-split provenance."""

import copy
import unittest

from select_walking_holdout import choose


def motion(number, split="train"):
    return {"member": f"data/g1/humanml/{number}_chunk_0000.npy", "source_group": f"humanml/{number}",
            "sha256": number, "source_schema_valid": True, "moderate_motion_candidate": True,
            "bucket": "locomotion", "split": split}


def alias(number, event):
    return number + ".npy", {"source_path": "./pose_data/" + event + "_poses.npy",
                             "new_name": number + ".npy", "start_frame": "0", "end_frame": "100"}


class WalkingSelectionTests(unittest.TestCase):
    def test_original_event_mirror_and_subject_excluded(self):
        aliases = dict([alias("000001", "BMLrub/rub001/circle_walk"),
                        alias("000002", "BioMotionLab_NTroje/rub001/circle_walk"),
                        alias("000003", "BMLrub/rub001/other_walk"),
                        alias("000004", "BMLrub/rub002/circle_walk")])
        rows = [motion("M000001"), motion("000002"), motion("000003"), motion("000004")]
        selected, proof = choose(rows, [motion("000001")], aliases, [])
        self.assertEqual([row["source_group"] for row, _ in selected], ["humanml/000004"])
        self.assertEqual(proof["exclusion_counts"]["previous_original_event_candidate"], 2)
        self.assertEqual(proof["exclusion_counts"]["previous_directory_subject_candidate"], 1)

    def test_one_directory_immutable_split_deterministic_order(self):
        aliases = dict([alias("000001", "BMLrub/rub001/circle_walk"),
                        alias("000002", "BMLrub/rub001/normal_walk"),
                        alias("000003", "BMLrub/rub002/normal_walk")])
        rows = [motion("000001", "test"), motion("000002"), motion("000003", "development")]
        before = copy.deepcopy(rows)
        selected, proof = choose(rows, [], aliases, [])
        reversed_selection, _ = choose(list(reversed(rows)), [], aliases, [])
        self.assertEqual(selected, reversed_selection)
        self.assertEqual(rows, before)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({info["directory_subject_candidate"] for _, info in selected}), 2)
        self.assertEqual(proof["eligible_original_event_candidates"], 3)

    def test_bytes_amass_and_treadmill_excluded(self):
        aliases = dict([alias("000001", "BMLrub/rub001/circle_walk"),
                        alias("000002", "BMLrub/rub002/circle_walk"),
                        alias("000003", "BMLrub/rub003/treadmill_walk")])
        rows = [motion("000001"), motion("000002"), motion("000003")]
        previous = [dict(motion("000009"), source_group="fitness/existing", sha256="000001")]
        selected, proof = choose(rows, previous, aliases,
            [{"relative_path": "BMLrub/rub002/other_stageii.npz"}])
        self.assertEqual(selected, [])
        self.assertEqual(proof["previous_unresolved_namespace_counts"], {"fitness": 1})

    def test_selected_bytes_cannot_repeat_across_different_directories(self):
        aliases = dict([alias("000001", "BMLrub/rub001/normal_walk"),
                        alias("000002", "BMLrub/rub002/normal_walk")])
        rows = [motion("000001"), dict(motion("000002"), sha256="000001")]
        selected, proof = choose(rows, [], aliases, [])
        self.assertEqual(len(selected), 1)
        self.assertEqual(proof["eligible_original_event_candidates"], 2)


if __name__ == "__main__":
    unittest.main()
