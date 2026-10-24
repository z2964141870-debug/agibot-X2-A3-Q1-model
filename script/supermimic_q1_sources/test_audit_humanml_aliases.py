import unittest

from audit_humanml_aliases import alias_key, indexed_match, original_key, overlap_groups


class HumanmlAliasTests(unittest.TestCase):
    def test_original_index_example_and_mirror_have_one_identity(self):
        self.assertEqual(alias_key("humanml/000000"), ("000000.npy", False))
        self.assertEqual(alias_key("humanml/M000000"), ("000000.npy", True))
        event = original_key("./pose_data/KIT/3/kick_high_left02_poses.npy")
        self.assertEqual(event, "KIT/3/kick_high_left02")
        self.assertEqual(indexed_match(event, {"KIT/3/kick_high_left02_stageii.npz": {}}),
                         ("KIT/3/kick_high_left02_stageii.npz", "EXACT_RELATIVE_EVENT_NAME_NOT_BYTE_IDENTITY"))

    def test_invalid_paths_and_unexpected_mirror_names_are_rejected(self):
        for value in ("humanml/1", "humanml/MM000000", "humanml/000000_chunk_0000", "cmu/000000"):
            with self.assertRaises(ValueError):
                alias_key(value)
        for value in ("../pose_data/KIT/3/x_poses.npy", "/pose_data/KIT/3/x_poses.npy", "./pose_data/KIT/3/x.npz"):
            with self.assertRaises(ValueError):
                original_key(value)

    def test_collection_rename_is_explicitly_only_candidate_evidence(self):
        self.assertEqual(indexed_match("MPI_HDM05/bk/HDM_bk_01-01_01_120",
                                      {"HDM05/bk/HDM_bk_01-01_01_120_stageii.npz": {}})[1],
                         "COLLECTION_RENAME_CANDIDATE_EXCLUSION_ONLY")

    def test_distinct_crops_and_mirrors_do_not_hide_cross_split_originals(self):
        rows = [{"clip": "train_crop", "split": "train", "event": "KIT/3/kick"},
                {"clip": "test_mirror", "split": "test", "event": "KIT/3/kick"},
                {"clip": "unrelated", "split": "train", "event": "KIT/4/walk"}]
        self.assertEqual(overlap_groups(rows, "event"),
                         [{"key": "KIT/3/kick", "splits": ["test", "train"], "clips": ["train_crop", "test_mirror"]}])


if __name__ == "__main__":
    unittest.main()
