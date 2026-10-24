"""Keep the IK sensitivity sample separate from every known held-out directory."""

import copy
import unittest

from probe_walking_retarget import select_training
from test_select_walking_holdout import alias, motion


class TrainingProbeTests(unittest.TestCase):
    def test_known_heldout_and_reserved_directories_excluded(self):
        rows = [motion(f"{i:06d}") for i in range(1, 8)]
        rows.append(motion("000008", "test"))
        aliases = dict(alias(f"{i:06d}", f"BMLrub/rub{i:03d}/circle_walk") for i in range(1, 8))
        aliases.update([alias("000008", "BMLrub/rub001/other_walk")])
        selected, proof = select_training(rows, aliases, ["BMLrub/rub002"])
        self.assertEqual(len(selected), 4)
        self.assertFalse({info["directory_subject_candidate"] for _, info in selected}
                         & {"BMLrub/rub001", "BMLrub/rub002"})
        self.assertTrue(all(row["split"] == "train" for row, _ in selected))
        self.assertIn("BMLrub/rub001", proof["excluded_directory_candidates"])

    def test_deterministic_one_directory_and_inputs_unchanged(self):
        rows = [motion(f"{i:06d}") for i in range(1, 7)]
        aliases = dict(alias(f"{i:06d}", f"BMLrub/rub{i:03d}/circle_walk") for i in range(1, 7))
        aliases.update([alias("000006", "BMLrub/rub001/other_walk")])
        before = copy.deepcopy(rows)
        selected, _ = select_training(rows, aliases, [])
        reversed_selected, _ = select_training(list(reversed(rows)), aliases, [])
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(rows, before)
        self.assertEqual(len({info["directory_subject_candidate"] for _, info in selected}), 4)

    def test_refuses_insufficient_pool_instead_of_using_heldout(self):
        with self.assertRaises(ValueError):
            select_training([motion("000001")], dict([alias("000001", "BMLrub/rub001/circle_walk")]), [])


if __name__ == "__main__":
    unittest.main()
