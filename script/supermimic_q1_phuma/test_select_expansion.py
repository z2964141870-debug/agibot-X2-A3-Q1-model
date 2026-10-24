"""Protect source isolation when growing the PHUMA reference pool."""

import unittest

from select_expansion import choose


def row(group, split="train", digest=None, member=None):
    return {"source_group": group, "source_key": group, "split": split,
            "sha256": digest or group, "member": member or group,
            "source_schema_valid": True, "moderate_motion_candidate": True,
            "bucket": "locomotion", "source_statistics": {"source_duration_s": 4.}}


QUOTAS = {key: {"locomotion": 10} for key in ("train", "development", "test")}


class ExpansionSelectionTests(unittest.TestCase):
    def test_same_recording_and_previous_candidate_are_not_reused(self):
        rows = [row("a", member="a1"), row("a", member="a2"), row("old"), row("b")]
        selected, proof = choose(rows, [row("old")], QUOTAS)
        self.assertEqual({x["source_group"] for x in selected}, {"a", "b"})
        self.assertEqual(proof["selected_unique_source_groups"], len(selected))

    def test_identical_bytes_are_reserved_even_when_test_candidate_is_ineligible(self):
        train = row("train", digest="same")
        test = row("test", "test", digest="same")
        test["moderate_motion_candidate"] = False
        selected, _ = choose([train, test], [], QUOTAS)
        self.assertEqual(selected, [])

    def test_test_then_development_priority_and_within_split_duplicates(self):
        rows = [row("train", digest="a"), row("dev", "development", digest="a"),
                row("test", "test", digest="a"), row("dev2", "development", digest="b"),
                row("dev3", "development", digest="b")]
        selected, proof = choose(rows, [], QUOTAS)
        self.assertEqual(len(selected), 2)
        self.assertEqual({x["sha256"] for x in selected}, {"a", "b"})
        self.assertEqual(proof["selected_unique_content_hashes"], 2)

    def test_order_is_deterministic_and_interaction_names_are_excluded(self):
        rows = [row("x/walk"), row("x/chair_sit"), row("x/stairs_up"), row("x/arm_wave")]
        a, _ = choose(rows, [], QUOTAS)
        b, proof = choose(list(reversed(rows)), [], QUOTAS)
        self.assertEqual(a, b)
        self.assertEqual({x["source_group"] for x in a}, {"x/walk", "x/arm_wave"})
        self.assertEqual(proof["excluded_counts"]["named_environment_interaction"], 2)


if __name__ == "__main__":
    unittest.main()
