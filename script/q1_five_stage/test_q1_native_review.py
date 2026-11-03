"""A policy must preview t+1 and hold the final reference, never cross clips."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from q1_native_review import execute, reference_window_indices


class ReferenceWindowTests(unittest.TestCase):
    def test_common_plant_rejects_unfinished_training_before_policy_import(self):
        args = SimpleNamespace(evaluation_commit="a" * 40, plan=Path("plan"), pipeline=Path("pipeline"),
                               audit=Path("audit"), frozen=Path("frozen"), checkpoint=Path("checkpoint"))
        plan = dict(purpose="train", robots=["g1", "q1"], code_commit="a" * 40)
        with patch("q1_native_review.read", side_effect=[plan, dict(status="FAILED"), {}, {}]), \
             patch("q1_native_review.identity", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "has not completed"):
                execute(args)

    def test_reference_lead_and_tail_hold(self):
        self.assertEqual(reference_window_indices(0, 20), list(range(1, 11)))
        self.assertEqual(reference_window_indices(17, 20), [18] + [19] * 9)
        self.assertEqual(reference_window_indices(18, 20), [19] * 10)

    def test_no_policy_after_the_final_transition(self):
        for frame, length in ((-1, 20), (19, 20), (0, 1)):
            with self.subTest(frame=frame), self.assertRaises(RuntimeError):
                reference_window_indices(frame, length)


if __name__ == "__main__":
    unittest.main()
