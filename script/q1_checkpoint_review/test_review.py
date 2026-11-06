"""Protect pooling, saved-checkpoint selection and the separate final gate."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import panel_rows, pooled
from review import check_iteration
from native_review import ready_training


class ReviewTests(unittest.TestCase):
    def test_iteration_whitelist_rejects_latest_and_formal_target(self):
        for value in (0, 52000, 160000, True, "48000"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                check_iteration(value)
        self.assertEqual(check_iteration(48000), 48000)

    def test_episode_pooling_uses_denominators_and_iteration_window(self):
        rows = [dict(iteration=1, finished=100, panels=[1., .1, 10., 1., 1., 1.]),
                dict(iteration=500, finished=1, panels=[0., 1., 20., 0., 0., 0.]),
                dict(iteration=501, finished=0, panels=[None]*6)]
        result = pooled(rows, 500)
        self.assertAlmostEqual(result[0], 100/101)
        self.assertAlmostEqual(result[1], 11/101)
        self.assertAlmostEqual(result[2], 1020/101)
        self.assertEqual(result[3], .5)
        self.assertEqual(pooled(rows, 501)[0], 0.)

    def test_missing_episode_denominator_is_not_zero_failure(self):
        rows = panel_rows({50: {"collect/failures": 0, "collect/mean_reward": .05}}, .1)
        self.assertIsNone(rows[0]["panels"][1])
        self.assertEqual(rows[0]["panels"][3], .5)
        self.assertIsNone(pooled(rows, 50)[0])

    def test_live_training_remains_rejected_by_formal_final_entry(self):
        plan = dict(purpose="train", robots=["g1", "q1"], code_commit="a"*40,
                    cfg=dict(iterations=160000), output="/data/train")
        with self.assertRaisesRegex(RuntimeError, "not completed"):
            ready_training(plan, dict(status="TRAINING"), {}, {}, "q1", {})


if __name__ == "__main__":
    unittest.main()
