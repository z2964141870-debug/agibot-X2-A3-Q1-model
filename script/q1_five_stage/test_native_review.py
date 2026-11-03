"""Native replay must fail closed and use only valid noninitial transitions."""

import copy
from pathlib import Path
import unittest

from native_review import episode_end, ready_training


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(purpose="train", robots=["g1", "q1"], code_commit="a" * 40,
                         cfg=dict(iterations=160000), output="/opt/data/private/ZHY/data/train")
        self.plan_id = dict(path="/opt/data/private/ZHY/data/plan.json", bytes=20, sha256="b" * 64)
        self.pipeline = dict(status="SHARED_TRAINING_COMPLETED_PENDING_EVALUATION",
                             formal_training_completed=True, formal_plan=self.plan_id)
        self.checkpoints = {robot: dict(path=str(Path(self.plan["output"]) / "run" / robot / "ckpt_0.pt"),
                                       bytes=30, sha256=robot[0] * 64) for robot in ("g1", "q1")}
        self.audit = dict(status="PASSED", cuda_initialized=False, iteration=0,
            records=[dict(robot=robot, iteration=0, checkpoint=self.checkpoints[robot])
                     for robot in ("g1", "q1")])

    def run_ready(self, plan=None, pipeline=None, audit=None, robot="q1", checkpoint=None):
        return ready_training(plan or self.plan, pipeline or self.pipeline, self.plan_id,
                              audit or self.audit, robot, checkpoint or self.checkpoints[robot])

    def test_requires_formal_completed_audited_matching_robot(self):
        self.assertEqual(self.run_ready(), 0)
        mutations = (("plan", "purpose", "profile"), ("plan", "code_commit", "abc"),
            ("pipeline", "status", "FAILED"), ("pipeline", "formal_training_completed", False),
            ("pipeline", "formal_plan", {}), ("audit", "status", "FAILED"),
            ("audit", "cuda_initialized", True), ("audit", "iteration", 8))
        for target, key, value in mutations:
            changed = copy.deepcopy(getattr(self, target))
            changed[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(RuntimeError):
                self.run_ready(**{target: changed})
        with self.assertRaises(RuntimeError):
            self.run_ready(checkpoint=self.checkpoints["g1"])
        changed = copy.deepcopy(self.audit)
        changed["records"] = changed["records"][:1]
        with self.assertRaises(RuntimeError):
            self.run_ready(audit=changed)
        changed = copy.deepcopy(self.audit)
        changed["records"][1]["iteration"] = 8
        with self.assertRaises(RuntimeError):
            self.run_ready(audit=changed)


class EpisodeTimingTests(unittest.TestCase):
    def test_terminal_fall_never_enters_valid_prefix_even_at_clip_end(self):
        self.assertEqual(episode_end(1, 10, True, False),
                         dict(complete=False, failure_frame=1, valid_steps=0))
        self.assertEqual(episode_end(9, 10, True, True),
                         dict(complete=False, failure_frame=9, valid_steps=8))

    def test_requires_original_clip_end(self):
        self.assertIsNone(episode_end(1, 10, False, False))
        self.assertEqual(episode_end(9, 10, False, True),
                         dict(complete=True, failure_frame=None, valid_steps=9))
        for args in ((8, 10, False, True), (9, 10, False, False),
                     (0, 10, False, False), (True, 10, False, False)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                episode_end(*args)


if __name__ == "__main__":
    unittest.main()
