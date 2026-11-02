"""Check failure containment and prevent accidental full training via profiles."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from train_native import tensor_sha, validate_plan, wait_for_workers


class WorkerLifecycleTests(unittest.TestCase):
    def test_scalar_hazard_parameters_have_lossless_hashes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Run tensor verification in the existing server Torch environment")
        self.assertEqual(tensor_sha(torch.tensor(1.)), tensor_sha(torch.tensor([1.])))
        self.assertNotEqual(tensor_sha(torch.tensor(1.)), tensor_sha(torch.tensor(2.)))

    def test_second_rank_failure_is_seen_while_first_rank_is_running(self):
        first = SimpleNamespace(poll=Mock(return_value=None))
        second = SimpleNamespace(poll=Mock(return_value=2))
        with patch("train_native.time.sleep", side_effect=AssertionError("Failure was missed")):
            with self.assertRaisesRegex(RuntimeError, "rank failed"):
                wait_for_workers([first, second])
        second.poll.assert_called_once()

    def test_success_waits_for_every_rank_to_finish(self):
        first = SimpleNamespace(poll=Mock(side_effect=[None, 0]))
        second = SimpleNamespace(poll=Mock(side_effect=[0, 0]))
        with patch("train_native.time.sleep") as sleep:
            wait_for_workers([first, second])
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(second.poll.call_count, 2)

    def test_profile_cannot_silently_start_long_or_partial_training(self):
        root = "/opt/data/private/ZHY/"
        plan = dict(code_commit="a" * 40, robots=["g1", "q1"], purpose="profile",
            source=root + "data/source", output=root + "data/supermimic_for_q1/profile",
            human_root=root + "data/human", subsets=[root + "data/g1.yaml", root + "data/q1.yaml"],
            cfg=dict(num_envs=512, iterations=8, wm_batch=256, imag_batch=512,
                     kp_imag_batch=512, kp_interface=True, wm_only_iters=0))
        validate_plan(plan)
        for key, value in (("iterations", 9), ("wm_only_iters", 1),
                           ("kp_student", "copy"), ("num_envs", -1)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(plan)
                invalid["cfg"][key] = value
                with self.assertRaises(ValueError):
                    validate_plan(invalid)


if __name__ == "__main__":
    unittest.main()
