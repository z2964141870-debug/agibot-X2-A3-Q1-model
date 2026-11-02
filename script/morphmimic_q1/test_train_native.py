"""Check failure containment and prevent accidental full training via profiles."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from train_native import (release_inactive_cuda_cache, tensor_sha, validate_plan,
                          wait_for_workers, wrap_learner_handoffs)


class PhysicsHandoffTests(unittest.TestCase):
    def test_disabled_learner_handoffs_preserve_bound_methods(self):
        trainer, cuda = Mock(), Mock()
        wrap_learner_handoffs(trainer, cuda, 0, {})
        self.assertEqual(trainer.mock_calls, [])
        self.assertEqual(cuda.mock_calls, [])

    def test_each_learner_phase_checks_cache_before_its_original_method(self):
        order, record = [], {}
        cuda = Mock()
        cuda.mem_get_info.return_value = (100, 1000)
        cuda.memory_allocated.return_value = 250
        cuda.memory_reserved.return_value = 800
        cuda.empty_cache.side_effect = lambda: order.append("release")
        methods = {
            "fit_world_model": Mock(side_effect=lambda: order.append("wm") or 10),
            "improve_policy": Mock(side_effect=lambda aux: order.append(aux) or 20),
            "improve_policy_kp": Mock(side_effect=lambda: order.append("kp") or 30)
        }
        trainer = SimpleNamespace(**methods)
        wrap_learner_handoffs(trainer, cuda, 300, record)
        self.assertEqual(trainer.fit_world_model(), 10)
        self.assertEqual(trainer.improve_policy("motion"), 20)
        self.assertEqual(trainer.improve_policy_kp(), 30)
        methods["improve_policy"].assert_called_once_with("motion")
        self.assertEqual(order, ["release", "wm", "release", "motion", "release", "kp"])
        state = record["learner_vram_handoff"]
        self.assertEqual(state["cache_releases"], 3)
        self.assertEqual(set(state["by_phase"]), {"world_model", "motion", "return_kp"})
        self.assertNotIn("physics_vram_handoff", record)

    def test_disabled_handoff_does_not_touch_cuda(self):
        cuda, record = Mock(), {}
        release_inactive_cuda_cache(cuda, 0, record)
        self.assertEqual(cuda.mock_calls, [])
        self.assertEqual(record, {})

    def test_sufficient_global_free_memory_keeps_torch_cache(self):
        cuda, record = Mock(), {}
        cuda.mem_get_info.return_value = (400, 1000)
        release_inactive_cuda_cache(cuda, 300, record)
        cuda.synchronize.assert_not_called()
        cuda.empty_cache.assert_not_called()
        self.assertEqual(record["physics_vram_handoff"]["checks"], 1)
        self.assertEqual(record["physics_vram_handoff"]["cache_releases"], 0)

    def test_low_free_memory_releases_cache_and_records_actual_allocator_values(self):
        cuda, record = Mock(), {}
        cuda.mem_get_info.side_effect = [(100, 1000), (650, 1000)]
        cuda.memory_allocated.return_value = 250
        cuda.memory_reserved.side_effect = [800, 250]
        with patch("train_native.time.monotonic", side_effect=[4., 4.125]):
            release_inactive_cuda_cache(cuda, 300, record)
        cuda.synchronize.assert_called_once()
        cuda.empty_cache.assert_called_once()
        event = record["physics_vram_handoff"]["last_event"]
        self.assertEqual(event["global_free_change_bytes"], 550)
        self.assertEqual(event["torch_reserved_decrease_bytes"], 550)
        self.assertEqual(event["before"]["allocated_bytes"], event["after"]["allocated_bytes"])
        self.assertEqual(event["seconds"], .125)

    def test_handoff_receipts_remain_bounded_during_long_training(self):
        cuda, record = Mock(), {}
        cuda.mem_get_info.return_value = (100, 1000)
        cuda.memory_allocated.return_value = 250
        cuda.memory_reserved.return_value = 800
        for _ in range(20):
            release_inactive_cuda_cache(cuda, 300, record)
        state = record["physics_vram_handoff"]
        self.assertEqual((state["checks"], state["cache_releases"]), (20, 20))
        self.assertEqual(len(state["events"]), 8)
        self.assertEqual(state["last_event"]["check"], 20)


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
        for value in (-1, True, 3.5):
            with self.subTest(handoff=value):
                invalid = copy.deepcopy(plan)
                for key in ("physics_handoff_min_free_bytes", "learner_handoff_min_free_bytes"):
                    invalid[key] = value
                    with self.assertRaises(ValueError):
                        validate_plan(invalid)
                    invalid.pop(key)


if __name__ == "__main__":
    unittest.main()
