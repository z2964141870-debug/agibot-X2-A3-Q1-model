import copy
import unittest

from common import (FIXED, TARGET_TRANSITIONS, capacity_failure, evaluation_gate,
                    matched_checkpoint, morph_settings)


class Gates(unittest.TestCase):
    def cases(self):
        return [dict(clip=clip, seed=seed, final_complete=True, common_steps=20,
                     initial_joint_rmse_rad=.3, final_joint_rmse_rad=.2,
                     initial_body_relative_rmse_m=.03, final_body_relative_rmse_m=.02)
                for clip in FIXED for seed in (0, 1, 2)]

    def test_gate_passes_without_natural_gait_threshold(self):
        self.assertTrue(evaluation_gate(self.cases(), True, True, True)["passed"])

    def test_missing_and_duplicate_cases_fail(self):
        rows = self.cases()
        rows[-1] = copy.deepcopy(rows[0])
        self.assertFalse(evaluation_gate(rows, True, True, True)["passed"])
        self.assertFalse(evaluation_gate(rows[:-1], True, True, True)["passed"])

    def test_early_fall_and_no_common_prefix_fail(self):
        for key, value in (("final_complete", False), ("common_steps", 0)):
            rows = self.cases()
            rows[0][key] = value
            self.assertFalse(evaluation_gate(rows, True, True, True)["passed"])

    def test_missing_nan_and_no_improvement_fail(self):
        for value in (None, float("nan"), .3):
            rows = self.cases()
            rows[0]["final_joint_rmse_rad"] = value
            if value == .3:
                for row in rows:
                    row["final_joint_rmse_rad"] = .3
            self.assertFalse(evaluation_gate(rows, True, True, True)["passed"])

    def test_delivery_is_required(self):
        for args in ((False, True, True), (True, False, True), (True, True, False)):
            self.assertFalse(evaluation_gate(self.cases(), *args)["passed"])

    def test_only_explicit_cuda_oom_allows_capacity_fallback(self):
        self.assertTrue(capacity_failure("CUDA out of memory"))
        for reason in ("MemoryLimit", "Connection closed by peer", "CUDA error 719", "timeout"):
            self.assertFalse(capacity_failure(reason))

    def test_shared_tiers_preserve_exposure(self):
        for env in (1024, 512):
            cfg = morph_settings(env)
            self.assertEqual(int((cfg["warmup_env_steps"] + cfg["iterations"] * 2.5) * env),
                             TARGET_TRANSITIONS)
        with self.assertRaises(RuntimeError):
            morph_settings(2048)

    def test_matching_exposure_requires_actual_counts(self):
        self.assertEqual(matched_checkpoint([dict(global_env_steps=TARGET_TRANSITIONS)])["exposure_relative_difference"], 0)
        with self.assertRaises(RuntimeError):
            matched_checkpoint([dict(global_env_steps=100)])


if __name__ == "__main__":
    unittest.main()
