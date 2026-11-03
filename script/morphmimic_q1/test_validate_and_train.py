"""Reject partial capacity evidence before authorizing an unattended formal run."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from validate_and_train import COUNTS, build_plans, check_native, check_phase, check_profile


HERE = Path(__file__).resolve().parent


class LaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "newgpu_train01_20261103.json").read_text())
        self.template = json.loads((HERE / "full1h_shared_profile_e2048_handoff_v04_20261103.json").read_text())
        self.profile, self.formal = build_plans(self.config, self.template, "a" * 40)

    def workers(self):
        return [dict(status="COMPLETED", robot=robot, code_commit="a" * 40,
            formal_training=False, iterations_completed=8, reference_clips=559,
            common_clip_names=list(range(559)), collected_environment_transitions=450560,
            optimizer_updates=COUNTS.copy(), optimizer_adam_steps={k: [v] for k, v in COUNTS.items()},
            native_rank_checks_required=True, native_rank_checks_passed=True, peer_audit=[{}, {}],
            resolved_cfg=self.profile["cfg"].copy(), nonfinite_updates={"wm/nonfinite_steps": 0},
            latent_draws=[{"it": 0}], last_latent_draw={"it": 7}, shared_parameter_sha={"wm": "same"},
            initial_buffers={"joint_presence": robot}, iteration_seconds=[2.] * 8)
            for robot in ("g1", "q1")]

    def test_full_profile_allows_only_fresh_full_exposure_training(self):
        check_profile(self.workers(), self.profile)
        self.assertEqual(self.formal["cfg"]["iterations"], 80000)
        self.assertEqual(self.formal["cfg"]["num_envs"], 2048)
        self.assertEqual(self.formal["cfg"]["wm_only_iters"], 1000)
        self.assertNotEqual(self.profile["output"], self.formal["output"])
        self.assertEqual(self.template["purpose"], "profile")
        changed = {k for k in self.profile["cfg"] if self.profile["cfg"][k] != self.formal["cfg"][k]}
        self.assertEqual(changed, {"iterations", "wm_only_iters", "ckpt_every", "log_every"})

    def test_partial_failed_or_different_profile_cannot_start_training(self):
        for key, value in (("status", "FAILED"), ("iterations_completed", 7),
                           ("reference_clips", 2), ("native_rank_checks_passed", False),
                           ("optimizer_updates", {}), ("optimizer_adam_steps", {}),
                           ("collected_environment_transitions", 1), ("code_commit", "b" * 40),
                           ("nonfinite_updates", {"wm/nonfinite_steps": 1})):
            with self.subTest(key=key):
                workers = self.workers()
                workers[1][key] = value
                with self.assertRaises(RuntimeError):
                    check_profile(workers, self.profile)
        workers = self.workers()
        workers[1]["resolved_cfg"]["imag_batch"] = 1024
        with self.assertRaises(RuntimeError):
            check_profile(workers, self.profile)

    def test_shared_parameter_draw_and_robot_buffer_checks_are_required(self):
        for key, value in (("shared_parameter_sha", {"wm": "different"}),
                           ("last_latent_draw", {"it": 6}),
                           ("initial_buffers", {"joint_presence": "g1"})):
            workers = self.workers()
            workers[1][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_profile(workers, self.profile)

    def test_failed_outer_guard_blocks_successful_worker_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = dict(status="COMPLETED", purpose="profile", code_commit="a" * 40,
                guard=dict(status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0)))
            (root / "status.json").write_text(json.dumps(result))
            check_phase(root, "profile", "a" * 40)
            result["status"] = "FAILED"
            (root / "status.json").write_text(json.dumps(result))
            with self.assertRaises(RuntimeError):
                check_phase(root, "profile", "a" * 40)

    def test_smaller_training_or_profile_weight_resume_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["formal_cfg_overrides"]["iterations"] = 40000
        with self.assertRaises(RuntimeError):
            build_plans(invalid, self.template, "a" * 40)
        invalid = copy.deepcopy(self.config)
        invalid["formal_cfg_overrides"]["resume"] = "/profile/ckpt_8.pt"
        with self.assertRaises(RuntimeError):
            build_plans(invalid, self.template, "a" * 40)

    def test_native_q1_requires_actual_torque_callbacks_and_zero_updates(self):
        result = dict(status="PASSED", robot="q1", environments=16, control_steps=24,
            environment_transitions=384, new_optimizer_updates=0, batched_physics_steps=240,
            reward_max_abs_error=0., observation_max_abs_error=0., target_max_abs_error=0.,
            torque_callback=dict(calls=241, max_abs_error_nm=1e-6))
        check_native(result, "q1")
        for key, value in (("new_optimizer_updates", 1), ("batched_physics_steps", 96),
                           ("torque_callback", dict(calls=1, max_abs_error_nm=0.))):
            invalid = copy.deepcopy(result)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check_native(invalid, "q1")


if __name__ == "__main__":
    unittest.main()
