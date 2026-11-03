"""Failures must never silently select a smaller profile or formal training."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from controller import MORPH_UPSTREAM, OPTIMIZERS, ROOT, Runner, audit_checkpoint, choose_profile, execute, updates
from contracts import expected_transitions


def success():
    return dict(status="COMPLETED", inputs_unchanged=True,
        guard=dict(status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0)))


def capacity_failure():
    return dict(status="FAILED", inputs_unchanged=True, cgroup_oom_kill_delta=0, gpu_recovery_action="None",
        guard=dict(status="WORKER_FAILED", cleanup=dict(worker_reaped=True, worker_return_code=1)))


class ControllerTests(unittest.TestCase):
    def test_phase_preserves_native_git_runtime_for_every_child(self):
        with tempfile.TemporaryDirectory() as directory, patch("controller.signal.signal"), \
             patch("controller.subprocess.Popen") as child, patch("controller.oom_count", return_value=0), \
             patch("controller.recovery", return_value="None"), patch("controller.save"), \
             patch("controller.read", return_value=dict(status="COMPLETED")), \
             patch.dict("controller.os.environ", {"Q1_TEST_INHERITED": "preserved"}):
            child.return_value.wait.return_value = 0
            runner = Runner(dict(morph_python="/python"), Path(directory), [])
            runner.phase("profile", ["worker"], 60, 1024)
            env = child.call_args.kwargs["env"]
            self.assertEqual(env["GIT_PYTHON_GIT_EXECUTABLE"], str(
                ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))
            self.assertEqual(env["Q1_TEST_INHERITED"], "preserved")

    def test_failed_full_gate_creates_no_experiment_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data/supermimic_for_q1/new"
            config = dict(output=str(output), frozen_data="frozen.json", frozen_data_sha256="a" * 64,
                          morph_upstream=MORPH_UPSTREAM)
            with patch("controller.ROOT", root), patch("controller.identity", return_value=dict(sha256="a" * 64)), \
                 patch("controller.read", return_value={}), patch("controller.validate_frozen", return_value=[]), \
                 patch("controller.final_gate", return_value=dict(passed=False, gate_reason=["missing_final_delivery"])), \
                 patch("controller.Runner") as runner:
                with self.assertRaisesRegex(RuntimeError, "missing_final_delivery"):
                    execute(config, root / "config.json", "a" * 40)
                runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_excess_cache_creates_no_experiment_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "data/supermimic_for_q1/new"
            config = dict(output=str(output), frozen_data="frozen.json", frozen_data_sha256="a" * 64,
                morph_upstream=MORPH_UPSTREAM, profile_maximum_initial_cache_bytes=4 * 1024**3)
            with patch("controller.ROOT", root), patch("controller.identity", return_value=dict(sha256="a" * 64)), \
                 patch("controller.read", return_value={}), patch("controller.validate_frozen", return_value=[]), \
                 patch("controller.final_gate", return_value=dict(passed=True, gate_reason=[])), \
                 patch("guard.profile_cache_admission", side_effect=RuntimeError("Initial cache exceeds")) as admit, \
                 patch("controller.Runner") as runner:
                with self.assertRaisesRegex(RuntimeError, "Initial cache exceeds"):
                    execute(config, root / "config.json", "a" * 40)
                admit.assert_called_once_with(4 * 1024**3)
                runner.assert_not_called()
            self.assertFalse(output.exists())

    def test_cache_limit_reaches_only_the_profile_guard(self):
        for name, configured, expected in (("e1024/profile", True, True), ("e512/profile", True, True),
                ("e1024/train", True, False), ("profile_audit", True, False),
                ("formal_audit_0", True, False), ("e1024/profile", False, False)):
            with self.subTest(name=name, configured=configured), tempfile.TemporaryDirectory() as directory, \
                 patch("controller.signal.signal"), patch("controller.subprocess.Popen") as child, \
                 patch("controller.oom_count", return_value=0), patch("controller.recovery", return_value="None"), \
                 patch("controller.save"), patch("controller.read", return_value=dict(status="COMPLETED")):
                child.return_value.wait.return_value = 0
                config = dict(morph_python="/python")
                if configured:
                    config["profile_maximum_initial_cache_bytes"] = 4 * 1024**3
                Runner(config, Path(directory), []).phase(name, ["worker"], 60, 1024)
                command = child.call_args.args[0]
                flag = "--maximum-initial-cache-bytes"
                self.assertEqual(flag in command, expected)
                if expected:
                    self.assertEqual(command[command.index(flag) + 1], str(4 * 1024**3))

    def test_first_success_does_not_run_another_capacity(self):
        calls = []

        def attempt(envs):
            calls.append(envs)
            return success(), "", dict(environments=envs)

        self.assertEqual(choose_profile(attempt), dict(environments=1024))
        self.assertEqual(calls, [1024])

    def test_only_proven_cuda_oom_uses_one_fresh_smaller_profile(self):
        calls = []

        def attempt(envs):
            calls.append(envs)
            return ((capacity_failure(), "CUDA out of memory", dict(environments=envs))
                    if envs == 1024 else (success(), "", dict(environments=envs)))

        self.assertEqual(choose_profile(attempt), dict(environments=512))
        self.assertEqual(calls, [1024, 512])

    def test_second_failure_stops_without_a_third_attempt(self):
        calls = []

        def attempt(envs):
            calls.append(envs)
            return capacity_failure(), "CUDA out of memory", dict(environments=envs)

        with self.assertRaises(RuntimeError):
            choose_profile(attempt)
        self.assertEqual(calls, [1024, 512])

    def test_device_guard_and_signal_errors_never_fall_back(self):
        for text in ("illegal memory access", "CUDA out of memory; requires reset", "memory reserve reached"):
            calls = []

            def attempt(envs):
                calls.append(envs)
                return capacity_failure(), text, {}

            with self.subTest(text=text), self.assertRaises(RuntimeError):
                choose_profile(attempt)
            self.assertEqual(calls, [1024])

    def test_success_requires_reap_zero_and_unchanged_inputs(self):
        for edit in (lambda p: p.update(inputs_unchanged=False),
                     lambda p: p["guard"]["cleanup"].update(worker_reaped=False),
                     lambda p: p["guard"]["cleanup"].update(worker_return_code=-15)):
            value = copy.deepcopy(success())
            edit(value)
            with self.assertRaises(RuntimeError):
                choose_profile(lambda envs: (value, "", {}))

    def test_actual_adam_targets_preserve_world_model_only_warmup(self):
        cfg = dict(wm_only_iters=2000)
        self.assertEqual(updates(cfg, 0), dict(wm_opt=0, policy_opt=0, critic_opt=0, kp_opt=0, critic_kp_opt=0))
        self.assertEqual(updates(cfg, 2000), dict(wm_opt=20000, policy_opt=0, critic_opt=0, kp_opt=0, critic_kp_opt=0))
        self.assertEqual(updates(cfg, 160000), dict(wm_opt=1600000, policy_opt=158000,
            critic_opt=3160000, kp_opt=158000, critic_kp_opt=3160000))

    def test_real_cpu_checkpoint_requires_both_ranks_and_saved_adam_counts(self):
        try:
            import torch
        except ImportError:
            self.skipTest("The real checkpoint audit runs in the existing server Torch environment")
        with tempfile.TemporaryDirectory() as root:
            cfg = dict(wm_only_iters=0, iterations=8, num_envs=1024, warmup_env_steps=400, utd=.4)
            plan = dict(output=root, robots=["g1", "q1"], cfg=cfg, code_commit="a" * 40)
            counts = updates(cfg, 8)
            for robot in plan["robots"]:
                value = dict(iteration=8, schedule=dict(step=8, world_size=2), cfg=cfg,
                    native_q1_contract=dict(code_commit=plan["code_commit"]),
                    actual_training=dict(iteration=8, optimizer_updates=counts,
                        collected_environment_transitions=expected_transitions(cfg), code_commit=plan["code_commit"],
                        fresh_initialization=True, profile_weights_reused=False), optim={})
                for hook, key in OPTIMIZERS.items():
                    value["optim"][key] = dict(state={0: dict(step=torch.tensor(float(counts[hook])), exp_avg=torch.zeros(2))})
                for name in ("wm", "policy", "critic", "critic_kp", "obs_norm", "goal_norm", "human_norm", "heading_norm"):
                    value[name] = dict(weight=torch.ones(2))
                path = Path(root) / "run" / robot / "ckpt_8.pt"
                path.parent.mkdir(parents=True)
                torch.save(value, path)
            self.assertEqual(audit_checkpoint(plan, 8)["status"], "PASSED")
            value["optim"]["critic_kp"]["state"][0]["step"].fill_(159)
            torch.save(value, path)
            with self.assertRaisesRegex(RuntimeError, "Saved Adam counter mismatch"):
                audit_checkpoint(plan, 8)


if __name__ == "__main__":
    unittest.main()
