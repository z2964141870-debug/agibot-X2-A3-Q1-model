"""Horizon admission, inherited state and one-shot orchestration controls."""

from contextlib import redirect_stdout
import copy
import fcntl
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import horizon_contract as contract
import run_horizon_pair as pair


class ContractTests(unittest.TestCase):
    def source(self):
        value = {key: {"fixed": key} for key in contract.CONTRACT_KEYS}
        value["assets"]["contact_protocol"] = {"version": "Q1_FOOT_PRIORITY_V2"}
        value["policy_trainer"] = {"horizon": 32, "batch_size": 128, "gamma": .99}
        return value

    def test_only_registered_horizon_change_is_admitted(self):
        source = self.source()
        for horizon in (8, 32):
            target = copy.deepcopy(source)
            target["policy_trainer"]["horizon"] = horizon
            target["horizon_experiment"] = {"horizon": horizon}
            contract.validate_contract(source, target)
            target["policy_trainer"]["gamma"] = .98
            with self.assertRaises(ValueError):
                contract.validate_contract(source, target)

    def test_assets_and_unregistered_horizon_are_rejected(self):
        source = self.source()
        target = copy.deepcopy(source)
        target["horizon_experiment"] = {"horizon": 16}
        target["policy_trainer"]["horizon"] = 16
        with self.assertRaises(ValueError):
            contract.validate_contract(source, target)
        target["horizon_experiment"]["horizon"] = 32
        target["policy_trainer"]["horizon"] = 32
        target["assets"]["fixed"] = "changed"
        with self.assertRaises(ValueError):
            contract.validate_contract(source, target)

    def test_loader_uses_strict_contract_and_restores_global_on_error(self):
        source = self.source()
        target = copy.deepcopy(source)
        target["policy_trainer"]["horizon"] = 8
        binding = {"horizon": 8, "sources": []}
        original = contract.initialization.validate_contract

        def failed(evidence, destination, modules, optimizers, sim):
            contract.initialization.validate_contract(source, destination)
            raise RuntimeError("loader fixture")

        with patch.object(contract.initialization, "load_parent", side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, "loader fixture"):
                contract.load_parent({}, target, (), (), None, binding)
        self.assertIs(contract.initialization.validate_contract, original)

    def test_between_arm_check_keeps_seed_and_optimizer_lineage(self):
        manifest = self.source()
        manifest.update({"args": {"output": "a", "seed": 7}, "warmup_steps": 128,
            "model_warmup_updates": 50, "learning_rate_step_offset": 15000,
            "code_commit": "0" * 40, "source_files": [], "initialization": {"retained": True},
            "horizon_experiment": {"arm": "A_H32", "horizon": 32, "plan": "same"}})
        other = copy.deepcopy(manifest)
        other["args"]["output"] = "b"
        other["policy_trainer"]["horizon"] = 8
        other["horizon_experiment"].update({"arm": "B_H8", "horizon": 8})
        self.assertEqual(contract.matched_settings(manifest), contract.matched_settings(other))
        other["args"]["seed"] = 8
        self.assertNotEqual(contract.matched_settings(manifest), contract.matched_settings(other))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Tiny retained-state fixture requires PyTorch")
class NativeRetentionTests(unittest.TestCase):
    def test_h8_retains_networks_adam_and_sampler(self):
        import torch
        from audit import equal_tree
        from test_initialization import ParentFixture

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = ParentFixture(Path(directory))
            fixture.manifest.update({"policy_trainer": {"horizon": 32}, "paper_deviations": []})
            fixture.write(fixture.run / "admission.json", fixture.manifest)
            modules = [torch.nn.Linear(2, 2) for _ in range(3)]
            optimizers = [torch.optim.Adam(m.parameters(), lr=2e-4) for m in modules]
            for module, optimizer in zip(modules, optimizers):
                module(torch.tensor([[.2, -.1]])).square().sum().backward()
                optimizer.step()
            source = {"step": 1, "wm_updates": 1, "policy_updates": 1, "manifest": fixture.manifest,
                      "bin_return": torch.tensor([.3, .7]), "bin_memory": torch.tensor([2., 3.])}
            source.update({k: m.state_dict() for k, m in zip(contract.initialization.MODEL_KEYS, modules)})
            source.update({k: o.state_dict() for k, o in zip(contract.initialization.OPTIMIZER_KEYS, optimizers)})
            torch.save(source, fixture.run / "final.pt")
            fixture.finish()
            evidence = contract.initialization.validated_parent(fixture.pair, fixture.run)
            target = copy.deepcopy(fixture.manifest)
            target["policy_trainer"]["horizon"] = 8
            destination = [torch.nn.Linear(2, 2) for _ in range(3)]
            destination_opts = [torch.optim.Adam(m.parameters(), lr=.1) for m in destination]
            sampler = type("Sampler", (), {"bin_return": torch.zeros(2), "bin_memory": torch.zeros(2)})()
            original = contract.initialization.validate_contract
            result = contract.load_parent(evidence, target, destination, destination_opts, sampler,
                                         {"horizon": 8, "sources": []})
            self.assertIs(contract.initialization.validate_contract, original)
            for before, after in zip(modules, destination):
                self.assertTrue(equal_tree(before.state_dict(), after.state_dict()))
            for before, after in zip(optimizers, destination_opts):
                self.assertTrue(equal_tree(before.state_dict(), after.state_dict()))
            self.assertTrue(torch.equal(source["bin_return"], sampler.bin_return))
            self.assertTrue(torch.equal(source["bin_memory"], sampler.bin_memory))
            self.assertEqual(result["learning_rate_step_offset"], 1)
            self.assertEqual(target["policy_trainer"]["horizon"], 8)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "logs").mkdir()

    def test_health_uses_explicit_runner_after_dependency_imports(self):
        import train_horizon

        runner = contract.continuation.runner
        with patch.object(runner, "reclaim_checkpoint_cache") as reclaim, \
                patch.object(runner, "health", return_value={"oom_kill": 14}) as sample:
            self.assertEqual(train_horizon.health(self.root, 14), {"oom_kill": 14})
            reclaim.assert_called_once_with(self.root)
            sample.assert_called_once_with(self.root, 14)

    def test_fixed_arms_keep_all_three_evaluation_clips(self):
        result, arms = pair.stages(self.root, self.root / "pair",
            {"motion_manifest": {"path": str(self.root / "motions/manifest.json")}},
            ["walk", "squat", "punch"])
        self.assertEqual(len(result), 16)
        self.assertEqual([arm["horizon"] for arm in arms], [32, 8])
        evaluations = [s for s in result if s["expected_status"] == "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"]
        self.assertEqual(len(evaluations), 4)
        for stage in evaluations:
            self.assertEqual(stage["argv"][stage["argv"].index("--clips") + 1:], ["walk", "squat", "punch"])

    def plan(self):
        output = self.root / "pair"
        output.mkdir()
        logs = self.root / "logs/pair"
        logs.mkdir()
        plan = {"root": str(self.root), "output": str(output), "logs": str(logs),
                "baseline_health": {"oom_kill": 14}, "stages": []}
        path = output / "plan.json"
        path.write_text(json.dumps(plan))
        return plan, path

    def test_shared_training_lock_prevents_a_second_controller(self):
        plan, path = self.plan()
        with (self.root / "logs/.q1_batch_pair.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                pair.execute(path)
        self.assertFalse((Path(plan["output"]) / "started.json").exists())

    def test_failure_stops_second_arm_and_cannot_restart(self):
        plan, path = self.plan()
        output = Path(plan["output"])
        plan["stages"] = [{"name": "first", "argv": [sys.executable, "-c", "raise SystemExit(7)"],
            "result": str(output / "first.json"), "expected_status": "SUCCESS", "cpu": True},
            {"name": "second", "argv": [sys.executable, "-c", "raise AssertionError('must not run')"],
             "result": str(output / "second.json"), "expected_status": "SUCCESS", "cpu": True}]
        path.write_text(json.dumps(plan))
        runner = pair.runner
        runner.STOP_REQUESTED = False
        with patch.object(pair, "verify_plan"), patch.object(runner, "verify_frozen"), \
                patch.object(runner, "health", return_value={"oom_kill": 14}), \
                patch.object(runner, "reclaim_checkpoint_cache", return_value={}), \
                patch.object(runner, "process_identity", side_effect=lambda pid: {"pid": pid}), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                pair.execute(path)
            terminal = (output / "receipt.json").read_bytes()
            self.assertEqual(json.loads(terminal)["status"], "FAILED")
            self.assertFalse((Path(plan["logs"]) / "second.log").exists())
            with self.assertRaises(FileExistsError):
                pair.execute(path)
            self.assertEqual((output / "receipt.json").read_bytes(), terminal)


if __name__ == "__main__":
    unittest.main()
