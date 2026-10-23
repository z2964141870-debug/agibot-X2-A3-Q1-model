"""State-retention and lineage checks, with optional tiny CPU Adam fixtures."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import initialization as init


class ParentFixture:
    def __init__(self, root):
        self.pair = root / "pair"
        self.pair.mkdir()
        self.run = self.pair / "A"
        self.run.mkdir()
        (self.run / "final.pt").write_bytes(b"checkpoint fixture")
        self.manifest = {key: {} for key in init.CONTRACT_KEYS}
        self.manifest.update({"assets": {"contact_protocol": {"version": "Q1_FOOT_PRIORITY_V2"}},
                              "source_files": [], "train_clips": [], "learning_rate_step_offset": 0})
        self.write(self.run / "admission.json", self.manifest)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data) + "\n")

    def finish(self):
        final = init.identity(self.run / "final.pt")
        self.write(self.run / "receipt.json", {"status": "TRAINING_FINISHED_PENDING_EVALUATION",
                   "artifacts": [final], "step": 1, "wm_updates": 1, "policy_updates": 1})
        self.write(self.pair / "A_training_audit.json", {"status": "TRAINING_SAVE_AUDIT_PASSED",
                   "gates": {"fixture": True}, "final": final,
                   "training_receipt": init.identity(self.run / "receipt.json")})
        evaluation = self.pair / "eval_final.json"
        self.write(evaluation, {"status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "rows": [],
                   "checkpoint": final, "usability_gates": {"joint_error": False}})
        self.write(self.pair / "A_eval_final_audit.json", {"status": "EVALUATION_TRACE_AUDIT_PASSED",
                   "verified": True, "receipt": init.identity(evaluation)})
        self.write(self.pair / "plan.json", {"arms": [{"name": "A", "run": str(self.run),
                   "evaluations": {"final": {"receipt": str(evaluation)}}}]})
        self.write(self.pair / "comparison.json", {"status": "MATCHED_BATCH_COMPARISON_COMPLETED",
                   "sole_setting_change_verified": True, "plan": init.identity(self.pair / "plan.json")})
        self.write(self.pair / "receipt.json", {"status": "PAIR_COMPLETED",
                   "comparison": init.identity(self.pair / "comparison.json")})
        events = [{"exit_code": 0, "result": init.identity(path)} for path in (
            self.pair / "A_training_audit.json", evaluation, self.pair / "A_eval_final_audit.json")]
        (self.pair / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")


class LineageTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.fixture = ParentFixture(Path(directory.name))
        self.fixture.finish()

    def test_failed_usability_gate_allows_continuation_of_audited_parent(self):
        evidence = init.validated_parent(self.fixture.pair, self.fixture.run)
        self.assertFalse(evidence["usability_gates"]["joint_error"])

    def test_stopped_parent_is_rejected(self):
        path = self.fixture.run / "receipt.json"
        row = json.loads(path.read_text())
        row["status"] = "STOPPED_SIGNAL"
        self.fixture.write(path, row)
        with self.assertRaisesRegex(ValueError, "successful save audit"):
            init.validated_parent(self.fixture.pair, self.fixture.run)

    def test_live_pair_is_rejected(self):
        self.fixture.write(self.fixture.pair / "receipt.json", {"status": "RUNNING"})
        with self.assertRaisesRegex(ValueError, "not completed"):
            init.validated_parent(self.fixture.pair, self.fixture.run)

    def test_modified_comparison_is_rejected(self):
        self.fixture.write(self.fixture.pair / "comparison.json", {"status": "changed"})
        with self.assertRaisesRegex(ValueError, "not completed"):
            init.validated_parent(self.fixture.pair, self.fixture.run)

    def test_unregistered_parent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a registered arm"):
            init.validated_parent(self.fixture.pair, self.fixture.pair / "other")

    def test_unbound_audit_is_rejected(self):
        (self.fixture.pair / "events.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "does not bind"):
            init.validated_parent(self.fixture.pair, self.fixture.run)

    def test_changed_reference_bytes_are_rejected(self):
        path = self.fixture.pair / "reference.npz"
        path.write_bytes(b"reference")
        record = init.identity(path)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            init.check_identity(record)

    def test_contract_rejects_other_robot_or_batch(self):
        source = self.fixture.manifest
        target = json.loads(json.dumps(source))
        target["policy_trainer"] = {"batch_size": 128}
        with self.assertRaisesRegex(ValueError, "policy_trainer"):
            init.validate_contract(source, target)
        target = json.loads(json.dumps(source))
        target["assets"]["contact_protocol"]["version"] = "OLD"
        with self.assertRaisesRegex(ValueError, "assets"):
            init.validate_contract(source, target)

    def test_json_list_and_checkpoint_tuple_are_equivalent(self):
        source, target = dict(self.fixture.manifest), dict(self.fixture.manifest)
        source["policy_trainer"], target["policy_trainer"] = {"betas": (.9, .95)}, {"betas": [.9, .95]}
        init.validate_contract(source, target)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Tiny tensor fixture requires PyTorch")
class NativeRetentionTest(unittest.TestCase):
    def test_networks_adam_moments_and_sampler_are_exactly_retained(self):
        import torch
        from audit import equal_tree

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            fixture = ParentFixture(Path(directory))
            source_modules = [torch.nn.Linear(2, 2) for _ in range(3)]
            source_opts = [torch.optim.Adam(module.parameters(), lr=2e-4) for module in source_modules]
            # Three tiny fixture Adam updates create nonzero moments and counters.
            for module, optimizer in zip(source_modules, source_opts):
                module(torch.tensor([[.2, -.1]])).square().sum().backward()
                optimizer.step()
            source = {"step": 1, "wm_updates": 1, "policy_updates": 1,
                      "manifest": fixture.manifest, "bin_return": torch.tensor([.3, .7]),
                      "bin_memory": torch.tensor([2., 3.])}
            source.update({key: module.state_dict() for key, module in zip(init.MODEL_KEYS, source_modules)})
            source.update({key: optimizer.state_dict() for key, optimizer in zip(init.OPTIMIZER_KEYS, source_opts)})
            torch.save(source, fixture.run / "final.pt")
            fixture.finish()
            evidence = init.validated_parent(fixture.pair, fixture.run)
            target_modules = [torch.nn.Linear(2, 2) for _ in range(3)]
            target_opts = [torch.optim.Adam(module.parameters(), lr=.1) for module in target_modules]
            sampler = type("Sampler", (), {"bin_return": torch.zeros(2), "bin_memory": torch.zeros(2)})()
            result = init.load_parent(evidence, fixture.manifest, target_modules, target_opts, sampler)
            for before, after in zip(source_modules, target_modules):
                self.assertTrue(equal_tree(before.state_dict(), after.state_dict()))
            for before, after in zip(source_opts, target_opts):
                self.assertTrue(equal_tree(before.state_dict(), after.state_dict()))
            self.assertTrue(torch.equal(source["bin_return"], sampler.bin_return))
            self.assertTrue(torch.equal(source["bin_memory"], sampler.bin_memory))
            self.assertEqual(result["learning_rate_step_offset"], 1)
            fixture.manifest["extra"] = "tampered admission"
            fixture.write(fixture.run / "admission.json", fixture.manifest)
            evidence["admission"] = init.identity(fixture.run / "admission.json")
            with self.assertRaisesRegex(ValueError, "admission differs"):
                init.load_parent(evidence, fixture.manifest, target_modules, target_opts, sampler)


if __name__ == "__main__":
    unittest.main()
