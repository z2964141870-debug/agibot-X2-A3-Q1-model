"""Small native Adam fixtures for partial checkpoint retention and rejection."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import parent


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class PartialRetentionTest(unittest.TestCase):
    def test_retains_networks_adam_and_sampler_without_optimizer_steps(self):
        self.exercise(False)

    def test_rejects_reset_parent_adam_counter(self):
        self.exercise(True)

    def exercise(self, corrupt):
        import torch

        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = [torch.nn.Linear(2, 1) for _ in range(3)]
            optimizers = [torch.optim.Adam(module.parameters()) for module in modules]
            counts = (19490, 1899, 37980)
            for module, optimizer, count in zip(modules, optimizers, counts):
                module(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                for state in optimizer.state.values():
                    state["step"].fill_(count)
            manifest = {key: {} for key in parent.CONTRACT_KEYS}
            manifest.update(assets={"contact_protocol": {"version": "Q1_FOOT_PRIORITY_V2"}},
                            source_files=[], train_clips=[], learning_rate_step_offset=0)
            source = {"manifest": manifest, "step": 5000, "wm_updates": 1949, "policy_updates": 1899,
                      "bin_return": torch.tensor([.2, .4]), "bin_memory": torch.tensor([3., 4.])}
            source.update({key: copy.deepcopy(module.state_dict()) for key, module in zip(parent.MODEL_KEYS, modules)})
            source.update({key: copy.deepcopy(optimizer.state_dict()) for key, optimizer in zip(parent.OPTIMIZER_KEYS, optimizers)})
            if corrupt:
                for state in source["wm_opt"]["state"].values():
                    state["step"].zero_()
            checkpoint = root / "step_005000.pt"
            torch.save(source, checkpoint)
            admission = root / "admission.json"
            admission.write_text(json.dumps(manifest))
            evidence = {"checkpoint": parent.identity(checkpoint), "admission": parent.identity(admission),
                        "parent_counts": {"step": 5000, "wm_updates": 1949, "policy_updates": 1899}}
            fresh_modules = [torch.nn.Linear(2, 1) for _ in range(3)]
            fresh_opts = [torch.optim.Adam(module.parameters()) for module in fresh_modules]
            sim = SimpleNamespace(bin_return=torch.zeros(2), bin_memory=torch.ones(2))
            if corrupt:
                with self.assertRaisesRegex(ValueError, "Adam counter mismatch"):
                    parent.load_parent(evidence, manifest, fresh_modules, fresh_opts, sim)
                return
            result = parent.load_parent(evidence, manifest, fresh_modules, fresh_opts, sim)
            self.assertEqual(result["learning_rate_step_offset"], 5000)
            self.assertEqual(result["parent_terminal_state"], "FAILED_UNCHANGED_NOT_COMPLETED")
            for before, after in zip(modules, fresh_modules):
                for a, b in zip(before.parameters(), after.parameters()):
                    self.assertTrue(torch.equal(a, b))
            for before, after, count in zip(optimizers, fresh_opts, counts):
                for old, new in zip(before.state.values(), after.state.values()):
                    self.assertEqual(int(new["step"]), count)
                    self.assertTrue(torch.equal(old["exp_avg"], new["exp_avg"]))
                    self.assertTrue(torch.equal(old["exp_avg_sq"], new["exp_avg_sq"]))
            self.assertTrue(torch.equal(sim.bin_return, source["bin_return"]))
            self.assertTrue(torch.equal(sim.bin_memory, source["bin_memory"]))


if __name__ == "__main__":
    unittest.main()
