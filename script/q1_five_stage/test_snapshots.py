"""A passive save must reject accidental model or RNG mutation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from snapshots import install


class SnapshotTests(unittest.TestCase):
    def run_save(self, hashes=("model", "model"), rng=("rng", "rng"), transitions=0):
        with tempfile.TemporaryDirectory() as root:
            trainer = SimpleNamespace(it=0, checkpoint=lambda: {"iteration": 0})
            trainer.save_checkpoint = lambda: Path(root) / "ckpt_0.pt"
            record = dict(robot="q1", optimizer_updates={"wm_opt": 0},
                          collected_environment_transitions=transitions)
            plan = dict(output=root, code_commit="a" * 40)
            install(trainer, plan, record)
            with patch("snapshots.model_sha", side_effect=hashes), patch("snapshots.rng_sha", side_effect=rng):
                trainer.save_checkpoint()
            self.assertEqual(trainer.checkpoint()["actual_training"]["optimizer_updates"], {"wm_opt": 0})
            self.assertEqual(record["zero_snapshot_audit"]["status"], "PASSED")
            self.assertTrue((Path(root) / "zero_snapshot_q1.json").is_file())

    def test_passive_save_keeps_parameters_and_rng(self):
        self.run_save()

    def test_parameter_mutation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "changed parameters or RNG"):
            self.run_save(hashes=("before", "after"))

    def test_rng_mutation_fails(self):
        with self.assertRaisesRegex(RuntimeError, "changed parameters or RNG"):
            self.run_save(rng=("before", "after"))

    def test_zero_snapshot_cannot_follow_collection(self):
        with self.assertRaisesRegex(RuntimeError, "precede all collection"):
            self.run_save(transitions=1)

    def test_tensor_hashes_include_buffers_without_advancing_cpu_rng(self):
        try:
            import torch
            from snapshots import model_sha, rng_sha
        except ImportError:
            self.skipTest("The real tensor check runs in the existing server Torch environment")
        module = torch.nn.Linear(2, 1)
        module.register_buffer("counter", torch.tensor(1.))
        trainer = SimpleNamespace(wm=module)
        before = rng_sha()
        initial = model_sha(trainer)
        self.assertEqual(before, rng_sha())
        with torch.no_grad():
            module.counter.add_(1.)
        self.assertNotEqual(initial, model_sha(trainer))
        self.assertEqual(before, rng_sha())

    def test_real_tensor_snapshot_is_passive_and_saves_actual_counters(self):
        try:
            import torch
        except ImportError:
            self.skipTest("The real tensor save runs in the existing server Torch environment")
        with tempfile.TemporaryDirectory() as root:
            module = torch.nn.Linear(2, 1)
            path = Path(root) / "ckpt_0.pt"
            trainer = SimpleNamespace(it=0, wm=module, checkpoint=lambda: {"wm": module.state_dict()})

            def save():
                torch.save(trainer.checkpoint(), path)
                return path

            trainer.save_checkpoint = save
            record = dict(robot="q1", optimizer_updates={"wm_opt": 0}, collected_environment_transitions=0)
            install(trainer, dict(output=root, code_commit="a" * 40), record)
            trainer.save_checkpoint()
            value = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(value["actual_training"]["optimizer_updates"], {"wm_opt": 0})
            self.assertEqual(record["zero_snapshot_audit"]["status"], "PASSED")
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
