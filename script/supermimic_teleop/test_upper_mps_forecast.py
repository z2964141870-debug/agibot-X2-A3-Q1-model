"""Pinned MPS numerical parity, checkpoint and independent admission tests."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from upper_mps_forecast import (MPSBudget, audit_cpu_tensors, cpu_tree, execute,
    fit_normalization, make_model, prediction, require_mps, specification)

try:
    import torch
except ImportError:
    torch = None


class MPSAdmissionTests(unittest.TestCase):
    def test_preregistration_matches_fixed_implementation(self):
        path = Path(__file__).with_name("upper_mps_forecast_plan_20261106.json")
        self.assertEqual(json.loads(path.read_text()), specification())

    def test_normalization_only_uses_train_and_has_physical_noise_floor(self):
        history = np.full((2, 25, 8), .3)
        normalization = fit_normalization(history, split="train")
        np.testing.assert_array_equal(normalization["scale"], np.full(200, .01))
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "Only train"):
                fit_normalization(history, split=split)

    def test_unavailable_backend_is_rejected_without_cpu_fallback(self):
        fake = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
            is_built=lambda: False, is_available=lambda: False)))
        with self.assertRaisesRegex(ValueError, "actual local Apple MPS"):
            require_mps(fake)

    def test_missing_or_low_system_memory_admission_is_rejected(self):
        for output in ("unknown", "System-wide memory free percentage: 24%\n"):
            with patch("upper_mps_forecast.subprocess.run", return_value=SimpleNamespace(stdout=output)):
                with self.assertRaisesRegex(RuntimeError, "headroom"):
                    MPSBudget().check()

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/"data/supermimic_for_q1/existing"
            output.mkdir(parents=True)
            marker = output/"result.json"
            marker.write_text("original")
            args = SimpleNamespace(local_mps_study=True, project_root=root,
                output=output, gpu_lock=root/"data/locks/mps.lock")
            with self.assertRaisesRegex(ValueError, "new independent"):
                execute(args)
            self.assertEqual(marker.read_text(), "original")


@unittest.skipIf(torch is None, "Requires the dedicated pinned Torch runtime")
class MPSNumericalTests(unittest.TestCase):
    def test_zero_head_is_hold_and_saved_weights_are_finite_cpu(self):
        torch.manual_seed(4)
        model = make_model(torch)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 94672)
        history = np.random.default_rng(5).normal(0., .1, (2, 25, 8))
        normalization = fit_normalization(history, split="train")
        predicted = prediction(torch, model, history, normalization, device="cpu")
        np.testing.assert_array_equal(predicted, np.repeat(history[:, -1, None], 10, axis=1))
        snapshot = cpu_tree(torch, model.state_dict())
        self.assertEqual(audit_cpu_tensors(torch, snapshot), 6)
        with self.assertRaises(ValueError):
            audit_cpu_tensors(torch, {"broken": torch.tensor(float("nan"))})

    def test_actual_mps_matches_cpu_for_nonzero_nonlinear_forecast(self):
        require_mps(torch)
        torch.set_num_threads(1)
        torch.mps.set_per_process_memory_fraction(.125)
        torch.manual_seed(6)
        model = make_model(torch)
        torch.nn.init.normal_(model[-1].weight, std=.01)
        torch.nn.init.normal_(model[-1].bias, std=.01)
        history = np.random.default_rng(7).normal(0., .1, (8, 25, 8))
        normalization = fit_normalization(history, split="train")
        expected = prediction(torch, model, history, normalization, device="cpu")
        actual = prediction(torch, model.to("mps"), history, normalization)
        torch.mps.synchronize()
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
        self.assertLess(torch.mps.driver_allocated_memory(), 1024**3)
        model.to("cpu")
        torch.mps.empty_cache()


if __name__ == "__main__":
    unittest.main()
