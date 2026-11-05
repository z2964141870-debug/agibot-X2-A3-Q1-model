"""Training-source boundaries and the optional native Torch model contract."""

from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from forecast_data import ForecastWindow, source_support
from forecast_geometry import FEATURE_DIM, TARGET_DIM, decode_targets, encode_history
from forecast_model import ForecastNormalization, ForecastSpec, NormalizationFit, make_model
from test_forecast_geometry import motion


def training_window():
    positions, quaternions = motion()
    return ForecastWindow("train/fixture.npz", "train", 24, tuple(range(25)), tuple(range(25, 35)),
        positions[:25], quaternions[:25], positions[25:35], quaternions[25:35],
        source_support(24)["dependency_end_ns"])


class ForecastModelTests(unittest.TestCase):
    def test_fitting_and_loading_reject_heldout_sources(self):
        fit = NormalizationFit()
        window = training_window()
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "training"):
                fit.update(replace(window, split=split))
        fit.update(window)
        frozen = fit.freeze()
        record = frozen.to_dict()
        restored = ForecastNormalization.from_dict(record, {window.relative})
        np.testing.assert_array_equal(restored.input_mean, frozen.input_mean)
        with self.assertRaisesRegex(ValueError, "held-out"):
            ForecastNormalization.from_dict(record, {"other/train.npz"})
        self.assertEqual(frozen.input_count, 25)
        self.assertEqual(frozen.target_count, 10)

    def test_zero_output_decodes_to_hold_even_with_nonzero_training_label_mean(self):
        fit = NormalizationFit()
        window = training_window()
        fit.update(window)
        normalization = fit.freeze()
        history = encode_history(window.input_positions, window.input_quaternions)
        normalized = normalization.inputs(history.features)
        self.assertEqual(normalized.shape, (25, FEATURE_DIM))
        decoded = decode_targets(normalization.physical(np.zeros((10, TARGET_DIM))), history.anchor)
        np.testing.assert_allclose(decoded.positions, np.repeat(window.input_positions[-1:], 10, axis=0), atol=3e-15)
        self.assertFalse(normalization.to_dict()["target_mean_subtracted"])
        self.assertTrue(np.all(normalization.target_scale[:72] >= .01))
        self.assertTrue(np.all(normalization.target_scale[72:] >= .05))
        self.assertFalse(normalization.input_scale.flags.writeable)

    def test_target_standardization_roundtrip_preserves_physical_time_and_units(self):
        fit = NormalizationFit()
        fit.update(training_window())
        normalization = fit.freeze()
        targets = np.linspace(-.2, .2, 10*TARGET_DIM).reshape(10, TARGET_DIM)
        restored = normalization.physical(normalization.targets(targets))
        np.testing.assert_allclose(restored, targets, atol=2e-8, rtol=1e-7)
        for values in (np.zeros((10, 143)), np.full((10, TARGET_DIM), np.nan)):
            with self.assertRaises(ValueError):
                normalization.physical(values)

    def test_model_spec_does_not_allow_bidirectional_or_longer_preview_variants(self):
        self.assertFalse(ForecastSpec().to_dict()["bidirectional"])
        self.assertEqual(ForecastSpec().to_dict()["layers"], 1)
        for options in (dict(history_frames=26), dict(horizon_frames=11), dict(hidden_size=1024)):
            with self.assertRaises(ValueError):
                ForecastSpec(**options)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the admitted server Torch runtime")
    def test_torch_shape_hold_initialization_backward_and_cpu_state_roundtrip(self):
        import torch
        torch.manual_seed(20261106)
        model = make_model()
        inputs = torch.randn(2, 25, FEATURE_DIM)
        result = model(inputs)
        self.assertEqual(tuple(result.shape), (2, 10, TARGET_DIM))
        self.assertEqual(torch.count_nonzero(result).item(), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        loss = (result-1.).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()))
        optimizer.step()
        self.assertGreater(torch.count_nonzero(model(inputs)).item(), 0)
        restored = make_model()
        restored.load_state_dict({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        self.assertTrue(torch.equal(model(inputs), restored(inputs)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
