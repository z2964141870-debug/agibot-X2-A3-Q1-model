"""Exact linear-map consistency, iid variance and source rejection tests."""

from copy import deepcopy
import unittest

import numpy as np

from upper_forecast import CHANNELS, predict
from upper_forecast_sensitivity import (PLAN_SHA, baseline_affine_map, checked_model,
                                        learned_affine_map, noise_gains)


def model_fixture():
    rng = np.random.default_rng(3)
    return dict(weights=rng.normal(0., .01, (201, 80)),
                mean=rng.normal(0., .1, 200), scale=rng.uniform(.1, .5, 200))


class UpperNoiseSensitivityTests(unittest.TestCase):
    def test_learned_raw_map_matches_forecast_and_finite_differences(self):
        model = model_fixture()
        history = np.random.default_rng(4).normal(0., .1, (2, 25, 8))
        jacobian, intercept = learned_affine_map(model)
        direct = predict(history, "ridge", {"ridge": model}).reshape(2, 80)
        linear = np.einsum("ni,ij->nj", history.reshape(2, 200), jacobian,
                           optimize=False) + intercept
        np.testing.assert_allclose(linear, direct, atol=1e-14)
        for index in (0, 17, 90, 191, 192, 199):
            modified = history.copy()
            modified.reshape(2, 200)[0, index] += 1e-6
            difference = (predict(modified, "ridge", {"ridge": model}).reshape(2, 80)[0]-direct[0])/1e-6
            np.testing.assert_allclose(difference, jacobian[index], atol=2e-9)

    def test_fixed_baseline_maps_match_existing_implementation(self):
        history = np.random.default_rng(5).normal(0., .1, (3, 25, 8))
        for method in ("hold", "cv", "damped_cv"):
            jacobian, intercept = baseline_affine_map(method)
            expected = predict(history, method, {}).reshape(3, 80)
            actual = np.einsum("ni,ij->nj", history.reshape(3, 200), jacobian,
                               optimize=False) + intercept
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        gains = noise_gains(baseline_affine_map("hold")[0])
        np.testing.assert_array_equal(gains["per_horizon_rms_std_gain"], np.ones(10))

    def test_noise_gain_matches_exact_covariance_and_zero_weights_equal_hold(self):
        model = model_fixture()
        model["weights"][:] = 0.
        jacobian, _ = learned_affine_map(model)
        np.testing.assert_array_equal(jacobian, baseline_affine_map("hold")[0])
        artificial = np.zeros((200, 80))
        artificial[0] = 3.
        artificial[1] = 4.
        np.testing.assert_allclose(noise_gains(artificial)["per_horizon_rms_std_gain"], 5.)
        for invalid in (np.zeros((199, 80)), np.full((200, 80), np.nan)):
            with self.assertRaises(ValueError):
                noise_gains(invalid)

    def test_schema_provenance_and_nonfinite_normalization_rejected(self):
        model = model_fixture()
        payload = dict(schema="q1_upper_linear_model_v1", channels=list(CHANNELS),
            history_frames=25, future_frames=10, plan_identity=dict(sha256=PLAN_SHA),
            selection=dict(best_learned_method="ridge_0.01"),
            model=dict(model, fitted_split="train", training_windows=8192, lambda_mean_loss=.01))
        checked_model(payload)
        invalid = deepcopy(payload)
        invalid["model"]["fitted_split"] = "test"
        with self.assertRaises(ValueError):
            checked_model(invalid)
        invalid = deepcopy(payload)
        invalid["model"]["scale"][0] = 0.
        with self.assertRaises(ValueError):
            checked_model(invalid)


if __name__ == "__main__":
    unittest.main()
