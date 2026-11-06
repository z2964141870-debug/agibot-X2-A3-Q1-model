"""Noise-risk algebra and validation/test separation for the exploratory fit."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from upper_forecast import history_features, predict
from upper_forecast_sensitivity import learned_affine_map
from upper_noise_robust import (PRIOR, STRESS_CASES, TRAIN_NOISE, evaluate, execute,
    expected_noise_profile, fit_noise_ridges, observe, raw_components, specification)


def prior_fixture():
    return dict(weights=np.zeros((201, 80)), mean=np.zeros(200), scale=np.ones(200))


class NoiseRiskTests(unittest.TestCase):
    def test_preregistration_matches_fixed_implementation(self):
        path = Path(__file__).with_name("upper_noise_robust_plan_20261106.json")
        self.assertEqual(json.loads(path.read_text()), specification())

    def test_raw_jacobian_includes_feature_and_output_anchor_noise(self):
        rng = np.random.default_rng(4)
        model = dict(weights=rng.normal(0., .01, (201, 80)),
                     mean=rng.normal(0., .1, 200), scale=rng.uniform(.1, .5, 200))
        components, anchor = raw_components(model["scale"])
        expected, _ = learned_affine_map(model)
        actual = np.einsum("ij,jk->ik", components, model["weights"], optimize=False)+anchor
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        with self.assertRaises(ValueError):
            raw_components(np.zeros(200))

    def test_exact_expected_noise_risk_matches_covariance_quadrature(self):
        rng = np.random.default_rng(5)
        history = rng.normal(0., .1, (1, 25, 8))
        target = rng.normal(0., .1, (1, 10, 8))
        model = dict(weights=rng.normal(0., .01, (201, 80)),
                     mean=np.zeros(200), scale=np.ones(200))
        jacobian, _ = learned_affine_map(model)
        clean = predict(history, "model", {"model": model})-target
        std = .01
        analytic = expected_noise_profile(np.sqrt(np.mean(clean*clean, axis=(0, 2))), jacobian, std)
        basis = np.eye(200)*np.sqrt(200)*std
        errors = []
        for sign in (-1., 1.):
            observed = history+sign*basis.reshape(200, 25, 8)
            errors.append(predict(observed, "model", {"model": model})-target)
        joined = np.concatenate(errors)
        np.testing.assert_allclose(analytic, np.sqrt(np.mean(joined*joined, axis=(0, 2))), atol=1e-15)

    def test_closed_form_fit_satisfies_full_noise_risk_gradient(self):
        rng = np.random.default_rng(6)
        history, target = rng.normal(0., .1, (16, 25, 8)), rng.normal(0., .1, (16, 10, 8))
        models = fit_noise_ridges(history, target, split="train")
        for model in models.values():
            features = history_features(history)
            design = np.column_stack(((features-model["mean"])/model["scale"], np.ones(16)))
            residual = predict(history, "model", {"model": model})-target
            jacobian, _ = learned_affine_map(model)
            components, _ = raw_components(model["scale"])
            gradient = np.einsum("ni,nj->ij", design, residual.reshape(16, 80), optimize=False)/16
            gradient += TRAIN_NOISE**2*np.einsum("ki,kj->ij", components, jacobian, optimize=False)
            penalty = model["weights"].copy()
            penalty[-1] = 0.
            gradient += model["lambda_mean_loss"]*penalty
            np.testing.assert_allclose(gradient, 0., atol=2e-14)
        for split in ("test", "validation"):
            with self.assertRaisesRegex(ValueError, "Only train"):
                fit_noise_ridges(history, target, split=split)

    def test_stress_is_history_only_reproducible_and_nonmutating(self):
        history = np.repeat(np.arange(25)[None, :, None], 8, axis=2).astype(float)
        original = history.copy()
        for case in STRESS_CASES:
            first, second = observe(history, case), observe(history, case)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(history, original)
        frozen = observe(history, "last5_frozen")
        np.testing.assert_array_equal(frozen[:, :20], history[:, :20])
        np.testing.assert_array_equal(frozen[:, 20:], np.full((1, 5, 8), 19.))
        with self.assertRaises(ValueError):
            observe(history, "posthoc")

    def test_validation_choice_persisted_before_test_and_reuse_explicit(self):
        plan = dict(history_frames=25, future_frames=10, clips=[
            dict(relative=split+"/clip.npz", split=split, frames=50)
            for split in ("train", "validation", "test")])

        class Corpus:
            def __init__(self, test_value):
                self.reads, self.test_value = [], test_value

            def load(self, name, split):
                self.reads.append(split)
                values = np.repeat(np.sin(np.arange(50)[:, None]*.1), 8, axis=1)
                if split == "test":
                    values[25:] = self.test_value
                return values

        choices = []
        for value in (-1e3, 1e3):
            corpus, persisted = Corpus(value), []

            def selected(selection, models):
                self.assertNotIn("test", corpus.reads)
                self.assertFalse(selection["test_labels_read"])
                persisted.append(deepcopy(selection))

            result = evaluate(plan, corpus, prior_fixture(), train_limit=8, eval_limit=2,
                              on_selection=selected)
            self.assertEqual(result["selection"], persisted[0])
            self.assertTrue(result["test_reused_from_previous_experiment"])
            self.assertFalse(result["independent_confirmation"])
            self.assertEqual(result["new_model_fits"], 3)
            self.assertIn(PRIOR, result["selection"]["methods"])
            self.assertEqual(len(result["test"]), 7*len(STRESS_CASES))
            self.assertTrue(all(not row["selection_allowed"] for row in result["test"]))
            choices.append(result["selection"])
        self.assertEqual(*choices)

    def test_existing_completed_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/"data/supermimic_for_q1/existing"
            output.mkdir(parents=True)
            marker = output/"result.json"
            marker.write_text("existing")
            with self.assertRaisesRegex(ValueError, "new independent"):
                execute(SimpleNamespace(local_cpu_study=True, project_root=root, output=output))
            self.assertEqual(marker.read_text(), "existing")


if __name__ == "__main__":
    unittest.main()
