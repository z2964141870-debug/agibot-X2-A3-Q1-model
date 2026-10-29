"""Counterexamples for command clipping, state timing and lag summaries."""

import unittest
import numpy as np

from saved_control_usage import control_values, evaluation_context, response_lag, target_metrics


class SavedControlTests(unittest.TestCase):
    def test_final_bound_must_not_be_subtracted_as_an_executed_residual(self):
        target, residual = np.zeros((4, 22)), np.full((4, 14), .1)
        target[:, 0] = 20.
        action, boundary, nominal = control_values(target, residual, np.zeros(22), np.ones(22))
        self.assertTrue(boundary[:, 0].all())
        self.assertTrue(np.isnan(nominal[:, 0]).all())
        np.testing.assert_array_equal(nominal[:, 1:], -.1)

    def test_invalid_residual_or_control_reconstruction_is_rejected(self):
        for target, residual, scale in ((np.full((3, 22), 21.), np.zeros((3, 14)), np.ones(22)),
                (np.zeros((3, 22)), np.full((3, 14), .151), np.ones(22)),
                (np.zeros((3, 22)), np.zeros((3, 14)), np.zeros(22)),
                (np.full((3, 22), -19.99), np.full((3, 14), .1), np.ones(22))):
            with self.assertRaises(ValueError):
                control_values(target, residual, np.zeros(22), scale)

    def test_lag_uses_no_wrap_and_constant_signal_has_no_estimate(self):
        values = np.random.default_rng(31).normal(size=100)
        actual = np.r_[np.zeros(3), values[:-3]]
        result = response_lag(values, actual)
        self.assertEqual(result["best_lag_ms"], 60)
        self.assertAlmostEqual(result["best_correlation"], 1.)
        self.assertIsNone(response_lag(np.zeros(20), np.ones(20))["best_lag_ms"])

    def test_pre_and_post_state_have_distinct_metrics(self):
        target, previous = np.full((20, 1), .1), np.zeros((20, 1))
        result = target_metrics(target, target * .2, target, previous,
                                residual=target, nominal=previous)
        self.assertAlmostEqual(result["target_actual_rmse_rad"], .08)
        self.assertAlmostEqual(result["observed_step_rms_rad"], .02)
        self.assertAlmostEqual(result["target_squared_error_reduction_rad2"], .01)
        self.assertEqual(result["residual_toward_reference_fraction"], 1.)

    def test_eval_base_requires_explicit_zero_offset_branch(self):
        manager = "class SuperMimicJointOffsetEvent:\n def __call__(self):\n  if command.eval_mode:\n   command.xi[ids, 1:23] = 0.0\n"
        builder = "def build_env_override():\n motion['eval_mode'] = True\n"
        self.assertTrue(evaluation_context(manager, builder))
        with self.assertRaises(ValueError):
            evaluation_context(manager.replace("= 0.0", "= 0.01"), builder)


if __name__ == "__main__":
    unittest.main()
