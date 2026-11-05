"""Causal baseline and evaluation aggregation checks."""

import unittest

import numpy as np

from forecast_eval import baseline_targets, _new_accumulator, _add, _finish
from forecast_geometry import decode_targets, encode_history
from test_forecast_geometry import motion


class ForecastEvaluationTests(unittest.TestCase):
    def test_hold_baseline_is_exactly_the_visible_pose(self):
        positions, quaternions = motion()
        labels = baseline_targets(positions[:25], quaternions[:25], "hold")
        decoded = decode_targets(labels, encode_history(positions[:25], quaternions[:25]).anchor)
        np.testing.assert_allclose(decoded.positions, np.repeat(positions[24][None], 10, axis=0), atol=3e-15)
        np.testing.assert_allclose(decoded.quaternions, np.repeat(quaternions[24][None], 10, axis=0), atol=3e-15)

    def test_constant_velocity_baseline_reconstructs_linear_motion(self):
        positions = np.zeros((40, 24, 3))
        positions[:, :, 0] = np.arange(40)[:, None] * .02
        positions[:, :, 1] = np.arange(24)[None] * .03
        positions[:, :, 2] = .8
        quaternions = np.zeros((40, 24, 4))
        quaternions[:, :, 0] = 1.
        labels = baseline_targets(positions[:25], quaternions[:25], "cv",
                                  max_linear_speed_m_s=None, max_angular_speed_rad_s=None)
        decoded = decode_targets(labels, encode_history(positions[:25], quaternions[:25]).anchor)
        np.testing.assert_allclose(decoded.positions, positions[25:35], atol=3e-13)
        np.testing.assert_allclose(decoded.quaternions, quaternions[25:35], atol=3e-13)

    def test_aggregation_reports_zero_for_exact_prediction(self):
        positions, quaternions = motion(35)
        prediction = decode_targets(
            __import__("forecast_geometry", fromlist=["encode_targets"]).encode_targets(
                positions[25:35], quaternions[25:35],
                encode_history(positions[:25], quaternions[:25]).anchor),
            encode_history(positions[:25], quaternions[:25]).anchor)
        accumulator = _new_accumulator(10)
        _add(accumulator, prediction, positions[25:35], quaternions[25:35], "fixture")
        result = _finish(accumulator, "fixture", "test")
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["sources"], 1)
        self.assertLess(max(result["position_rmse_m"]), 3e-14)
        self.assertLess(max(result["orientation_geodesic_rmse_rad"]), 3e-14)


if __name__ == "__main__":
    unittest.main()
