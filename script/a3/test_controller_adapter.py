"""CV initialization and finite residual input-contract checks."""

import unittest

import numpy as np

from script.a3.controller_adapter_trial import time_matrix
from script.a3.joint_forecast_replay import forecast_window


class AdapterTests(unittest.TestCase):
    def test_time_matrix_matches_linear_interpolation(self):
        support = np.concatenate(([-1/30], np.arange(11)/50))
        for phase in (0., -1/150, -1/75):
            values = support[:, None] * np.arange(1, 30)[None]
            expected = (phase+np.arange(11)/50)[:, None]*np.arange(1, 30)[None]
            np.testing.assert_allclose(time_matrix(phase) @ values, expected, atol=1e-12)

    def test_zero_residual_is_exact_cv_and_nonzero_is_causal(self):
        model = dict(mean=np.zeros(145), std=np.ones(145), weights=np.zeros((145, 290)), bias=np.zeros(290))
        times = np.arange(8)/30
        q = times[:, None] * np.arange(1, 30)[None]
        query = times[-1]-.0133333333333333+np.arange(11)/50
        np.testing.assert_array_equal(forecast_window(q, times, query, "cv", None),
                                      forecast_window(q, times, query, "cv_residual", model))
        model["bias"][:] = .001
        result = forecast_window(q, times, query, "cv_residual", model)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(np.abs(result-forecast_window(q, times, query, "cv", None)).max(), 0)


if __name__ == "__main__":
    unittest.main()
