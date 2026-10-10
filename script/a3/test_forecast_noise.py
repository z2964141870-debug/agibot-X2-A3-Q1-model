"""Monte Carlo check of the analytic noisy-input/noisy-origin objective."""

import unittest

import numpy as np

from script.a3.forecast_noise import noise_covariances


class ForecastNoiseTests(unittest.TestCase):
    def test_expected_covariance_matches_independent_samples(self):
        generator = np.random.default_rng(7)
        samples, histories, joints, horizons = 50000, 3, 2, 4
        sigma = .01
        std = np.linspace(.1, .3, histories*joints)
        noise = generator.normal(size=(samples, histories+1, joints))*sigma
        current = noise[:, -1]
        x = (noise[:, :-1]-current[:, None]).reshape(samples, -1)/std
        y = np.tile(-current, (1, horizons))
        covariance, cross = noise_covariances(std, joints, horizons, sigma)
        covariance_se = np.sqrt((np.diag(covariance)[:, None]*np.diag(covariance)[None]
                                  + covariance**2)/samples)
        cross_se = np.sqrt((np.diag(covariance)[:, None]*sigma**2 + cross**2)/samples)
        self.assertTrue(np.all(np.abs(x.T@x/samples-covariance) < 5*covariance_se))
        self.assertTrue(np.all(np.abs(x.T@y/samples-cross) < 5*cross_se))

    def test_zero_noise_preserves_clean_normal_equations(self):
        covariance, cross = noise_covariances(np.ones(6), 2, 4, 0)
        self.assertEqual(np.count_nonzero(covariance), 0)
        self.assertEqual(np.count_nonzero(cross), 0)


if __name__ == "__main__":
    unittest.main()
