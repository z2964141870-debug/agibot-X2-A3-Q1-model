"""Stored Gaussian replay must distinguish equality from a changed mean."""

import unittest

import torch

from script.a3.likelihood_probe import compare


class LikelihoodProbeTests(unittest.TestCase):
    def test_exact_stored_distribution_has_unit_ratio(self):
        mean = torch.zeros(2, 3, 29)
        std = torch.ones(29) * .3
        actions = torch.ones_like(mean) * .1
        old = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
        result = compare(mean, mean, std, std, actions, old)
        self.assertEqual(result["action_rmse"], 0)
        self.assertEqual(result["max_abs_ratio_minus_one"], 0)

    def test_changed_mean_does_not_pass_unit_ratio(self):
        mean = torch.zeros(2, 3, 29)
        std = torch.ones(29) * .3
        actions = torch.ones_like(mean) * .1
        old = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
        result = compare(mean+.1, mean, std, std, actions, old)
        self.assertGreater(result["max_abs_ratio_minus_one"], .1)
        self.assertEqual(result["changed_frames_over_1e_4"], 6)


if __name__ == "__main__":
    unittest.main()
