"""Reject incomplete or duplicated initial probability coverage."""

import unittest

import torch

from script.a3.zhy_precision_trainer import validate_initial_batches


class InitialCoverageTests(unittest.TestCase):
    def setUp(self):
        mean = torch.zeros(16, 24, 29)
        std = torch.ones_like(mean)
        self.source = dict(old_mu_batch=mean, old_sigma_batch=std, actions=mean,
            logprobs=torch.distributions.Normal(mean, std).log_prob(mean).sum(-1))
        self.records = [dict(indices=torch.arange(i, i+4), mean=mean[i:i+4], std=std[i:i+4])
                        for i in range(0, 16, 4)]

    def test_all_environments_recompute_exact_probabilities(self):
        rows = validate_initial_batches(self.records, self.source)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['max_abs_ratio_minus_one'] == 0 for r in rows))

    def test_duplicate_environment_rejects_coverage(self):
        self.records[-1]['indices'] = self.records[0]['indices']
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            validate_initial_batches(self.records, self.source)


if __name__ == '__main__':
    unittest.main()
