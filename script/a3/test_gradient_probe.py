"""Contracts for diagnosing gradients without stepping an optimizer."""

import unittest

import torch

from script.a3.gradient_probe import gradient_geometry, parameter_digest, GradientProbeTrainer


class GradientProbeTests(unittest.TestCase):
    def test_conflict_and_disconnected_critic(self):
        named = [("policy.encoder.weight", torch.nn.Parameter(torch.ones(2))),
                 ("value_model.weight", torch.nn.Parameter(torch.ones(1)))]
        gradients = dict(policy=[torch.tensor([1., 2.]), None],
                         aux=[torch.tensor([-2., -4.]), None],
                         entropy=[torch.zeros(2), None], value=[None, torch.ones(1)])
        result = gradient_geometry(named, gradients)
        self.assertAlmostEqual(result["actor_encoder"]["gradient_cosines"]["policy_vs_aux"], -1)
        self.assertEqual(result["critic"]["gradient_norms"]["policy"], 0)
        self.assertIsNone(result["critic"]["gradient_cosines"]["value_vs_policy"])

    def test_parameter_digest_detects_mutation(self):
        parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
        named = [("p", parameter)]
        before = parameter_digest(named)
        with torch.no_grad():
            parameter.add_(0.001)
        self.assertNotEqual(before, parameter_digest(named))

    def test_optimizer_updates_are_forbidden(self):
        trainer = object.__new__(GradientProbeTrainer)
        with self.assertRaisesRegex(RuntimeError, "forbids"):
            trainer._before_optimizer_step(None, (), {})


if __name__ == "__main__":
    unittest.main()
