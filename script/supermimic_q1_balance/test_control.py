"""Contracts for latent/physical units, fallback state and reward decomposition."""

import unittest

import torch

from control import Fallback, ResidualHead, FEATURES, RADIUS, compose, risk_features, balance_rewards, physical_base


class ControlTests(unittest.TestCase):
    def test_partial_terminal_uses_matching_default_rows(self):
        default, context = torch.randn(4096, 22), torch.randn(4096, 26)
        mask = torch.arange(4096) % 7 != 0
        torch.testing.assert_close(physical_base(default, context[mask], mask),
                                   (default + context[:, 1:23])[mask], rtol=0, atol=0)
        torch.testing.assert_close(physical_base(default[0], context[mask]),
                                   default[0] + context[mask, 1:23], rtol=0, atol=0)

    def test_zero_head_and_latent_units(self):
        head = ResidualHead()
        x = torch.randn(17, FEATURES)
        torch.testing.assert_close(head(x), torch.zeros(17, 14), rtol=0, atol=0)
        nominal, reference = torch.randn(17, 22), torch.randn(17, 22)
        scale = torch.linspace(.1, .6, 22)
        alpha = torch.full((17,), .75)
        action, residual = compose(nominal, reference, head(x), alpha, scale)
        torch.testing.assert_close(action[:, :14], nominal[:, :14], rtol=0, atol=0)
        torch.testing.assert_close(action[:, 14:], .25 * nominal[:, 14:] + .75 * reference[:, 14:])
        latent = torch.full((17, 14), .1)
        action, residual = compose(nominal, reference, latent, alpha, scale)
        torch.testing.assert_close(residual, RADIUS * torch.tanh(latent))
        torch.testing.assert_close((action[:, :14] - nominal[:, :14]) * scale[:14], residual)
        self.assertLess(float(residual.max()), .015)

    def test_gate_hysteresis_and_preview_is_pure(self):
        gate = Fallback(2, [1.] * 4, "cpu")
        cap = torch.full((2,), .75)
        gate.reset(torch.tensor([True, True]), cap)
        before = [v.clone() for v in gate.state_tensors()]
        preview = gate.preview(torch.full((2, 4), 20.), cap)
        for a, b in zip(gate.state_tensors(), before):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        gate.advance(torch.full((2, 4), 20.), cap)
        torch.testing.assert_close(gate.alpha, torch.full((2,), .72))
        for a, b in zip(gate.state_tensors(), preview):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for _ in range(100):
            gate.advance(torch.full((2, 4), 20.), cap)
        self.assertTrue(bool((gate.alpha == 0).all()))
        for _ in range(200):
            old = gate.alpha.clone()
            gate.advance(torch.zeros(2, 4), cap)
            self.assertTrue(bool((gate.alpha - old <= .005001).all()))
        torch.testing.assert_close(gate.alpha, cap)
        gate.reset(torch.tensor([True, False]), torch.zeros(2))
        self.assertEqual(float(gate.alpha[0]), 0.)

    def test_risk_reference_alignment_and_loaded_contact(self):
        s = torch.zeros(2, 57)
        s[:, 3] = 1
        ref = s[:, 3:7].clone()
        risk = risk_features(s, s[:, :3], ref, torch.tensor([[0., 0.], [10., 2.]]),
                             torch.tensor([[100., 100.], [.3, 100.]]))
        torch.testing.assert_close(risk[:, :3], torch.zeros(2, 3))
        torch.testing.assert_close(risk[:, 3], torch.tensor([0., .3]))
        s[:, 0] = .2
        s[:, 10] = 2
        risk = risk_features(s, torch.zeros(2, 3), ref, torch.zeros(2, 2), torch.zeros(2, 2))
        torch.testing.assert_close(risk[:, 0], torch.full((2,), .2))
        torch.testing.assert_close(risk[:, 2], torch.full((2,), 2.))

    def test_reward_units_and_only_head_receives_gradient(self):
        original = torch.tensor([.1, .05])
        state, reference = torch.zeros(2, 57), torch.zeros(2, 22)
        reward, root, leg, cost = balance_rewards(original, state, state[:, :3], reference, torch.zeros(2, 14))
        torch.testing.assert_close(reward, original + .03)
        torch.testing.assert_close(root, torch.ones(2))
        torch.testing.assert_close(leg, torch.ones(2))
        torch.testing.assert_close(cost, torch.zeros(2))
        head = ResidualHead()
        head(torch.randn(2, FEATURES)).sum().backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()))


if __name__ == "__main__":
    unittest.main()
