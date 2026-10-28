"""Check teacher retention, action routing and synthetic-observation semantics."""

import unittest

import torch
from torch import nn

from q1_reference_balance import ReferenceBalance, lower_gaussian_kl, swap_upper_history


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(80 + 590, 22)
        self.obs_norm = nn.Identity()
        self.goal_norm = nn.Identity()

    def forward(self, history, goal):
        return self.linear(torch.cat((history[:, -1], goal), -1))


class BalanceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.teacher = Teacher()
        self.scale = torch.linspace(.2, 1.3, 22)
        self.model = ReferenceBalance(self.teacher, self.scale)
        self.h, self.g = torch.randn(7, 10, 80), torch.randn(7, 590)
        self.ref, self.base = torch.randn(7, 22), torch.randn(7, 22)

    def test_zero_head_preserves_lower_targets_and_corrects_arms(self):
        self.model.train()
        mean = self.model(self.h, self.g, self.ref, self.base)
        nominal = self.model.teacher_actions(self.h, self.g)
        torch.testing.assert_close(mean[:, :14], nominal[:, :14], atol=0, rtol=0)
        torch.testing.assert_close((self.base + self.scale * mean)[:, 14:],
                                   self.ref[:, 14:], atol=5e-7, rtol=1e-6)
        self.assertFalse(self.teacher.training)
        self.assertTrue(all(not p.requires_grad for p in self.teacher.parameters()))
        torch.testing.assert_close(lower_gaussian_kl(mean, nominal), torch.zeros(7))

    def test_nonzero_head_is_bounded_in_radians_and_cannot_change_arms(self):
        with torch.no_grad():
            self.model.head[-1].bias.copy_(torch.linspace(-100, 100, 14))
        mean = self.model(self.h, self.g, self.ref, self.base)
        nominal = self.model.teacher_actions(self.h, self.g)
        delta = self.scale[:14] * (mean[:, :14] - nominal[:, :14])
        self.assertLessEqual(float(delta.detach().abs().max()), .150001)
        self.assertLess(float(delta[:, 0].detach().max()), -.149)
        self.assertGreater(float(delta[:, -1].detach().min()), .149)
        torch.testing.assert_close((self.base + self.scale * mean)[:, 14:],
                                   self.ref[:, 14:], atol=5e-7, rtol=1e-6)

    def test_only_lower_loss_reaches_head_teacher_stays_fixed(self):
        before = {k: v.clone() for k, v in self.teacher.state_dict().items()}
        actions = self.model(self.h, self.g, self.ref, self.base)
        actions[:, 14:].sum().backward(retain_graph=True)
        self.assertTrue(all(p.grad is None or not bool(p.grad.any())
                            for p in self.model.head.parameters()))
        self.model.zero_grad(set_to_none=True)
        actions[:, :14].sum().backward()
        self.assertGreater(float(self.model.head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        for key, value in self.teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_swap_only_replaces_requested_upper_fields_without_mutation(self):
        before = self.h.clone()
        donor = self.h + 100
        value = swap_upper_history(self.h, donor, ("position", "previous_action"))
        mask = torch.zeros(80, dtype=torch.bool)
        mask[14:22], mask[64:72] = True, True
        torch.testing.assert_close(value[..., mask], donor[..., mask])
        torch.testing.assert_close(value[..., ~mask], self.h[..., ~mask])
        torch.testing.assert_close(self.h, before)

    def test_kl_counts_lower_action_units_and_rejects_invalid_scale(self):
        nominal, shifted = torch.zeros(7, 22), torch.zeros(7, 22)
        shifted[:, :14], shifted[:, 14:] = .1, 200
        torch.testing.assert_close(lower_gaussian_kl(shifted, nominal), torch.full((7,), 7.))
        for invalid in (torch.zeros(22), torch.ones(21), torch.full((22,), float("nan"))):
            with self.assertRaises(ValueError):
                ReferenceBalance(Teacher(), invalid)
        with self.assertRaises(ValueError):
            lower_gaussian_kl(shifted, nominal, 0)


if __name__ == "__main__":
    unittest.main()
