"""Counterexamples for target units, initialization and environment boundaries."""

import unittest

import torch

from target_control import CANDIDATES, LegTargetFilter


class TargetFilterTests(unittest.TestCase):
    def test_baseline_is_bit_exact_even_for_large_targets(self):
        target = torch.arange(24, dtype=torch.float32).reshape(2, 12)/3
        control = LegTargetFilter(2, "baseline", "cpu")
        for value in (target, -target):
            self.assertTrue(torch.equal(control.filter(value), value))

    def test_first_target_has_no_startup_transient(self):
        target = torch.full((2, 12), .6)
        for name in CANDIDATES:
            self.assertTrue(torch.equal(LegTargetFilter(2, name, "cpu").filter(target), target))

    def test_rate_is_in_radians_per_twenty_ms(self):
        control = LegTargetFilter(2, "rate010", "cpu")
        control.filter(torch.zeros(2, 12))
        target = torch.full((2, 12), .6)
        for _ in range(8):
            previous = control.value.clone()
            value = control.filter(target)
            self.assertLessEqual(float((value-previous).abs().max()), .100001)
        torch.testing.assert_close(value, target)

    def test_ema_suppresses_alternating_input_and_has_lag(self):
        control = LegTargetFilter(1, "ema040", "cpu")
        control.filter(torch.zeros(1, 12))
        values = [control.filter(torch.full((1, 12), sign*.2)) for sign in (1., -1.)*10]
        self.assertLess(float(torch.stack(values).abs().max()), .15)
        self.assertGreater(float(values[0].abs().max()), 0)

    def test_reset_prevents_cross_episode_target_memory(self):
        control = LegTargetFilter(2, "rate010", "cpu")
        control.filter(torch.ones(2, 12))
        control.reset(torch.tensor([True, False]))
        value = control.filter(torch.full((2, 12), -1.))
        self.assertTrue(torch.equal(value[0], torch.full((12,), -1.)))
        torch.testing.assert_close(value[1], torch.full((12,), .9))

    def test_filter_never_overshoots_previous_and_requested_target(self):
        for name in CANDIDATES:
            control = LegTargetFilter(2, name, "cpu")
            previous = control.filter(torch.full((2, 12), -.2))
            target = torch.full((2, 12), .3)
            value = control.filter(target)
            self.assertTrue(bool(((value >= previous) & (value <= target)).all()))

    def test_wrong_shape_and_nonfinite_are_rejected(self):
        control = LegTargetFilter(2, "ema020", "cpu")
        with self.assertRaises(ValueError):
            control.filter(torch.zeros(2, 14))
        with self.assertRaises(FloatingPointError):
            control.filter(torch.full((2, 12), float("nan")))


if __name__ == "__main__":
    unittest.main()
