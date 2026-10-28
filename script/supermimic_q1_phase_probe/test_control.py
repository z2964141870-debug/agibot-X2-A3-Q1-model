"""Counterexamples for deterministic leg-reference interventions."""

import unittest

import torch

from phase_control import LegReference


class LegReferenceTests(unittest.TestCase):
    def setUp(self):
        self.nominal = torch.zeros(2, 12)
        self.reference = torch.ones(2, 12)
        self.stance = torch.tensor([[True, False], [False, True]])
        self.swing = ~self.stance

    def test_zero_gain_is_exactly_zero(self):
        control = LegReference(2, "baseline", "cpu")
        self.assertTrue(torch.equal(control.correction(self.nominal, self.reference, self.stance, self.swing), self.nominal))

    def test_swing_leaves_stance_leg_unchanged(self):
        control = LegReference(2, "swing050", "cpu")
        value = control.correction(self.nominal, self.reference, self.stance, self.swing)
        self.assertTrue(torch.equal(value[0, :6], torch.zeros(6)))
        self.assertTrue(torch.equal(value[1, 6:], torch.zeros(6)))
        self.assertTrue(bool((value[0, 6:] > 0).all() & (value[1, :6] > 0).all()))

    def test_radius_and_per_step_change_are_bounded(self):
        control = LegReference(2, "uniform025", "cpu")
        previous = torch.zeros(2, 12)
        for sign in (1., -1.):
            for _ in range(100):
                value = control.correction(self.nominal, sign*self.reference*100., self.stance, self.swing)
                self.assertLessEqual(float(value.abs().max()), .150001)
                self.assertLessEqual(float((value-previous).abs().max()), .020001)
                previous = value

    def test_reset_clears_only_selected_environment(self):
        control = LegReference(2, "uniform010", "cpu")
        before = control.correction(self.nominal, self.reference, self.stance, self.swing)
        control.reset(torch.tensor([True, False]))
        self.assertTrue(torch.equal(control.value[0], torch.zeros(12)))
        self.assertTrue(torch.equal(control.value[1], before[1]))

    def test_overlapping_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            LegReference(2, "swing025", "cpu").correction(self.nominal, self.reference, self.stance, self.stance)

    def test_nonfinite_target_is_rejected(self):
        reference = self.reference.clone()
        reference[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            LegReference(2, "uniform010", "cpu").correction(self.nominal, reference, self.stance, self.swing)


if __name__ == "__main__":
    unittest.main()
