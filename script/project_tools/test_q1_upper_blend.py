"""Check blend endpoints and physical-angle semantics before simulation."""

import unittest

import numpy as np

from q1_upper_blend import blended_upper_actions
from q1_upper_reference import reference_centered_actions


class UpperBlendTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(28)
        self.actions = rng.normal(size=(7, 22)).astype(np.float32)
        self.reference = rng.normal(size=(7, 22)).astype(np.float32)
        self.base = rng.normal(size=(7, 22)).astype(np.float32)
        self.scale = np.linspace(.2, 1.3, 22, dtype=np.float32)

    def test_zero_exact_and_no_mutation(self):
        original = self.actions.copy()
        actual = blended_upper_actions(self.actions, self.reference, self.base, self.scale, 0.)
        np.testing.assert_array_equal(actual, original)
        actual[:] = 0
        np.testing.assert_array_equal(self.actions, original)

    def test_reference_endpoint_matches_previous_controller(self):
        actual = blended_upper_actions(self.actions, self.reference, self.base, self.scale, 1.)
        expected = reference_centered_actions(self.actions, self.reference, self.base, self.scale, 0.)
        np.testing.assert_array_equal(actual, expected)

    def test_intermediate_physical_targets_and_lower_actions(self):
        original = self.actions.copy()
        target = self.base + self.scale * original
        for alpha in (.25, .5, .75):
            actual = blended_upper_actions(original, self.reference, self.base, self.scale, alpha)
            np.testing.assert_array_equal(actual[:, :14], original[:, :14])
            np.testing.assert_allclose((self.base + self.scale * actual)[:, 14:],
                ((1-alpha) * target + alpha * self.reference)[:, 14:], atol=5e-7)
        np.testing.assert_array_equal(original, self.actions)

    def test_invalid_inputs_rejected(self):
        for alpha in (-.01, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                blended_upper_actions(self.actions, self.reference, self.base, self.scale, alpha)
        with self.assertRaises(ValueError):
            blended_upper_actions(self.actions, self.reference[:, :8], self.base, self.scale, .5)


if __name__ == "__main__":
    unittest.main()
