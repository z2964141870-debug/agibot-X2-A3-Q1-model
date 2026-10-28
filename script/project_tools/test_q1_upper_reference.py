"""Check physical PD-target semantics independently of the simulator."""

import unittest

import numpy as np

from q1_upper_reference import reference_centered_actions


class ReferenceActionsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.actions = rng.normal(size=(7, 22)).astype(np.float32)
        self.reference = rng.normal(size=(7, 22)).astype(np.float32)
        self.base = rng.normal(size=(7, 22)).astype(np.float32)
        self.scale = np.linspace(.2, 1.3, 22, dtype=np.float32)

    def test_reference_target_cancels_offsets_and_scales(self):
        before = self.actions.copy()
        result = reference_centered_actions(self.actions, self.reference, self.base,
                                            self.scale, 0.)
        np.testing.assert_array_equal(result[:, :14], self.actions[:, :14])
        np.testing.assert_allclose((self.base + self.scale * result)[:, 14:],
                                   self.reference[:, 14:], atol=4e-7)
        np.testing.assert_array_equal(self.actions, before)

    def test_small_compensation_is_preserved_large_bias_is_bounded(self):
        desired = self.reference.copy()
        desired[:, 14:] += np.linspace(-.5, .5, 8)
        original = (desired - self.base) / self.scale
        before = original.copy()
        result = reference_centered_actions(original, self.reference, self.base,
                                            self.scale, .15)
        residual = (self.base + self.scale * result - self.reference)[:, 14:]
        np.testing.assert_allclose(residual, np.broadcast_to(
            np.linspace(-.5, .5, 8).clip(-.15, .15), residual.shape), atol=4e-7)
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(result[:, :14], before[:, :14])

    def test_invalid_radius_or_shape_is_rejected(self):
        for radius in (-.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                reference_centered_actions(self.actions, self.reference, self.base,
                                            self.scale, radius)
        with self.assertRaises(ValueError):
            reference_centered_actions(self.actions, self.reference[:, :8],
                                        self.base, self.scale, .15)


if __name__ == "__main__":
    unittest.main()
