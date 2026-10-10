"""Causality and interpolation contracts for reference forecasting."""

import unittest

import numpy as np

from script.a3.future_reference_probe import windows


class FutureReferenceTests(unittest.TestCase):
    def test_constant_velocity_labels_and_extrapolation_agree(self):
        q = np.arange(40)[:, None] / 30 * np.array([1., -2.])[None]
        _, y, cv, _ = windows(q)
        np.testing.assert_allclose(y, cv, atol=1e-14)

    def test_future_changes_labels_but_not_current_causal_input(self):
        q = np.arange(40)[:, None] / 30
        x, y, _, index = windows(q)
        changed = q.copy()
        changed[index[0]+1:] += 10
        new_x, new_y, _, _ = windows(changed)
        np.testing.assert_array_equal(x[0], new_x[0])
        self.assertGreater(np.max(np.abs(y[0]-new_y[0])), 1)

    def test_nonfinite_source_rejected(self):
        with self.assertRaises(ValueError):
            windows(np.full((40, 29), np.nan))


if __name__ == "__main__":
    unittest.main()
