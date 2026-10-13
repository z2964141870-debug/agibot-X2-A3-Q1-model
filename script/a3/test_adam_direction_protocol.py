"""Analytic controls for the unique non-updating Adam candidate."""

import math
import unittest
from script.a3.adam_direction_protocol import Budget, candidate, clip_factor, require_group_ids


class AdamDirectionTests(unittest.TestCase):
    def test_first_step_bias_correction_and_decay(self):
        value, m, v = candidate(2., 3., 0., 0., 0, .01, .9, .999, 1e-8, .1)
        self.assertAlmostEqual(value, 2 * .999 - .01 * 3 / (3 + 1e-8), places=13)
        self.assertAlmostEqual(m, .3)
        self.assertAlmostEqual(v, .009)

    def test_saved_momentum_can_reverse_current_gradient(self):
        value, m, v = candidate(1., 1., -2., 4., 16, .01, .9, .999, 1e-8, 0.)
        self.assertLess(m, 0)
        self.assertGreater(value, 1.)
        self.assertGreater(v, 0)

    def test_global_clip_uses_actor_and_critic_norm(self):
        self.assertAlmostEqual(clip_factor(math.hypot(3., 4.)), .1 / (5. + 1e-6))
        self.assertEqual(clip_factor(0), 1.)
        with self.assertRaises(ValueError):
            clip_factor(float('nan'))

    def test_ids_and_budget_reject_before_work(self):
        require_group_ids([3, 4], 2, 3)
        with self.assertRaises(ValueError):
            require_group_ids([4, 3], 2, 3)
        budget = Budget()
        for name, count in budget.LIMITS.items():
            for _ in range(count):
                budget.take(name)
            with self.assertRaises(RuntimeError):
                budget.take(name)
            self.assertEqual(budget.counts[name], count)


if __name__ == '__main__':
    unittest.main()
