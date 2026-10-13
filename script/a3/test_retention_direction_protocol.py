"""Controls for causal sign, degenerate gradients and immutable probe budgets."""

import unittest
from script.a3.retention_direction_protocol import Budget, direction


class DirectionTests(unittest.TestCase):
    def test_descent_direction_sign(self):
        self.assertEqual(direction(-2., 2., 1.)['interpretation'], 'LOCAL_TEACHER_INCREASE')
        self.assertEqual(direction(2., 2., 1.)['interpretation'], 'LOCAL_TEACHER_DECREASE')

    def test_zero_gradient_has_no_fabricated_cosine(self):
        self.assertIsNone(direction(0., 0., 1.)['cosine'])
        for args in ((1., 0., 1.), (float('nan'), 1., 1.), (2., 1., 1.)):
            with self.assertRaises(ValueError):
                direction(*args)

    def test_budget_rejects_before_extra_work(self):
        budget = Budget()
        for name, limit in budget.LIMITS.items():
            for _ in range(limit):
                budget.take(name)
            with self.assertRaises(RuntimeError):
                budget.take(name)
            self.assertEqual(budget.counts[name], limit)


if __name__ == '__main__':
    unittest.main()
