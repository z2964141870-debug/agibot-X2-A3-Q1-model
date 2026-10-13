"""Independent low-dimensional counterexamples and bounded E49 control checks."""

import math
import unittest

from script.a3.candidate_retention_protocol import Budget, TOLERANCES, acceptance, match_control, projection_scale


class ProjectionContract(unittest.TestCase):
    def test_outward_component_removed_and_inward_untouched(self):
        g, delta = [3., 4.], [2., 1.]
        alpha = projection_scale(sum(a*b for a,b in zip(g, delta)), 25.)
        projected = [d-alpha*a for a,d in zip(g, delta)]
        self.assertAlmostEqual(sum(a*b for a,b in zip(g, projected)), 0.)
        self.assertEqual(projection_scale(-2., 25.), 0.)

    def test_nonlinear_counterexample_is_rejected(self):
        # f(x,y) = x + y*y at (0,0). Tangent (0,1) has g.delta=0,
        # but its finite increase is positive even though the original guard passes.
        result = acceptance(0., .00001, -.01, .047, .24, TOLERANCES)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['rejection_reasons'], ['finite_teacher'])

    def test_rounded_direction_and_clothing_objective_are_independent_gates(self):
        self.assertFalse(acceptance(2e-8, -1e-5, -.01, .04, .2, TOLERANCES)['accepted'])
        self.assertFalse(acceptance(-1e-8, -1e-5, 2e-7, .04, .2, TOLERANCES)['accepted'])
        self.assertTrue(acceptance(0., -1e-5, -.01, .04, .2, TOLERANCES)['accepted'])

    def test_invalid_geometry_and_control_mismatch_stop(self):
        for dot, squared in [(1.,0.), (1.,-1.), (math.nan,1.), (1.,math.inf)]:
            with self.assertRaises(ValueError):
                projection_scale(dot, squared)
        with self.assertRaises(ValueError):
            match_control(-6.25e-5, 6.25e-5, TOLERANCES, True)

    def test_no_second_projected_candidate(self):
        budget = Budget()
        budget.take('projected_candidates')
        with self.assertRaises(RuntimeError):
            budget.take('projected_candidates')


if __name__ == '__main__':
    unittest.main()
