"""Check plane point Jacobians and prospective corner protection."""

import unittest

import numpy as np

from script.a3 import test_smpl_checked_integration as checked_tests
from script.a3.smpl_vertex_floor import FootEnvelopeLimit


class FootEnvelopeChecks(unittest.TestCase):
    def test_height_jacobian_matches_finite_difference(self):
        c = checked_tests.CheckedIntegrationTests().configuration(.2)
        c.update(np.array([.2]))
        limit = FootEnvelopeLimit(c.model, c.foot_geometries)
        expected = -limit.compute_qp_inequalities(c, 1 / 240).G[:, 0]
        geom, corners = limit.envelopes[0]
        def height():
            return (corners @ c.data.geom_xmat[geom].reshape(3, 3).T + c.data.geom_xpos[geom])[:, 2]
        c.update(np.array([.2 + 1e-6]))
        plus = height()
        c.update(np.array([.2 - 1e-6]))
        minus = height()
        np.testing.assert_allclose(expected, (plus - minus) / 2e-6, atol=1e-9)

    def test_next_corner_can_reject_rotation_before_integration(self):
        c = checked_tests.CheckedIntegrationTests().configuration(.021)
        limit = FootEnvelopeLimit(c.model, c.foot_geometries)
        constraint = limit.compute_qp_inequalities(c, 1 / 240)
        self.assertFalse(np.all(constraint.G @ np.array([.1]) <= constraint.h))
        self.assertTrue(np.all(constraint.G @ np.zeros(1) <= constraint.h + 1e-12))


if __name__ == '__main__':
    unittest.main()
