"""Qualify actual-vertex support and the rotational displacement Jacobian."""

import unittest

import numpy as np

from script.a3 import test_smpl_checked_integration as checked_tests
from script.a3.smpl_hull_floor import FootHullLimit, support_vertices


class FootHullChecks(unittest.TestCase):
    def test_support_values_preserved_without_empty_corners(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [.1, .1, .1], [.2, .2, .2], [0, 0, 0]], dtype=float)
        hull, method = support_vertices(points)
        self.assertEqual(method, 'SCIPY_CONVEX_HULL_ACTUAL_VERTICES')
        self.assertEqual(len(hull), 4)
        self.assertTrue(all(np.any(np.all(points == vertex, axis=1)) for vertex in hull))
        directions = np.vstack([np.eye(3), -np.eye(3), np.ones((1, 3)),
            -np.ones((1, 3)), np.random.default_rng(23).normal(size=(512, 3))])
        np.testing.assert_allclose((hull @ directions.T).min(0),
            (points @ directions.T).min(0), atol=1e-12, rtol=0)
        self.assertFalse(np.any(np.all(hull == [1, 1, 1], axis=1)))

    def test_degenerate_geometry_keeps_actual_vertices(self):
        for points in (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                np.array([[0, 0, 0], [1, 0, 0]]), np.array([[0, 0, 0]])):
            hull, method = support_vertices(points)
            self.assertEqual(method, 'ALL_ACTUAL_VERTICES_DEGENERATE')
            np.testing.assert_array_equal(hull, np.unique(points, axis=0))
        for points in (np.empty((0, 3)), [[0, 0, float('nan')]], [[0, 0]]):
            with self.assertRaises(ValueError):
                support_vertices(points)

    def test_actual_vertex_height_jacobian(self):
        c = checked_tests.CheckedIntegrationTests().configuration(.2)
        c.update(np.array([.2]))
        limit = FootHullLimit(c.model, c.foot_geometries)
        expected = -limit.compute_qp_inequalities(c, 1 / 240).G[:, 0]
        geom, vertices = limit.envelopes[0]
        def height():
            return (vertices @ c.data.geom_xmat[geom].reshape(3, 3).T + c.data.geom_xpos[geom])[:, 2]
        c.update(np.array([.2 + 1e-6]))
        plus = height()
        c.update(np.array([.2 - 1e-6]))
        minus = height()
        np.testing.assert_allclose(expected, (plus - minus) / 2e-6, atol=1e-9, rtol=0)


if __name__ == '__main__':
    unittest.main()
