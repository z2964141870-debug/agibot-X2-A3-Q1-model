"""Analytic collision geometry and optional real-Torch reward/gradient checks."""

import importlib.util
import unittest

import numpy as np

from sole_geometry import FEET, FootSurface, SoleGeometry


def surfaces():
    points = [[-.12, -.05, -.02], [.12, -.05, -.02], [.12, .05, -.02], [-.12, .05, -.02]]
    return SoleGeometry(tuple(FootSurface(name, points, [0.] * 4) for name in FEET))


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.geometry = surfaces()
        self.p = np.array([[[0., .1, .3], [0., -.1, .4]]])
        self.r = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()

    def test_translation_and_batch_dimensions(self):
        np.testing.assert_allclose(self.geometry.heights(self.p, self.r), [[.28, .38]])
        p = np.repeat(self.p[None], 3, axis=0)
        r = np.repeat(self.r[None], 3, axis=0)
        np.testing.assert_allclose(self.geometry.heights(p + [0., 0., .07], r),
                                   np.tile([[[.35, .45]]], (3, 1, 1)))

    def test_tilt_changes_sole_even_when_ankle_z_is_fixed(self):
        angle = np.pi / 6
        c, s = np.cos(angle), np.sin(angle)
        self.r[:, 1] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        expected = .4 - .12 * s - .02 * c
        np.testing.assert_allclose(self.geometry.heights(self.p, self.r), [[.28, expected]])

    def test_sphere_radius_is_subtracted_in_world_z(self):
        geometry = SoleGeometry(tuple(FootSurface(name, [[.1, 0., 0.]], [.03]) for name in FEET))
        self.r[:] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        np.testing.assert_allclose(geometry.heights(self.p, self.r), [[.17, .27]])

    def test_lowest_of_mesh_vertices_and_spheres_is_selected(self):
        geometry = SoleGeometry(tuple(FootSurface(name, [[0., 0., -.02], [0., 0., -.01]],
                                                 [0., .03]) for name in FEET))
        np.testing.assert_allclose(geometry.heights(self.p, self.r), [[.26, .36]])

    def test_rejects_nonfinite_empty_negative_radius_and_foot_order(self):
        for points, radii in (([], []), ([[np.nan, 0., 0.]], [0.]),
                              ([[0., 0., 0.]], [-1.]), ([[0., 0., 0.]], [np.inf]),
                              ([[0., 0., 0.]], [])):
            with self.subTest(points=points, radii=radii), self.assertRaises(ValueError):
                FootSurface(FEET[0], points, radii)
        with self.assertRaises(ValueError):
            SoleGeometry(tuple(reversed(self.geometry.feet)))

    def test_rejects_pose_shape_mismatch_and_nonfinite(self):
        for p, r in ((self.p[:, :1], self.r), (self.p, self.r[0]),
                     (self.p * np.nan, self.r)):
            with self.assertRaises(ValueError):
                self.geometry.heights(p, r)

    def test_geometry_identity_uses_immutable_copies(self):
        source = np.array([[0., 0., -.02]])
        foot = FootSurface(FEET[0], source, [0.])
        before = foot.identity()
        source[:] = 99
        self.assertEqual(foot.identity(), before)
        with self.assertRaises(ValueError):
            foot.points[:] = 99


@unittest.skipUnless(importlib.util.find_spec("torch"), "Real Torch unavailable; tensor/native validation pending")
class TensorTest(unittest.TestCase):
    def test_tensor_matches_numpy_with_body_reordering(self):
        import torch
        from tensor_geometry import TensorSole

        geometry = surfaces()
        names = [FEET[1], "pelvis", FEET[0]]
        p = torch.tensor([[[0., 0., .4], [0., 0., 9.], [0., 0., .3]]], dtype=torch.float64)
        r = torch.eye(3, dtype=torch.float64).repeat(1, 3, 1, 1)
        r[:, 0] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=torch.float64)
        actual = TensorSole(geometry, names, "cpu").double()(p, r)
        expected = geometry.heights(p[:, [2, 0]].numpy(), r[:, [2, 0]].numpy())
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-9)

    def test_pose_gradient_matches_finite_difference(self):
        import torch
        from tensor_geometry import TensorSole

        geometry = surfaces()
        sole = TensorSole(geometry, FEET, "cpu").double()
        p = torch.tensor([[[0., 0., .3], [0., 0., .4]]], dtype=torch.float64, requires_grad=True)
        angle = torch.tensor(.31, dtype=torch.float64, requires_grad=True)

        def value(a):
            c, s, zero, one = a.cos(), a.sin(), a * 0, a * 0 + 1
            rotation = torch.stack((c, zero, s, zero, one, zero, -s, zero, c)).reshape(3, 3)
            return sole(p, rotation.repeat(1, 2, 1, 1)).sum()

        result = value(angle)
        result.backward()
        epsilon = 1e-5
        fd = (value(angle.detach() + epsilon) - value(angle.detach() - epsilon)) / (2 * epsilon)
        torch.testing.assert_close(angle.grad, fd, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(p.grad[..., 2], torch.ones(1, 2, dtype=torch.float64))

    def test_swing_correction_retains_mask_scale_and_non_swing_exactly(self):
        import torch
        from tensor_geometry import swing_z_correction

        ankle = torch.tensor([[.08, .08], [.08, .08]], dtype=torch.float64)
        reference = torch.zeros_like(ankle)
        sole = torch.zeros_like(ankle)
        swing = torch.tensor([[True, False], [False, False]])
        correction = swing_z_correction(ankle, reference, sole, reference, swing)
        torch.testing.assert_close(correction, torch.tensor([.01 * (1 - np.exp(-4)), 0.], dtype=torch.float64))
        torch.testing.assert_close(swing_z_correction(ankle, reference, ankle, reference, swing),
                                   torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
