import unittest

import numpy as np

from analyze_phuma_support import pose_min_z


class RigidFootSupportTests(unittest.TestCase):
    def geom(self, kind, position=(0., 0., 0.), size=(0., 0., 0.), **extra):
        return {"kind": kind, "position": np.array(position), "rotation": np.eye(3),
                "size": np.array(size), **extra}

    def test_translated_sphere_rotates_with_body(self):
        rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        geom = self.geom("sphere", position=(.1, .2, 0.), size=(.05, 0., 0.))
        self.assertAlmostEqual(pose_min_z([geom], np.array([0., 0., 1.]), rotation), 1.15)

    def test_box_extent_uses_world_vertical_after_rotation(self):
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        geom = self.geom("box", size=(.1, .2, .3))
        self.assertAlmostEqual(pose_min_z([geom], np.array([0., 0., 1.]), rotation), .9)

    def test_mesh_vertices_and_multiple_geometries(self):
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        geom = self.geom("mesh", vertices=np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., -.4]]))
        sphere = self.geom("sphere", size=(.05, 0., 0.))
        self.assertAlmostEqual(pose_min_z([sphere, geom], np.array([0., 0., 1.]), rotation), 0.)

    def test_unknown_or_absent_geometry_is_rejected(self):
        for geometries in ([], [self.geom("unreviewed")]):
            with self.assertRaises(ValueError):
                pose_min_z(geometries, np.zeros(3), np.eye(3))


if __name__ == "__main__":
    unittest.main()
