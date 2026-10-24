import importlib.util
from types import SimpleNamespace
import unittest

from verify_amass_reference import independent_mapping


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Uses the server's existing SciPy")
class AmassMappingTests(unittest.TestCase):
    def test_horizontal_motion_does_not_become_height_under_tilted_calibration(self):
        import numpy as np
        from scipy.spatial.transform import Rotation

        names = ["pelvis", "left_ankle"]
        calibration = np.array([[0., 0., 1.], [0., .1, .1]])
        rotation = Rotation.from_euler("x", 30, degrees=True)
        body = SimpleNamespace(names=names, times=np.array([0., .02]),
                               calibration_positions=calibration,
                               calibration_orientations=np.tile(rotation.as_quat(scalar_first=True), (2, 1)),
                               positions=np.stack([calibration, calibration + [.2, 0., 0.]]),
                               orientations=np.tile(rotation.as_quat(scalar_first=True), (2, 2, 1)))
        robot = np.array([[0., 0., .5], [0., .05, 0.]])
        robot_rot = np.tile(np.eye(3), (2, 1, 1))
        points, rotations, roots, quats = independent_mapping(body, robot, robot_rot, .5)
        np.testing.assert_allclose(points[0], robot, atol=1e-14)
        np.testing.assert_allclose(points[1], robot + [.1, 0., 0.], atol=1e-14)
        np.testing.assert_allclose(roots, [[0., 0., .5], [.1, 0., .5]], atol=1e-14)
        np.testing.assert_allclose(rotations, np.tile(np.eye(3), (2, 2, 1, 1)), atol=1e-14)
        np.testing.assert_allclose(quats, [[1., 0., 0., 0.]] * 2, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
