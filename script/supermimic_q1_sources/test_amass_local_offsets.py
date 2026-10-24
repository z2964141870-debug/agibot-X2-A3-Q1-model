import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "q1"))
from amass_local_offsets import independent_local_offsets, local_offset_targets


@unittest.skipUnless(importlib.util.find_spec("scipy") and importlib.util.find_spec("mink"),
                     "Uses the server's existing SciPy and Mink")
class LocalOffsetsTests(unittest.TestCase):
    def fixture(self):
        from scipy.spatial.transform import Rotation

        calibration = np.array([[0., 0., 1.], [.25, 0., 1.]])
        positions = np.stack([calibration, [[0., 0., 1.], [0., .25, 1.]]])
        orient = np.array([[1., 0., 0., 0.], Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True)])
        quats = np.tile([1., 0., 0., 0.], (2, 2, 1))
        quats[:, 1] = orient
        body = SimpleNamespace(names=["pelvis", "wrist"], times=np.array([0., .02]),
                               positions=positions, orientations=quats, calibration_positions=calibration,
                               calibration_orientations=quats[0])
        robot = np.array([[0., 0., .5], [.13, 0., .5]])
        return body, robot, np.tile(np.eye(3), (2, 1, 1))

    def test_body_local_offset_rotates_with_a_rigid_source_segment(self):
        body, robot, rotation = self.fixture()
        points, _, roots = local_offset_targets(body, robot, rotation, 1.)
        np.testing.assert_allclose(points[0], robot, atol=1e-14)
        np.testing.assert_allclose(points[1, 1], [0., .13, .5], atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(points[:, 1] - roots, axis=-1), .13, atol=1e-14)

    def test_root_translation_stays_in_the_world_horizontal_plane(self):
        body, robot, rotation = self.fixture()
        body.positions[1] += [.2, 0., 0.]
        _, _, roots = local_offset_targets(body, robot, rotation, .5)
        np.testing.assert_allclose(roots, [[0., 0., .5], [.1, 0., .5]], atol=1e-14)

    def test_independent_scalar_mapping_agrees_under_rotated_frames(self):
        from scipy.spatial.transform import Rotation

        body, robot, rotation = self.fixture()
        transform = Rotation.from_euler("z", 35, degrees=True)
        body.positions = body.positions @ transform.as_matrix().T
        body.calibration_positions = body.calibration_positions @ transform.as_matrix().T
        body.orientations = (transform * Rotation.from_quat(body.orientations.reshape(-1, 4), scalar_first=True)).as_quat(scalar_first=True).reshape(2, 2, 4)
        body.calibration_orientations = body.orientations[0].copy()
        vectorized = local_offset_targets(body, robot, rotation, .7)
        scalar = independent_local_offsets(body, robot, rotation, .7)
        for expected, actual in zip(vectorized, scalar):
            np.testing.assert_allclose(actual, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
