"""Check body parsing, rotation correction, SMPL composition and interpolation."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from egolocate_input import SMPL_TO_ROBOT_BASIS, repair_local_rotations, resample, save_test_only
from inspect_egolocate import body_record, global_rotations, SMPL_PARENTS
from q1_retarget import BodyReference


class RecordedBodyTests(unittest.TestCase):
    def test_nonfinite_and_wrong_smpl_schema_are_rejected(self):
        payload = {"type": "pose", "source": "egolocate_fgp", "parents": SMPL_PARENTS.tolist(),
                   "pose_matrix": np.tile(np.eye(3).reshape(9), (24, 1)).tolist(),
                   "fused_pose_matrix": np.tile(np.eye(3).reshape(9), (24, 1)).tolist(),
                   "axis_angle": [0.] * 72, "trans": [0.] * 3, "joints": np.zeros((24, 3)).tolist(),
                   "frame_index": 0, "time": 1., "fps": 30.}
        record = {"host_monotonic_s": 1., "elapsed_s": 0., "payload": payload}
        self.assertEqual(body_record(record)["pose_matrix"].shape, (24, 9))
        payload["trans"][0] = float("nan")
        with self.assertRaises(ValueError): body_record(record)
        payload["trans"][0] = 0
        payload["parents"][3] = 2
        with self.assertRaises(ValueError): body_record(record)

    def test_small_matrix_error_is_recorded_and_reflection_or_large_error_rejected(self):
        raw = np.tile(np.eye(3), (2, 24, 1, 1))
        raw[:, 0, 0, 0] *= 1.0005
        repaired, details = repair_local_rotations(raw)
        np.testing.assert_allclose(repaired.transpose(0, 1, 3, 2) @ repaired, np.tile(np.eye(3), (2, 24, 1, 1)), atol=1e-10)
        self.assertGreater(details["max_matrix_repair_frobenius_change"], 0)
        for matrix in (np.diag([-1., 1, 1]), np.eye(3) * 1.1):
            raw[0, 0] = matrix
            with self.assertRaises(ValueError): repair_local_rotations(raw)

    def test_parent_rotation_is_composed_before_child(self):
        local = np.tile(np.eye(3), (1, 24, 1, 1))
        local[:, 0] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        local[:, 1] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
        world = global_rotations(local)
        expected = (Rotation.from_euler("y", 90, degrees=True) * Rotation.from_euler("x", 45, degrees=True)).as_matrix()
        np.testing.assert_allclose(world[0, 1], expected, atol=1e-12)
        np.testing.assert_allclose(world[0, 4], expected, atol=1e-12)

    def test_basis_preserves_handedness_and_turns_smpl_up_into_robot_up(self):
        basis = SMPL_TO_ROBOT_BASIS
        self.assertEqual(np.linalg.det(basis), 1)
        np.testing.assert_array_equal(basis @ [0, 1, 0], [0, 0, 1])
        np.testing.assert_array_equal(basis @ [0, 0, 1], [1, 0, 0])
        old = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        new = Rotation.from_matrix(basis @ old @ basis.T)
        np.testing.assert_allclose(new.as_matrix(), Rotation.from_euler("z", 90, degrees=True).as_matrix(), atol=1e-12)

    def test_irregular_timestamp_slerp_does_not_extrapolate_or_bridge_long_gaps(self):
        times = [100, 100.03, 100.07]
        positions = np.asarray([[[0, 0, 0]], [[0.03, 0, 0]], [[0.07, 0, 0]]])
        rotations = Rotation.from_euler("z", [0, 0.3, 0.7]).as_matrix()[:, None]
        output_times, points, q = resample(times, positions, rotations)
        np.testing.assert_allclose(output_times, [0, 0.02, 0.04, 0.06])
        np.testing.assert_allclose(points[:, 0, 0], output_times)
        np.testing.assert_allclose(Rotation.from_quat(q[:, 0], scalar_first=True).as_rotvec()[:, 2], output_times * 10, atol=1e-10)
        with self.assertRaises(ValueError): resample([0, 0.03, 1.], positions, rotations)
        with self.assertRaises(ValueError): resample([0, 0.03, 0.03], positions, rotations)

    def test_test_only_flags_survive_export(self):
        source = BodyReference([0, 0.02], ["pelvis"], np.zeros((2, 1, 3)),
                               np.tile([1., 0, 0, 0], (2, 1, 1)), np.zeros((1, 3)),
                               [[1, 0, 0, 0]], ["pelvis"], "TEST_ONLY")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "body.npz"
            save_test_only(source, path)
            with np.load(path, allow_pickle=False) as output:
                self.assertEqual(output["usage"].item(), "TEST_ONLY")
                self.assertIs(output["training_allowed"].item(), False)
            loaded = BodyReference.load(path, ["pelvis"])
            np.testing.assert_array_equal(loaded.positions, source.positions)


if __name__ == "__main__":
    unittest.main()
