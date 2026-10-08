"""Reject ambiguous skeleton data and check root preservation independently of IK."""

from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from q1_retarget import BodyReference, Q1Retargeter


class BodyContractTests(unittest.TestCase):
    def make_reference(self, **overrides):
        values = dict(times=[0, 0.02], names=["pelvis", "spine3"],
                      positions=np.zeros((2, 2, 3)),
                      orientations=np.tile([1., 0, 0, 0], (2, 2, 1)),
                      calibration_positions=np.zeros((2, 3)),
                      calibration_orientations=np.tile([1., 0, 0, 0], (2, 1)),
                      required_names=["pelvis", "spine3"], source_kind="TEST_ONLY")
        values.update(overrides)
        return BodyReference(**values)

    def test_invalid_units_axes_quaternions_and_times_are_rejected(self):
        for overrides in ({"convention": "y_up_mm_xyzw"}, {"times": [0, 0]},
                          {"orientations": np.zeros((2, 2, 4))},
                          {"positions": np.full((2, 2, 3), np.nan)},
                          {"source_kind": ""}, {"names": ["pelvis", "pelvis"]}):
            with self.assertRaises(ValueError):
                self.make_reference(**overrides)

    def test_serialization_requires_no_pickle_and_preserves_calibration(self):
        reference = self.make_reference()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            reference.save(path)
            loaded = BodyReference.load(path, ["spine3", "pelvis"])
        self.assertEqual(loaded.names, ["spine3", "pelvis"])
        np.testing.assert_array_equal(loaded.positions, reference.positions[:, ::-1])
        np.testing.assert_array_equal(loaded.calibration_orientations,
                                      reference.calibration_orientations[::-1])


class RetargetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter = Q1Retargeter()

    def source(self, times=(0, 0.02), displaced=False):
        r = self.retargeter
        positions = np.tile(r.robot_positions, (2, 1, 1))
        orientations = np.tile(Rotation.from_matrix(r.robot_rotations).as_quat(scalar_first=True), (2, 1, 1))
        if displaced:
            rotation = Rotation.from_euler("z", 0.1)
            origin = r.robot_positions[r.root]
            positions[1] = rotation.apply(positions[1] - origin) + origin + [0.03, -0.02, 0]
            orientations[1] = (rotation * Rotation.from_quat(orientations[1], scalar_first=True)).as_quat(scalar_first=True)
        return BodyReference(times, r.names, positions, orientations, r.robot_positions,
                             orientations[0], r.names, "SYNTHETIC_TEST_NOT_SMPL")

    def test_calibration_neutral_is_fixed_point(self):
        r = self.retargeter
        reference, q, _, _, errors = r.retarget(self.source(), 1.0)
        np.testing.assert_allclose(reference.positions, np.tile(r.sim.neutral, (2, 1)), atol=1e-6)
        np.testing.assert_allclose(errors, 0, atol=1e-6)

    def test_global_translation_and_yaw_do_not_turn_into_joint_motion(self):
        r = self.retargeter
        source = self.source(displaced=True)
        reference, q, _, _, errors = r.retarget(source, 1.0)
        np.testing.assert_allclose(q[:, :3], source.positions[:, r.root], atol=1e-7)
        rotations = Rotation.from_quat(q[:, 3:7], scalar_first=True)
        expected = Rotation.from_quat(source.orientations[:, r.root], scalar_first=True)
        np.testing.assert_allclose((rotations * expected.inv()).magnitude(), 0, atol=1e-7)
        np.testing.assert_allclose(reference.positions, np.tile(r.sim.neutral, (2, 1)), atol=1e-5)
        np.testing.assert_allclose(errors, 0, atol=1e-5)

    def test_scale_and_reference_timing_must_be_explicit(self):
        for scale in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                self.retargeter.retarget(self.source(), scale)
        with self.assertRaises(ValueError):
            self.retargeter.retarget(self.source(times=(0, 0.1)), 1.0)

    def test_displacement_scale_preserves_neutral_limbs(self):
        r = self.retargeter
        source = self.source(displaced=True)
        reference, q, _, _, errors = r.retarget(source, 2.0)
        np.testing.assert_allclose(q[1, :3] - q[0, :3], [0.06, -0.04, 0], atol=1e-7)
        np.testing.assert_allclose(reference.positions, np.tile(r.sim.neutral, (2, 1)), atol=1e-5)
        np.testing.assert_allclose(errors, 0, atol=1e-5)

    def test_calibration_heading_aligns_source_world_to_robot(self):
        r = self.retargeter
        source = self.source()
        origin = r.robot_positions[r.root]
        heading = Rotation.from_euler("z", np.pi / 2)
        positions = heading.apply(source.positions.reshape(-1, 3) - origin).reshape(2, -1, 3) + origin
        positions[1] += [0, 0.03, 0]
        quaternions = (heading * Rotation.from_quat(source.orientations.reshape(-1, 4),
                                                   scalar_first=True)).as_quat(scalar_first=True).reshape(2, -1, 4)
        calibrated = BodyReference(source.times, r.names, positions, quaternions,
                                    positions[0], quaternions[0], r.names, "SYNTHETIC_TEST_NOT_SMPL")
        reference, q, _, _, errors = r.retarget(calibrated, 1.0)
        np.testing.assert_allclose(q[1, :3] - q[0, :3], [0.03, 0, 0], atol=1e-7)
        np.testing.assert_allclose(reference.positions, np.tile(r.sim.neutral, (2, 1)), atol=1e-5)
        np.testing.assert_allclose(errors, 0, atol=1e-5)

    def test_large_pose_jump_is_velocity_limited_across_all_iterations(self):
        r = self.retargeter
        source = self.source()
        data = mujoco.MjData(r.sim.model)
        data.qpos[:] = r.sim.data.qpos
        data.qpos[r.sim.q_indices[0]] = 0.8
        data.qpos[r.sim.q_indices[3]] = 1.0
        mujoco.mj_forward(r.sim.model, data)
        source.positions[1] = data.xpos[r.bodies]
        source.orientations[1] = Rotation.from_matrix(
            data.xmat[r.bodies].reshape(-1, 3, 3)).as_quat(scalar_first=True)
        reference, _, _, _, errors = r.retarget(source, 1.0)
        self.assertLessEqual(np.abs(np.diff(reference.positions, axis=0)).max(), 0.04 + 1e-7)
        self.assertGreater(np.linalg.norm(errors[1], axis=-1).max(), 0.01)


if __name__ == "__main__":
    unittest.main()
