"""Controls for the PHUMA time, quaternion and spatial export contracts."""

import unittest
from pathlib import Path
import os

import mujoco
import numpy as np

from retarget import ContactCorrectedAssets, grounded_solver, initialize_first_pose, resample_qpos, source_reference


class ResampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vendor = Path(os.environ["PHUMA_VENDOR"])
        cls.model = mujoco.MjModel.from_xml_path(str(vendor / "asset/humanoid_model/g1/custom.xml"))

    def fixture(self):
        t = np.arange(61) / 30.
        return {"root_trans": np.column_stack([10 + .3 * t, -2 + .1 * t, .79 + 0 * t]),
                "root_ori": np.tile([0., 0., 0., 1.], (61, 1)),
                "dof_pos_23": np.zeros((61, 23)), "fps": 30.}

    def test_native_duration_and_root_displacement(self):
        times, qpos = resample_qpos(self.model, self.fixture())
        self.assertEqual(len(times), 101)
        self.assertEqual(times[-1], 2.)
        np.testing.assert_allclose(qpos[-1, :3], [.6, .2, .79], atol=1e-14)
        np.testing.assert_allclose(np.diff(times), .02, atol=1e-14)

    def test_antipodal_quaternions_and_explicit_column_order(self):
        motion = self.fixture()
        motion["root_ori"][1::2] *= -1
        motion["dof_pos_23"][:] = np.arange(23) / 100.
        _, qpos = resample_qpos(self.model, motion)
        np.testing.assert_allclose(np.abs(qpos[:, 3]), 1., atol=1e-14)
        np.testing.assert_allclose(qpos[:, 4:7], 0., atol=1e-14)
        np.testing.assert_allclose(qpos[:, 7:], np.tile(np.arange(23) / 100., (101, 1)))

    def test_other_robot_layout_rejected(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
        with self.assertRaises(ValueError):
            resample_qpos(model, self.fixture())

    def test_initial_crouch_fitted_before_temporal_speed_limits(self):
        assets = ContactCorrectedAssets(Path(os.environ["PHUMA_VENDOR"]).resolve().parents[2])
        solver, _ = grounded_solver(assets)
        motion = self.fixture()
        motion["root_trans"][:, 2] = .68
        motion["dof_pos_23"][:, [0, 6]] = -.35
        motion["dof_pos_23"][:, [3, 9]] = .7
        motion["dof_pos_23"][:, [4, 10]] = -.35
        reference, _ = source_reference(self.model, motion, solver.names)
        result = initialize_first_pose(solver, reference, .50818)
        self.assertEqual(result["added_reference_frames"], 0)
        self.assertGreater(np.abs(solver.sim.data.qpos[7:] - assets.q0).max(), .1)
        np.testing.assert_allclose(solver.sim.data.qpos[:2], [0., 0.], atol=1e-8)

    def test_geometry_and_body_velocity_do_not_require_constraint_acceleration(self):
        assets = ContactCorrectedAssets(Path(os.environ["PHUMA_VENDOR"]).resolve().parents[2])
        rng = np.random.default_rng(2026102407)
        for _ in range(8):
            full, kinematic = mujoco.MjData(assets.model), mujoco.MjData(assets.model)
            full.qpos[7:] = assets.q0 + rng.normal(0, .03, 22)
            full.qvel[:] = rng.normal(0, .1, 28)
            kinematic.qpos[:], kinematic.qvel[:] = full.qpos, full.qvel
            mujoco.mj_forward(assets.model, full)
            mujoco.mj_fwdPosition(assets.model, kinematic)
            mujoco.mj_fwdVelocity(assets.model, kinematic)
            for key in ("xpos", "xquat", "geom_xpos", "geom_xmat", "cvel"):
                np.testing.assert_array_equal(getattr(full, key), getattr(kinematic, key))
            self.assertEqual(full.ncon, kinematic.ncon)
            np.testing.assert_array_equal([c.dist for c in full.contact], [c.dist for c in kinematic.contact])


if __name__ == "__main__":
    unittest.main()
