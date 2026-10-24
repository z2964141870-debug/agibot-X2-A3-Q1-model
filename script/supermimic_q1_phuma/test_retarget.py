"""Controls for the PHUMA time, quaternion and spatial export contracts."""

import unittest
from pathlib import Path
import os

import mujoco
import numpy as np

from retarget import resample_qpos


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


if __name__ == "__main__":
    unittest.main()
