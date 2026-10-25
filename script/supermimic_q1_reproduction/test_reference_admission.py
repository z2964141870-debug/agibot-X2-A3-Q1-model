"""Regression checks for the user's removal of local physical admission gates."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from reference_admission import TRACKED, validate_arrays


class AdmissionTests(unittest.TestCase):
    def payload(self):
        count = 5
        qpos = np.zeros((count, 29))
        qpos[:, 2], qpos[:, 3] = 1., 1.
        return {"time_s": np.arange(count) / 50., "fps": np.asarray(50.),
            "body_names": np.asarray(TRACKED), "qpos": qpos, "qvel": np.full((count, 28), 8.),
            "joint_pos": qpos[:, 7:], "joint_vel": np.full((count, 22), 8.),
            "body_pos_w": np.full((count, len(TRACKED), 3), .1),
            "body_quat_w": np.tile([1., 0., 0., 0.], (count, len(TRACKED), 1)),
            "body_lin_vel_w": np.zeros((count, len(TRACKED), 3)),
            "body_ang_vel_w": np.zeros((count, len(TRACKED), 3))}

    def test_readable_fast_or_ungrounded_reference_is_not_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez(path, **self.payload())
            self.assertTrue(validate_arrays(path)["io_valid"])

    def test_nonfinite_training_input_still_rejected(self):
        values = self.payload()
        values["qpos"][0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez(path, **values)
            with self.assertRaises(ValueError):
                validate_arrays(path)


if __name__ == "__main__":
    unittest.main()
