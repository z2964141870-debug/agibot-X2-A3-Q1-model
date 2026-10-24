import importlib.util
import unittest

from prepare_amass_body import resampled_reference, selected_rows


class AmassBodyTests(unittest.TestCase):
    def selection(self):
        return {"status": "AMASS_BOUNDED_RAW_MOVING_CANDIDATES_SELECTED", "qualified_for_q1_training": False,
                "selected": [{"split": "UNASSIGNED_DIAGNOSTIC_ONLY", "qualified_for_q1_training": False,
                              "raw_window_candidate": True, "source_start_frame": 0,
                              "source_stop_frame_exclusive": 481, "source_hz": 120., "num_betas": 16}]}

    def test_assigned_split_and_changed_native_time_are_rejected(self):
        self.assertEqual(len(selected_rows(self.selection(), 0, 1)), 1)
        for key, value in (("split", "train"), ("source_hz", 30.), ("num_betas", 10),
                           ("source_stop_frame_exclusive", 480), ("qualified_for_q1_training", True)):
            selection = self.selection()
            selection["selected"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                selected_rows(selection, 0, 1)

    @unittest.skipUnless(importlib.util.find_spec("scipy") and importlib.util.find_spec("mujoco"),
                         "Native-time body resampling uses the server's existing SciPy/MuJoCo environment")
    def test_world_z_translation_and_four_seconds_survive_resampling(self):
        import numpy as np

        times = np.arange(481) / 120.
        points = np.zeros((481, 2, 3))
        points[:, :, 0] = .3 * times[:, None]
        points[:, :, 2] = np.array([1., .1])
        rotations = np.broadcast_to(np.eye(3), (481, 2, 3, 3)).copy()
        body = resampled_reference(times, points, rotations, ["pelvis", "left_ankle"])
        np.testing.assert_allclose(body.times, np.arange(201) / 50., atol=1e-12, rtol=0)
        np.testing.assert_allclose(body.positions[:, 0, 0], .3 * body.times, atol=1e-12, rtol=0)
        np.testing.assert_allclose(body.positions[:, 0, 2], 1., atol=1e-12, rtol=0)
        self.assertEqual(body.times[-1], 4.)


if __name__ == "__main__":
    unittest.main()
