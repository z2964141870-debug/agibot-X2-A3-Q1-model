import tempfile
from pathlib import Path
import unittest

import numpy as np

from inventory_amass import inspect_motion
from select_amass_pilot import check_window


class AmassPilotTests(unittest.TestCase):
    def fixture(self, folder, fps=120., betas=16, speed=.3, field_mismatch=False):
        path = Path(folder) / "ACCAD/subject/walk_stageii.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        pose = np.zeros((481, 165))
        trans = np.zeros((481, 3))
        trans[:, 0] = np.arange(481) / 120. * speed
        fields = {"poses": pose, "trans": trans, "betas": np.zeros(betas),
                  "mocap_frame_rate": fps, "gender": "male", "surface_model_type": "smplx"}
        for key, start, stop in (("root_orient", 0, 3), ("pose_body", 3, 66),
                                 ("pose_jaw", 66, 69), ("pose_eye", 69, 75), ("pose_hand", 75, 165)):
            fields[key] = pose[:, start:stop].copy()
        if field_mismatch:
            fields["pose_body"][240, 2] = .1
        np.savez_compressed(path, **fields)
        return path, inspect_motion(path, Path(folder))

    def test_four_seconds_keep_native_time_and_sixteen_betas(self):
        with tempfile.TemporaryDirectory() as folder:
            path, row = self.fixture(folder)
            result = check_window(path, row)
            self.assertEqual(result["source_stop_frame_exclusive"], 481)
            self.assertEqual(result["source_window_duration_s"], 4.)
            self.assertEqual(result["planned_target_frames"], 201)
            self.assertEqual(result["num_betas"], 16)
            self.assertAlmostEqual(result["root_statistics"]["root_max_horizontal_displacement_m"], 1.2)
            self.assertTrue(result["raw_window_candidate"])

    def test_rate_shape_and_pose_field_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for options in ({"fps": 30.}, {"betas": 10}, {"field_mismatch": True}):
                with self.subTest(options=options):
                    path, row = self.fixture(folder, **options)
                    with self.assertRaises(ValueError):
                        check_window(path, row)

    def test_static_or_fast_translation_is_not_a_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            for speed in (0., 2.):
                path, row = self.fixture(folder, speed=speed)
                self.assertFalse(check_window(path, row)["raw_window_candidate"])

    def test_changed_inventory_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, row = self.fixture(folder)
            row["source_frames"] += 1
            with self.assertRaises(ValueError):
                check_window(path, row)


if __name__ == "__main__":
    unittest.main()
