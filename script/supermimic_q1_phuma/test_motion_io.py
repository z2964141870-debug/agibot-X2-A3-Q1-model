"""Controls for source decoding, padding, timing and leakage-free chunk grouping."""

import io
import unittest

import numpy as np

from motion_io import decode_g1, member_key, source_group
from scan_corpus import source_split


def fixture():
    return {"root_trans": np.zeros((60, 3)), "root_ori": np.tile([0., 0., 0., 1.], (60, 1)),
            "dof_pos": np.zeros((60, 29)), "fps": 30}


def encode(motion):
    stream = io.BytesIO()
    np.save(stream, motion)
    return stream.getvalue()


class MotionIOTests(unittest.TestCase):
    def test_padded_official_order(self):
        data = fixture()
        data["dof_pos"][:, :19] = np.arange(19)
        data["dof_pos"][:, 22:26] = np.arange(19, 23)
        decoded = decode_g1(encode(data))
        np.testing.assert_array_equal(decoded["dof_pos_23"][0], np.arange(23))

    def test_wrong_fps_and_nonzero_fixed_wrists_rejected(self):
        for field in ("fps", "wrists"):
            data = fixture()
            if field == "fps":
                data["fps"] = 120
            else:
                data["dof_pos"][0, 20] = .01
            with self.assertRaises(ValueError):
                decode_g1(encode(data))

    def test_bad_quaternion_or_mismatched_frames_rejected(self):
        for field in ("root_ori", "root_trans", "unit_quaternion"):
            data = fixture()
            if field == "unit_quaternion":
                data["root_ori"][:] = 0
            else:
                data[field] = np.zeros((1, 4)) if field == "root_ori" else np.zeros((59, 3))
            with self.assertRaises(ValueError):
                decode_g1(encode(data))

    def test_pickle_function_rejected_without_execution(self):
        data = fixture()
        data["fps"] = eval
        with self.assertRaisesRegex(ValueError, "pickle global"):
            decode_g1(encode(data))

    def test_chunks_and_related_clips_share_source_group(self):
        keys = [member_key(f"data/g1/idea400/walking_clip{clip}_chunk_{chunk:04d}.npy")
                for clip, chunk in ((1, 0), (1, 1), (2, 0))]
        self.assertEqual({source_group(key) for key in keys}, {"idea400/walking"})
        self.assertEqual(source_split("idea400/walking", {"train", "test"}), "test")
        self.assertIn(source_split("idea400/walking", {"train"}), ("train", "development"))
        fitness = [f"fitness/subset_0021/Squat_clip_{clip}_chunk_{chunk:04d}"
                   for clip, chunk in ((17, 0), (17, 1), (18, 0))]
        self.assertEqual({source_group(key) for key in fitness}, {"fitness/subset_0021/Squat"})

    def test_external_archive_member_rejected(self):
        for path in ("data/h1_2/foo.npy", "data/g1/../foo.npy", "/data/g1/foo.npy"):
            with self.assertRaises(ValueError):
                member_key(path)


if __name__ == "__main__":
    unittest.main()
