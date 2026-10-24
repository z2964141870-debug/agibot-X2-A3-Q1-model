import csv
import io
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bones_g1_csv import JOINT_NAMES_29, decode_bones_g1, resample_native_motion


HEADER = ("Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,"
          "left_hip_pitch_joint_dof,left_hip_roll_joint_dof,left_hip_yaw_joint_dof,left_knee_joint_dof,"
          "left_ankle_pitch_joint_dof,left_ankle_roll_joint_dof,right_hip_pitch_joint_dof,"
          "right_hip_roll_joint_dof,right_hip_yaw_joint_dof,right_knee_joint_dof,"
          "right_ankle_pitch_joint_dof,right_ankle_roll_joint_dof,waist_yaw_joint_dof,"
          "waist_roll_joint_dof,waist_pitch_joint_dof,left_shoulder_pitch_joint_dof,"
          "left_shoulder_roll_joint_dof,left_shoulder_yaw_joint_dof,left_elbow_joint_dof,"
          "left_wrist_roll_joint_dof,left_wrist_pitch_joint_dof,left_wrist_yaw_joint_dof,"
          "right_shoulder_pitch_joint_dof,right_shoulder_roll_joint_dof,right_shoulder_yaw_joint_dof,"
          "right_elbow_joint_dof,right_wrist_roll_joint_dof,right_wrist_pitch_joint_dof,"
          "right_wrist_yaw_joint_dof").split(",")


def encode(rows, header=HEADER):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    for row in rows:
        values = {key: 0. for key in HEADER}
        values.update(row)
        writer.writerow([values.get(key, 0.) for key in header])
    return stream.getvalue().encode("utf-8")


class BonesG1CsvTests(unittest.TestCase):
    def test_units_origin_quaternion_order_and_nonzero_frame_start(self):
        payload = encode([{"Frame": 100 + i, "root_translateX": 123., "root_translateZ": 82.,
                           "root_rotateZ": 90., "left_knee_joint_dof": 30.} for i in range(3)])
        motion = decode_bones_g1(payload, source_hz=120., expected_frames=3)
        np.testing.assert_allclose(motion["time_s"], [0, 1 / 120, 2 / 120])
        np.testing.assert_allclose(motion["qpos_wxyz"][:, :3], [[1.23, 0, .82]] * 3)
        np.testing.assert_allclose(motion["qpos_wxyz"][0, 3:7], [2 ** -.5, 0, 0, 2 ** -.5], atol=1e-15)
        knee = 7 + JOINT_NAMES_29.index("left_knee_joint")
        self.assertAlmostEqual(motion["qpos_wxyz"][0, knee], np.pi / 6)
        self.assertFalse(motion["qualified_for_q1_training"])

    def test_columns_are_named_and_can_be_reordered(self):
        rows = [{"Frame": i, "left_hip_yaw_joint_dof": 23., "right_hip_yaw_joint_dof": -17.,
                 "root_translateY": 250.} for i in range(2)]
        first = decode_bones_g1(encode(rows), source_hz=120.)
        second = decode_bones_g1(encode(rows, header=list(reversed(HEADER))), source_hz=120.)
        np.testing.assert_array_equal(first["qpos_wxyz"], second["qpos_wxyz"])

    def test_multiaxis_rotation_matches_independent_zyx_matrix(self):
        x, y, z = np.deg2rad([20., 30., 40.])
        rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        motion = decode_bones_g1(encode([{"Frame": i, "root_rotateX": 20.,
                                        "root_rotateY": 30., "root_rotateZ": 40.} for i in range(2)]),
                                 source_hz=120.)
        w, a, b, c = motion["qpos_wxyz"][0, 3:7]
        np.testing.assert_allclose(Rotation.from_quat([a, b, c, w]).as_matrix(), rz @ ry @ rx, atol=1e-14)

    def test_120_to_50_hz_preserves_native_time_and_linear_velocity(self):
        rows = [{"Frame": 1000 + i, "root_translateX": 1000. + i / 120 * 100.,
                 "left_knee_joint_dof": i / 120 * 6.} for i in range(241)]
        result = resample_native_motion(decode_bones_g1(encode(rows), source_hz=120.), target_hz=50.)
        self.assertEqual(len(result["time_s"]), 101)
        self.assertEqual(result["source_duration_s"], 2.)
        self.assertEqual(result["target_duration_s"], 2.)
        np.testing.assert_allclose(result["qpos_wxyz"][:, 0], 10. + result["time_s"], atol=1e-14)
        knee = 7 + JOINT_NAMES_29.index("left_knee_joint")
        np.testing.assert_allclose(result["qpos_wxyz"][:, knee], result["time_s"] * np.pi / 30, atol=1e-14)

    def test_orientation_wrap_uses_shortest_slerp(self):
        motion = decode_bones_g1(encode([{"Frame": 0, "root_rotateZ": 179.},
                                        {"Frame": 1, "root_rotateZ": -179.}]), source_hz=120.)
        result = resample_native_motion(motion, target_hz=240.)
        middle = result["qpos_wxyz"][1, 3:7]
        self.assertAlmostEqual(abs(middle[0]), 0., places=12)
        self.assertAlmostEqual(abs(middle[3]), 1., places=12)

    def test_fractional_endpoint_is_not_padded_or_time_stretched(self):
        result = resample_native_motion(decode_bones_g1(
            encode([{"Frame": i} for i in range(122)]), source_hz=120.), target_hz=50.)
        self.assertEqual(result["target_duration_s"], 1.)
        self.assertAlmostEqual(result["endpoint_truncation_s"], 1 / 120)
        self.assertEqual(result["time_scale"], 1.)

    def test_missing_extra_or_duplicate_columns_are_rejected(self):
        rows = [{"Frame": i} for i in range(2)]
        for header in (HEADER[:-1], HEADER + ["unknown"], HEADER[:-1] + [HEADER[-2]]):
            with self.assertRaisesRegex(ValueError, "schema"):
                decode_bones_g1(encode(rows, header=header), source_hz=120.)

    def test_previously_incorrect_hips_and_short_joint_aliases_are_rejected(self):
        rows = [{"Frame": i} for i in range(2)]
        for header in ([name.replace("root_", "Hips_") for name in HEADER],
                       [name.replace("_joint_dof", "_dof") for name in HEADER]):
            with self.assertRaisesRegex(ValueError, "schema"):
                decode_bones_g1(encode(rows, header=header), source_hz=120.)

    def test_every_named_joint_maps_to_its_explicit_model_slot(self):
        row = {name: index + 1. for index, name in enumerate(HEADER[7:])}
        payload = encode([{**row, "Frame": i} for i in range(2)], header=HEADER[7:] + HEADER[:7])
        motion = decode_bones_g1(payload, source_hz=120.)
        np.testing.assert_allclose(motion["qpos_wxyz"][:, 7:],
                                   np.tile(np.deg2rad(np.arange(1, 30)), (2, 1)))

    def test_nonfinite_gapped_fractional_or_duplicate_frames_are_rejected(self):
        for rows in ([{"Frame": 0}, {"Frame": 2}], [{"Frame": 0}, {"Frame": 0}],
                     [{"Frame": 0}, {"Frame": .5}], [{"Frame": 0}, {"Frame": -1}],
                     [{"Frame": 0}, {"Frame": 1, "left_knee_joint_dof": float("nan")}],
                     [{"Frame": 0}, {"Frame": 1, "root_translateZ": float("inf")}]):
            with self.assertRaises(ValueError):
                decode_bones_g1(encode(rows), source_hz=120.)

    def test_rate_and_count_must_be_explicit_and_output_is_bounded(self):
        payload = encode([{"Frame": i} for i in range(4)])
        for rate in (None, 0, float("nan"), True):
            with self.assertRaises(ValueError):
                decode_bones_g1(payload, source_hz=rate)
        with self.assertRaisesRegex(ValueError, "frame counts"):
            decode_bones_g1(payload, source_hz=120., expected_frames=5)
        with self.assertRaisesRegex(ValueError, "bound"):
            decode_bones_g1(payload, source_hz=120., maximum_bytes=10)
        with self.assertRaisesRegex(ValueError, "output frame count"):
            resample_native_motion(decode_bones_g1(payload, source_hz=120.), target_hz=1e9)


if __name__ == "__main__":
    unittest.main()
