import struct
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .ble import ACTIVE_LOWER, ACTIVE_UPPER, BleSource, IMU, decode_packet
from .calibration import quaternion_mean, summarize_window
from .engine import LatestFgp, WEIGHTS
from .server import Preview


def packet(counter=1):
    data = bytearray(217)
    data[:2] = b"\xff\xfe"
    struct.pack_into("<H", data, 2, 213)
    for i, offset in enumerate([8, 42, 76, 110, 148, 182]):
        IMU.pack_into(data, offset, counter, 1000+i, i, 9.81, 0, 0, 0, 0, 1)
    return data


class PreviewTests(unittest.TestCase):
    def test_misidentified_weight_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong = root / WEIGHTS[0][0]
            wrong.parent.mkdir(parents=True)
            wrong.write_bytes(b"not the EgoLocate canonicalizer")
            with self.assertRaisesRegex(ValueError, "EgoLocate-profile asset mismatch"):
                LatestFgp(root)

    def test_wire_layout_and_active_order(self):
        upper, lower = decode_packet(packet(), "upper"), decode_packet(packet(), "lower")
        self.assertEqual(upper["values"].shape, (6, 7))
        self.assertEqual(lower["values"].shape, (5, 7))
        np.testing.assert_array_equal(upper["values"][:, 3:], [[1, 0, 0, 0]]*6)
        active = BleSource.active_values(dict(upper=upper, lower=lower))
        np.testing.assert_array_equal(active[:, 0], [5, 1, 4, 0, 3, 5, 1, 4, 0, 2])

    def test_corrupt_payloads(self):
        for data in (packet()[:-1], bytes(217)):
            with self.assertRaises(ValueError):
                decode_packet(data, "upper")
        data = packet()
        struct.pack_into("<f", data, 14, float("nan"))
        with self.assertRaises(ValueError):
            decode_packet(data, "upper")

    def test_counter_wrap_duplicate_reset(self):
        source = BleSource()
        self.assertTrue(source.feed("upper", packet(65535)))
        self.assertTrue(source.feed("upper", packet(0)))
        self.assertFalse(source.feed("upper", packet(0)))
        self.assertTrue(source.feed("upper", packet(4)))
        self.assertEqual(source.status["upper"]["missing_packets"], 3)
        source.feed("upper", packet(2))
        self.assertEqual(source.status["upper"]["epoch"], 1)
        self.assertEqual(len(source.frames["upper"]), 1)

    def test_pair_age_and_skew(self):
        source = BleSource()
        now = time.monotonic_ns()
        source.feed("upper", packet(), now-300_000_000)
        source.feed("lower", packet(), now)
        self.assertIsNone(source.pair())
        source.feed("upper", packet(2), now-100_000_000)
        self.assertIsNone(source.pair())
        source.feed("upper", packet(3), now)
        self.assertEqual(source.active_values(source.pair()).shape, (10, 7))

    def test_sign_invariant_calibration(self):
        values = np.zeros((90, 10, 7))
        values[:, :, 1] = 9.81
        values[:, :, 3] = 1
        values[::2, :, 3] = -1
        result = summarize_window(values)
        self.assertTrue(result["static_window_passed"])
        self.assertFalse(result["tpose_geometry_verified"])
        np.testing.assert_allclose(np.array(result["calibration_data"])[30:].reshape(10, 3, 3), np.tile(np.eye(3), (10, 1, 1)))
        np.testing.assert_allclose(quaternion_mean([[1, 0, 0, 0], [-1, 0, 0, 0]]), [1, 0, 0, 0])

    def test_motion_rejects_static_calibration(self):
        values = np.zeros((90, 10, 7))
        values[:, :, 3] = 1
        values[:, :, 0] = np.arange(90)[:, None]
        self.assertFalse(summarize_window(values)["static_window_passed"])

    def test_calibration_requires_live_pair_and_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = Preview("ble", Path(directory), dict(joints=[], vertices=[]))
            with self.assertRaises(ValueError):
                preview.command(dict(action="calibrate"))
            for role in ("upper", "lower"):
                preview.source.feed(role, packet(10))
            preview.command(dict(action="calibrate"))
            preview.collect_calibration(preview.calibration_started+.1)
            self.assertEqual(len(preview.calibration_rows), 1)
            preview.collect_calibration(preview.calibration_started+.2)
            self.assertEqual(len(preview.calibration_rows), 1)
            preview.source.feed("upper", packet(2))
            with self.assertRaisesRegex(ValueError, "epoch changed"):
                preview.collect_calibration(preview.calibration_started+.3)
            self.assertFalse(list(Path(directory).glob("*.json")))

    def test_static_result_saved_without_pose_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            preview = Preview("ble", Path(directory), dict(joints=[], vertices=[]))
            frames = np.zeros((90, 10, 7))
            frames[:, :, 3] = 1
            preview.calibration_rows = list(frames)
            preview.calibration_started = 10
            preview.collect_calibration(13.1)
            self.assertEqual(preview.calibration["state"], "static_passed")
            self.assertFalse(preview.calibration["tpose_geometry_verified"])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
            self.assertEqual(preview.state()["hardware_commands"], 0)

    def test_stale_mesh_is_never_current_smpl(self):
        with tempfile.TemporaryDirectory() as directory:
            preview=Preview("ble",Path(directory),dict(joints=[],vertices=[]))
            preview.live_status="receiving"
            preview.smpl=dict(received_ns=time.monotonic_ns()-300_000_000,
                              inference_ms=30,rotation_orthogonality_max_error=1e-6)
            self.assertFalse(preview.smpl_state()["realtime_valid"])
            self.assertFalse(preview.state()["smpl_realtime_valid"])
            preview.smpl["received_ns"]=time.monotonic_ns()
            self.assertTrue(preview.smpl_state()["realtime_valid"])
            preview.live_status="stale_hold"
            self.assertFalse(preview.smpl_state()["realtime_valid"])


if __name__ == "__main__":
    unittest.main()
