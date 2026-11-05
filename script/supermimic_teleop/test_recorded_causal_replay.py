"""TEST_ONLY usage and fixed stress replay checks without sensor-data fitting."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_resampler import ArrivedPose, IntakeSource
from recorded_causal_replay import RecordedInput, SMPL_PARENTS, replay_recording
from stream_contract import HumanPose


class RecordingFixture:
    source = IntakeSource("synthetic", "fixed", "payload", "host")
    arrivals = [10_000_000_000+i*30_000_000 for i in range(14)]

    def packet(self, index):
        pose = HumanPose([(index*.03, 0., 0.)]*24, [(1., 0., 0., 0.)]*24)
        return ArrivedPose(self.source, index, 1_000_000_000_000+index*30_000_000,
                           self.arrivals[index], pose)


class RecordedCausalReplayTests(unittest.TestCase):
    def test_fixed_synthetic_dropout_causes_forecast_fallback_without_future_use(self):
        result = replay_recording(RecordingFixture(), (30_000_000, 300_000_000))
        self.assertEqual(result["deliberately_dropped_packets"], 10)
        for row in result["methods"].values():
            self.assertEqual(row["local_causality_violations"], 0)
        self.assertIn("EXTRAPOLATION_LIMIT", result["methods"]["damped_now"]["status"])
        self.assertIn("STALE_SUPPORT", result["methods"]["hold_now"]["status"])
        self.assertIn("SUPPORT_GAP_LIMIT", result["methods"]["linear_buffer40"]["status"])
        self.assertEqual(result["methods"]["linear_buffer40"]["scheduled_buffer_ms"], 40.)

    def test_reader_preserves_test_only_clocks_and_parent_orientation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"body_only.npz"
            angles = np.zeros((3, 72))
            angles[:, 2] = np.pi/2
            np.savez(path, usage="TEST_ONLY", training_allowed=False, parents=SMPL_PARENTS,
                host_monotonic_s=[10., 10.03, 10.06], source_time_s=[1000., 1000.03, 1000.06],
                frame_index=[0, 1, 2], joints=np.zeros((3, 24, 3)), axis_angle=angles,
                trans=np.zeros((3, 3)))
            manifest = dict(usage="TEST_ONLY", training_allowed=False,
                bundle=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            inspection = path.with_name("inspection.json")
            inspection.write_text(json.dumps(manifest))
            recording = RecordedInput(path)
            packet = recording.packet(1)
            self.assertEqual(packet.arrival_ns, 10_030_000_000)
            self.assertEqual(packet.payload_ns, 1_000_030_000_000)
            self.assertIsNone(packet.payload_dependency_end_ns)
            self.assertIsNone(packet.receiver_ready_ns)
            np.testing.assert_allclose(packet.pose.orientations[0], packet.pose.orientations[23])
            np.testing.assert_allclose(packet.pose.orientations[0], [2**-.5, 0, 0, 2**-.5])
            recording.verify()
            manifest["training_allowed"] = True
            inspection.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "TEST_ONLY"):
                RecordedInput(path)
            with self.assertRaisesRegex(RuntimeError, "changed"):
                recording.verify()


if __name__ == "__main__":
    unittest.main()
