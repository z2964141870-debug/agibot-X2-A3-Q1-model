"""Analytic timing and TEST_ONLY boundary checks; no real sensor data fitting."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recorded_intake_audit import audit_bundle, interpolation_dependency, validated_times


class RecordedIntakeAuditTests(unittest.TestCase):
    def test_irregular_interpolation_exposes_right_support_wait_and_hold_age(self):
        row = interpolation_dependency([0., .03, .09, .12])
        self.assertEqual(row["output_frames"], 7)
        self.assertAlmostEqual(row["future_support_fraction"], 5 / 7)
        self.assertAlmostEqual(row["offline_interpolation_future_dependency"]["maximum_ms"], 50.)
        self.assertAlmostEqual(row["causal_hold_data_age"]["maximum_ms"], 50.)
        exact = interpolation_dependency(np.arange(7) * .02)
        self.assertEqual(exact["future_support_fraction"], 0.)
        self.assertLess(exact["offline_interpolation_future_dependency"]["maximum_ms"], 1e-9)

    def test_clock_origin_does_not_create_a_measured_latency(self):
        first = interpolation_dependency([0., .03, .09, .12])
        second = interpolation_dependency(np.array([0., .03, .09, .12]) + 1234.)
        self.assertEqual(first["output_frames"], second["output_frames"])
        self.assertAlmostEqual(first["offline_interpolation_future_dependency"]["maximum_ms"],
                               second["offline_interpolation_future_dependency"]["maximum_ms"], places=6)
        self.assertIn("neither is measured", first["timing_note"])

    def test_invalid_recording_times_rejected(self):
        for values in ([0.], [0., 0.], [1., .5], [0., np.nan]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validated_times(values)

    def test_test_only_contract_and_source_identity_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "body_only.npz"
            for allowed in (False, True):
                np.savez(path, usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(allowed),
                    host_monotonic_s=[10., 10.03, 10.09, 10.12],
                    source_time_s=[100., 100.03, 100.09, 100.12], elapsed_s=[0., .03, .09, .12],
                    frame_index=[0, 1, 2, 3], joints=np.zeros((4, 24, 3)),
                    pose_matrix=np.zeros((4, 24, 9)), trans=np.zeros((4, 3)),
                    source_sha256=np.asarray("a" * 64))
                manifest = dict(usage="TEST_ONLY", training_allowed=False,
                    bundle=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                path.with_name("inspection.json").write_text(json.dumps(manifest))
                if allowed:
                    with self.assertRaisesRegex(ValueError, "training data"):
                        audit_bundle(path)
                else:
                    value = audit_bundle(path)
                    self.assertFalse(value["training_allowed"])
                    self.assertFalse(value["world_locomotion_following_assessable"])
                    self.assertIsNone(value["measured_end_to_end_latency_ms"])
                    self.assertEqual(value["input_integrity"], "PASSED")
                    manifest["bundle"]["sha256"] = "b" * 64
                    path.with_name("inspection.json").write_text(json.dumps(manifest))
                    with self.assertRaisesRegex(ValueError, "unchanged"):
                        audit_bundle(path)


if __name__ == "__main__":
    unittest.main()
