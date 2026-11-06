"""Offline source resampling must not be mislabeled causal capture."""

import unittest

import numpy as np

from forecast_data import source_support
from stream_contract import DT_NS
from stream_events import EventSpec
from student_data import OfflineStudentStream
from student_inputs import current_features


def stream(kind="nominal", future_marker=False):
    positions = np.zeros((40, 24, 3))
    positions[:, :, 2] = 1.
    positions[:, :, 0] = np.arange(40)[:, None]*.01
    if future_marker:
        positions[13:, :, :] = 1e6
    rotations = np.zeros((40, 24, 4))
    rotations[:, :, 0] = 1.
    return OfflineStudentStream(positions, rotations,
        spec=EventSpec(kind, first_frame=12, duration_frames=4), sha256="a"*64,
        session="synthetic_only")


class StudentDataTests(unittest.TestCase):
    def test_interpolated_f0_uses_the_previous_sample_until_raw_support_is_available(self):
        value = stream().student(12, mode="f0_history")
        self.assertGreater(source_support(12)["right_support_ns"], 0)
        self.assertEqual(value.samples[-1].sequence, 11)
        self.assertLessEqual(value.samples[-1].dependency_end_ns, 12*DT_NS)
        self.assertEqual(value.timing()["upstream_preprocessing"], "offline_unverified")
        self.assertEqual(stream().student(15, mode="f0_history").samples[-1].sequence, 15)

    def test_short_preview_records_actual_prefix_when_last_interpolant_is_not_ready(self):
        result = stream().student(12, mode="short_preview", horizon=4)
        self.assertEqual(result.requested_frames, 4)
        self.assertEqual(result.prefix_frames, 3)
        self.assertLessEqual(result.max_used_dependency_ns, 16*DT_NS)

    def test_privileged_teacher_labels_can_change_without_changing_student(self):
        original, altered = stream(), stream(future_marker=True)
        left = original.student(12, mode="f0_history")
        right = altered.student(12, mode="f0_history")
        np.testing.assert_array_equal(current_features(left, (1., 0., 0., 0.)),
                                      current_features(right, (1., 0., 0., 0.)))
        self.assertFalse(np.array_equal(original.teacher_labels(12).positions,
                                        altered.teacher_labels(12).positions))
        self.assertTrue(original.teacher_labels(12).privileged_training_labels_only)
        with self.assertRaises(ValueError):
            original.teacher_labels(30)

    def test_dropout_and_late_burst_apply_source_readiness_before_arrival_selection(self):
        for kind in ("dropout", "late_burst"):
            data = stream(kind)
            during = data.student(15, mode="f0_history")
            self.assertEqual(during.samples[-1].sequence, 11)
            recovered = data.student(17, mode="f0_history")
            self.assertEqual(recovered.samples[-1].sequence, 16)
            self.assertTrue(all(row is None or row.arrival_ns <= 17*DT_NS
                                for row in recovered.samples))


if __name__ == "__main__":
    unittest.main()
