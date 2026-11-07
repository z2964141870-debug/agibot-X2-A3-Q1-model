"""Visibility and provenance regressions for the independent R1 evaluator."""

import unittest

import numpy as np

from student_review import review_cases, validate_timing
from common import FIXED
from stream_events import EventSpec, EventTimeline
from student_data import OfflineStudentStream


class StudentReviewTests(unittest.TestCase):
    def setUp(self):
        positions = np.zeros((128, 24, 3))
        rotations = np.zeros((128, 24, 4))
        rotations[..., 0] = 1.
        self.values = positions, rotations

    def sample(self, event="nominal", horizon=4, frame=0):
        spec = EventSpec(event, 50, 10, 0)
        stream = OfflineStudentStream(*self.values, spec=spec, sha256="a"*64, session="review-fixture")
        window = stream.student(frame, mode="short_preview", horizon=horizon)
        fallback = None if window.can_step else ("initial_zero_delta" if frame == 0 else "previous_action_hold")
        value = dict(window.timing(), frame=frame, horizon_frames=horizon, variant="teacher", event=event,
            used_sequences=list(window.used_sequences), input_ready=window.can_step, fallback=fallback,
            policy_device="cuda", gpu_policy_wall_compute_ns=1, worker_action_wall_ns=2,
            packing_maximum_action_difference=0.)
        return value, EventTimeline(128, spec)

    def test_each_student_nominal_and_event_case_has_a_matched_teacher(self):
        rows = review_cases()
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(len(rows), 23)
        for event, variant, horizon, clip in rows:
            if variant == "student":
                self.assertIn((event, "teacher", horizon, clip), rows)
        self.assertEqual({clip for _, _, _, clip in rows}, set(FIXED))

    def test_parent_checks_the_actual_120hz_support_for_every_event_and_short_horizon(self):
        for event in ("nominal", "stop", "reverse", "dropout", "late_burst"):
            for horizon in (1, 4, 10):
                for frame in (0, 1, 49, 50, 55, 61, 100, 126):
                    with self.subTest(event=event, horizon=horizon, frame=frame):
                        value, timeline = self.sample(event, horizon, frame)
                        self.assertIs(validate_timing(value, frame, horizon, "teacher", timeline), value)

    def test_not_ready_h1_cannot_claim_that_the_first_interpolated_target_arrived(self):
        value, timeline = self.sample(horizon=1)
        self.assertFalse(value["input_ready"])
        self.assertEqual(value["fallback"], "initial_zero_delta")
        bad = dict(value, input_ready=True, used_sequences=[1], observed_prefix_frames=1, policy_prefix_frames=1)
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            validate_timing(bad, 0, 1, "teacher", timeline)

    def test_missing_first_target_cannot_skip_forward_to_a_visible_later_slot(self):
        value, timeline = self.sample("dropout", 4, 50)
        self.assertFalse(value["input_ready"])
        self.assertEqual(value["fallback"], "previous_action_hold")
        bad = dict(value, used_sequences=[61])
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            validate_timing(bad, 50, 4, "teacher", timeline)

    def test_hardware_latency_and_unchecked_teacher_packing_are_rejected(self):
        value, timeline = self.sample()
        for key, replacement in (("measured_end_to_end_latency_ns", 80_000_000),
                                 ("packing_maximum_action_difference", .001),
                                 ("max_used_dependency_ns", 10**12)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_timing(dict(value, **{key:replacement}), 0, 4, "teacher", timeline)


if __name__ == "__main__":
    unittest.main()
