"""Check pure behavior, terminal/split ownership and causal label separation."""

import copy
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from student_input_audit import expected_flags
from student_onpolicy_data import behavior, collection_jobs, validate_records
from student_onpolicy_worker import OnPolicySession
from student_proprio_review_worker import ProprioReviewSession


def fixture():
    event = dict(kind="nominal", first_frame=1, duration_frames=2, transport_frames=0)
    jobs = [dict(split="train", maximum_steps=4, packet_event=event)]
    actual = np.arange(5*79, dtype=np.float64).reshape(5, 79)
    student = np.zeros((4, 29), dtype=np.float32)
    teacher = np.ones((4, 29), dtype=np.float32)
    frames = np.array([0, 2], dtype=np.int64)
    f0 = np.zeros((2, 10, 226), dtype=np.float32)
    for i, frame in enumerate(frames):
        f0[i, :, 223:] = expected_flags(int(frame), 20, event)
    values = dict(f0=f0, history=np.zeros((2, 930), dtype=np.float32),
        teacher_action=teacher[frames], student_action=student[frames], actual_state=actual[frames],
        frame=frames, clip_id=np.zeros(2, dtype=np.int64), teacher_dependency_ns=(frames+10)*20_000_000)
    traces = {0:dict(actual=actual, canonical_action=student, teacher_action=teacher,
        input_ready=np.ones(4, dtype=bool), reference_frames=20)}
    return values, jobs, traces


class OnPolicyTests(unittest.TestCase):
    def test_collection_is_owned_deterministic_and_outcome_independent(self):
        candidate = dict(original_jobs=[dict(split=split, group_id=split+str(i), clip=split+str(i),
            maximum_steps=256) for split, count in (("train", 12), ("validation", 6)) for i in range(count)])
        jobs = collection_jobs(candidate)
        self.assertEqual(len(jobs), 9)
        self.assertTrue(all(j["maximum_steps"] == 192 for j in jobs))
        self.assertEqual(sum(j["split"] == "validation" for j in jobs), 3)
        changed = copy.deepcopy(candidate)
        changed["original_jobs"][0]["clip"] = "fixed__A022"
        with self.assertRaises(ValueError):
            collection_jobs(changed)
        changed = copy.deepcopy(candidate)
        changed["original_jobs"][12]["group_id"] = "train0"
        with self.assertRaises(ValueError):
            collection_jobs(changed)

    def test_shadow_teacher_cannot_change_the_executed_student_action(self):
        student = np.arange(29, dtype=np.float32)[None]
        for value in (-100., 0., 100.):
            teacher = np.full_like(student, value)
            self.assertTrue(np.array_equal(behavior(student, teacher), student))
        with self.assertRaises(ValueError):
            behavior(student, np.full_like(student, np.nan))

    def test_rows_preserve_true_student_state_and_next_transition(self):
        values, jobs, traces = fixture()
        future, result = validate_records(values, "train", jobs, traces)
        self.assertTrue(np.array_equal(future, traces[0]["actual"][[1, 3]]))
        self.assertEqual(result["executed_teacher_fraction"], 0.)
        self.assertTrue(result["student_visibility_verified"])
        for field in ("actual_state", "student_action", "teacher_action"):
            changed = copy.deepcopy(values); changed[field][0, 0] += 1.
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_records(changed, "train", jobs, traces)

    def test_labels_cannot_cross_split_terminal_or_startup(self):
        values, jobs, traces = fixture()
        with self.assertRaises(ValueError):
            validate_records(values, "validation", jobs, traces)
        changed = copy.deepcopy(traces); changed[0]["input_ready"][0] = False
        with self.assertRaises(ValueError):
            validate_records(values, "train", jobs, changed)
        changed = {k:v[1:] for k, v in values.items()}
        with self.assertRaisesRegex(ValueError, "first available"):
            validate_records(changed, "train", jobs, traces)

    def test_future_labels_are_not_disguised_as_current_student_features(self):
        values, jobs, traces = fixture()
        changed = copy.deepcopy(values); changed["teacher_dependency_ns"][:] = 0
        with self.assertRaisesRegex(ValueError, "privilege"):
            validate_records(changed, "train", jobs, traces)
        changed = copy.deepcopy(values); changed["f0"][0, 0, -3] = 1.
        with self.assertRaisesRegex(ValueError, "visibility"):
            validate_records(changed, "train", jobs, traces)

    def test_unsampled_action_is_exactly_the_inherited_student_response(self):
        actor = OnPolicySession.__new__(OnPolicySession)
        actor.args = SimpleNamespace(deadline_ns=time.monotonic_ns()+60_000_000_000)
        actor.job = dict(maximum_steps=4)
        actor.torch = None
        reply = dict(canonical_action=[.25]*29, native_action=[.25]*22, stream=dict(input_ready=True))
        with patch.object(ProprioReviewSession, "action", return_value=copy.deepcopy(reply)) as original:
            result = actor.action(dict(frame=1))
        self.assertEqual(result["canonical_action"], reply["canonical_action"])
        self.assertEqual(result["native_action"], reply["native_action"])
        self.assertIsNone(result["teacher_action"])
        self.assertFalse(result["teacher_queried"])
        original.assert_called_once()


if __name__ == "__main__":
    unittest.main()
