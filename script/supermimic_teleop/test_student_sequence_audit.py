"""Sequence boundaries must follow physical time and actual terminal states."""

import copy
import unittest

import numpy as np

from student_sequence_audit import source_summary, windows


def fixture():
    frames = np.arange(0, 10, 2, dtype=np.int64)
    actual = np.arange(11*79, dtype=np.float64).reshape(11, 79)
    actions = np.arange(10*29, dtype=np.float32).reshape(10, 29)/100
    f0 = np.zeros((5, 10, 226), dtype=np.float32)
    f0[:, -1, 223:225] = 1.
    trace = dict(actual=actual, canonical_action=actions, teacher_action=actions+.1,
                 input_ready=np.ones(10, dtype=bool), failure_frame=np.int64(10))
    values = dict(frame=frames, f0=f0, history=np.zeros((5, 930), dtype=np.float32),
        actual_state=actual[frames].copy(), teacher_action=trace['teacher_action'][frames].copy(),
        next_state=actual[frames+1].copy(), terminal_after_action=np.zeros(5, dtype=bool))
    return values, trace, dict(failure_frame=10)


class SequenceTests(unittest.TestCase):
    def test_window_uses_labels_not_compressed_control_time(self):
        frames = np.array([8, 0, 4, 2, 6], dtype=np.int64)
        result = windows(frames, None)
        np.testing.assert_array_equal(frames[result], [[0, 2, 4, 6], [2, 4, 6, 8]])
        self.assertEqual(int(np.diff(frames[result][0]).sum())*20, 120)

    def test_gaps_duplicates_and_terminal_cannot_be_joined(self):
        self.assertEqual(len(windows(np.array([0, 2, 6, 8, 10, 12]), None)), 1)
        self.assertEqual(len(windows(np.array([0, 2, 4, 6]), 7)), 0)
        self.assertEqual(len(windows(np.array([0, 2, 4, 6]), 8)), 1)
        with self.assertRaises(ValueError):
            windows(np.array([0, 2, 2, 4]), None)
        ready = np.ones(10, dtype=bool)
        ready[1] = False
        frames = np.arange(0, 10, 2, dtype=np.int64)
        result = windows(frames, None, input_ready=ready)
        np.testing.assert_array_equal(frames[result], [[2, 4, 6, 8]])

    def test_next_state_is_one_control_step_not_next_saved_row(self):
        values, trace, job = fixture()
        result = source_summary(values, trace, job, origin=3)
        self.assertEqual(result['next_state_spacing_ms'], 20)
        self.assertEqual(result['label_spacing_ms'], 40)
        self.assertEqual(result['four_label_windows'], 2)
        changed = copy.deepcopy(values)
        changed['next_state'][0] = trace['actual'][2]
        with self.assertRaisesRegex(ValueError, '20ms'):
            source_summary(changed, trace, job, origin=3)

    def test_terminal_flag_and_source_state_must_match_actual_trace(self):
        values, trace, job = fixture()
        for field in ('terminal_after_action', 'actual_state', 'teacher_action'):
            changed = copy.deepcopy(values)
            changed[field].flat[0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                source_summary(changed, trace, job, origin=3)

    def test_opposite_deltas_are_descriptive_not_a_recovery_label(self):
        values, trace, job = fixture()
        trace['teacher_action'] = -trace['canonical_action']
        values['teacher_action'] = trace['teacher_action'][values['frame']].copy()
        result = source_summary(values, trace, job, origin=3)
        self.assertEqual(result['opposite_action_delta_pairs'], 4)
        self.assertTrue(result['opposite_deltas_are_not_proof_of_harm'])


if __name__ == '__main__':
    unittest.main()
