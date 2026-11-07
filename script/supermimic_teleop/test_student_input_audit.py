"""Saved-data audit contracts; these fixtures do not run a real robot policy."""

import unittest

import numpy as np

from student_input_audit import (expected_flags, neighbor_proxy, normalization_metrics,
                                 normalized_input, validate_rows)
from student_data import OfflineStudentStream
from stream_events import EventSpec


class SavedInputAuditTests(unittest.TestCase):
    def test_clock_reconstruction_matches_real_visibility_interface(self):
        positions = np.zeros((80, 24, 3))
        rotations = np.zeros((80, 24, 4))
        rotations[..., 0] = 1.
        for event in ('nominal', 'dropout', 'late_burst'):
            for transport in (0, 1, 2):
                spec = EventSpec(event, 20, 8, transport)
                stream = OfflineStudentStream(positions, rotations, spec=spec,
                                              sha256='a'*64, session='fixture')
                for frame in (0, 1, 9, 10, 20, 24, 28, 30):
                    window = stream.student(frame, mode='f0_history')
                    wanted = np.array([(0., 0., 0.) if row is None else
                        (1., float(row.capture_ns == target), (target-row.capture_ns)/1e9)
                        for target, row in zip(window.targets_ns, window.samples)])
                    np.testing.assert_array_equal(expected_flags(frame, 80, vars(spec)), wanted)

    def test_missing_rows_do_not_create_normalization_outliers(self):
        values = np.zeros((2, 10, 226))
        values[:, -1, :223] = .5
        values[:, -1, 223] = 1.
        stats = dict(mean=[.5]*223, scale=[.01]*223)
        metrics = normalization_metrics(values, stats, np.zeros(223), np.ones(223))
        self.assertEqual(metrics['clipped_coordinate_fraction'], 0.)
        self.assertEqual(metrics['valid_history_fraction'], .1)
        values[0, -1, 220] = 1.
        self.assertGreater(normalization_metrics(values, stats, np.zeros(223), np.ones(223))
                           ['blocks']['relative_torso']['clipped_fraction'], 0.)

    def fixture(self):
        spec = dict(kind='nominal', first_frame=20, duration_frames=8, transport_frames=0)
        jobs = [dict(split='train', maximum_steps=40, packet_event=spec)]
        values = np.zeros((1, 10, 226), dtype=np.float32)
        values[0, :, 223:] = expected_flags(10, 50, spec)
        data = dict(f0=values, frame=np.array([10]), clip_id=np.array([0]),
            actual_state=np.zeros((1, 79)), teacher_action=np.zeros((1, 29)))
        trace = {0:dict(actual=np.zeros((41, 79)), canonical_action=np.zeros((40, 29)))}
        return data, jobs, trace

    def test_reject_changed_state_future_age_duplicate_and_split(self):
        data, jobs, traces = self.fixture()
        self.assertTrue(validate_rows(data, 'train', jobs, traces, 0)['saved_flags_clock_match'])
        for change in ('state', 'age', 'duplicate', 'split'):
            data, jobs, traces = self.fixture()
            if change == 'state':
                data['actual_state'][0, 0] = 1.
            elif change == 'age':
                data['f0'][0, -1, 225] = -.02
            elif change == 'duplicate':
                data = {k:np.concatenate((v, v)) for k,v in data.items()}
            else:
                jobs[0]['split'] = 'test'
            with self.assertRaises(ValueError):
                validate_rows(data, 'train', jobs, traces, 0)

    def test_normalized_input_has_actual_encoder_blocks_and_clipping(self):
        f0 = np.zeros((1, 10, 226), dtype=np.float32)
        f0[:, -1, 223:] = (1., 0., .02)
        f0[:, -1, 0] = 99.
        data = dict(f0=f0, history=np.ones((1, 930), dtype=np.float32))
        out = normalized_input(data, dict(mean=[0.]*223, scale=[1.]*223))
        self.assertEqual(out.shape, (1, 3190))
        self.assertAlmostEqual(float(out[0, 9*226]), 10/np.sqrt(4520), places=6)
        self.assertTrue(np.all(out[0, :9*226] == 0.))

    def test_neighbor_queries_cannot_reuse_the_same_source(self):
        def data(ids, labels):
            f0 = np.zeros((len(ids), 10, 226), dtype=np.float32)
            f0[..., 223] = 1.
            return dict(f0=f0, history=np.zeros((len(ids), 930), dtype=np.float32),
                        clip_id=np.array(ids), teacher_action=np.array(labels, dtype=np.float32))
        origins = [dict(train=data([0, 1], [[0.]*29, [2.]*29]),
                        validation=data([2], [[1.]*29]))]
        result = neighbor_proxy(origins, dict(mean=[0.]*223, scale=[1.]*223), [0, 22])
        rows = result['rows']
        self.assertEqual([r['nearest_label_mse_mean'] for r in rows], [4., 4., 1.])
        self.assertEqual(result['training_bank_rows'], 2)


if __name__ == '__main__':
    unittest.main()
