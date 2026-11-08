"""Check real optimizer exposure, sequence timing and old-capability selection."""

import copy
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from student_fit import module_sha
from student_onpolicy_pair_fit import warm_student
from student_retained_sequence_data import sample_windows, sequence_data
from student_retained_sequence_fit import ARMS, fit_pair, increment_loss
from test_student_onpolicy_pair import fixtures as old_fixtures


def fixtures():
    native, payload, origins, candidate, pure, jobs = old_fixtures()
    rows = []
    for origin, values in enumerate(origins+[pure]):
        for split, data in values.items():
            data['frame'] = np.arange(8, dtype=np.int64)*2
            data['terminal_after_action'] = np.zeros(8, dtype=bool)
            local = (candidate['new_collection_jobs'] if origin == 2 else jobs)[0 if split == 'train' else 1]
            rows.append(dict(origin=origin, split=split, clip_id=0 if split == 'train' else 1,
                records=8, four_label_starts=list(range(0, 10, 2)), failure_frame=None, relative=local['relative']))
    return native, payload, origins, candidate, pure, jobs, dict(status='SAVED_SEQUENCE_AUDIT_PASSED', sequence_width=4, records=rows)


class SequenceTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_windows_preserve_origin_source_time_and_terminal_masks(self):
        _, _, origins, candidate, pure, jobs, summary = fixtures()
        old, _, current, sources, windows = sequence_data(origins, pure, candidate, jobs, summary)
        self.assertEqual(windows['old']['train'].shape, (15, 4))
        self.assertEqual(windows['pure']['train'].shape, (5, 4))
        first = sample_windows(old['train'], windows['old']['train'], sources, np.random.default_rng(9), 4, 0)
        second = sample_windows(old['train'], windows['old']['train'], sources, np.random.default_rng(9), 4, 0)
        np.testing.assert_array_equal(first, second)
        for index in first:
            self.assertEqual(len(set(old['train']['origin_id'][index])), 1)
            self.assertEqual(len(set(old['train']['clip_id'][index])), 1)
        pure['train']['terminal_after_action'][6] = True
        with self.assertRaisesRegex(ValueError, 'terminal'):
            sequence_data(origins, pure, candidate, jobs, summary)

    def test_source_reassignment_cannot_turn_validation_into_training(self):
        _, _, origins, candidate, pure, jobs, summary = fixtures()
        summary['records'][0]['relative'] = jobs[1]['relative']
        with self.assertRaisesRegex(ValueError, 'source mapping'):
            sequence_data(origins, pure, candidate, jobs, summary)

    def test_increment_target_matches_motion_instead_of_suppressing_it(self):
        target = torch.arange(4, dtype=torch.float32)[None, :, None].repeat(2, 1, 29)
        present = torch.arange(29) < 22
        self.assertEqual(float(increment_loss(target+3., target, present)), 0.)
        self.assertGreater(float(increment_loss(torch.zeros_like(target), target, present)), 0.)
        with self.assertRaises(ValueError):
            increment_loss(target[:, :3], target[:, :3], present)

    def test_future_state_never_reaches_model_arrays(self):
        _, _, origins, candidate, pure, jobs, summary = fixtures()
        for data in list(pure.values())+[v for o in origins for v in o.values()]:
            data['next_state'] = object()
            data['actual_state'] = object()
        old, _, current, _, _ = sequence_data(origins, pure, candidate, jobs, summary)
        self.assertEqual(set(current['train']), {'f0', 'history', 'teacher_action', 'clip_id', 'origin_id'})
        self.assertEqual(set(old['validation']), set(current['train']))

    def test_actual_paired_adam_and_frozen_reference(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        before = module_sha(native.policy)
        settings = dict(seed=46, maximum_updates=2, windows_per_distribution=2, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1, retention_weight=20., increment_weight=.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, root,
                time.monotonic_ns()+120_000_000_000, dict(scope='synthetic_cpu_only'))
            self.assertEqual(result['actual_optimizer_updates'], 4)
            self.assertEqual(result['arms'][0]['schedule_sha256'], result['arms'][1]['schedule_sha256'])
            for arm in ARMS:
                last = torch.load(root/arm/'last.pt', weights_only=True)
                selected = torch.load(root/arm/'selected.pt', weights_only=True)
                self.assertEqual(last['actual_adam_steps'], [2]*4)
                self.assertEqual(last['schema'], 'q1_f0_retained_sequence_v1')
                self.assertTrue(last['saved_state_open_loop_fitting'])
                self.assertEqual(last['human_real_future_frames'], 0)
                self.assertTrue(all(torch.equal(v, payload['base_encoder'][k]) for k,v in last['base_encoder'].items()))
                self.assertTrue(all(v <= a+1e-12 for a,v in zip(selected['initial_fixed_anchor_risks'], selected['selected_fixed_anchor_risks'])))
            with self.assertRaises(FileExistsError):
                fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, root, 0, {})
        self.assertEqual(module_sha(native.policy), before)

    def test_exact_initial_validation_retains_update_zero(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        student = warm_student(native, payload)
        for values in [v for o in origins for v in o.values()]+list(pure.values()):
            with torch.no_grad():
                values['teacher_action'] = student(torch.from_numpy(values['history']), torch.from_numpy(values['f0'])).numpy().copy()
        settings = dict(seed=47, maximum_updates=1, windows_per_distribution=1, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1, retention_weight=20., increment_weight=.5)
        with tempfile.TemporaryDirectory() as directory:
            result = fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, Path(directory),
                time.monotonic_ns()+120_000_000_000, dict(scope='synthetic_cpu_only'))
            self.assertTrue(all(r['selected_update'] == 0 and r['selected_adapter_unchanged'] for r in result['arms']))


if __name__ == '__main__':
    unittest.main()
