"""Actual CPU branch, causality, paired optimizer and selection contracts."""

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from student_conditioning_models import ARMS, conditioned_student, zero_human_branch
from student_conditioning_fit import fit_pair, initial_audit
from student_fit import module_sha
from student_onpolicy_pair_fit import warm_student
from student_retained_sequence_data import sequence_data
from test_student_retained_sequence import fixtures


class ConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initial_actions_capacity_and_frozen_modules_match_actual_reference(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        old, _, current, _, _ = sequence_data(origins, pure, candidate, jobs, summary)
        result = initial_audit(native, payload, old, current, 48)
        self.assertEqual(result['status'], 'CONDITIONING_INITIAL_ACTION_PARITY_PASSED')
        self.assertEqual(result['actual_autograd_calls'], 2)
        models = []
        for arm in ARMS:
            torch.manual_seed(48)
            model = conditioned_student(native, payload, arm).train()
            self.assertFalse(model.encoder.training or model.decoder.training)
            self.assertTrue(all(p.requires_grad == n.startswith(('adapter.', 'human_adapter.'))
                for n,p in model.named_parameters()))
            self.assertTrue(zero_human_branch(model.human_adapter.state_dict()))
            self.assertTrue(all(torch.equal(v, payload['adapter'][k]) for k,v in model.adapter.state_dict().items()))
            self.assertEqual(len(tuple(model.trainable_parameters())), 8)
            models.append(model)
        self.assertEqual(module_sha(models[0]), module_sha(models[1]))

    def test_only_real_arm_branch_receives_causal_human_signal(self):
        native, payload, origins, _, _, _, _ = fixtures()
        models = []
        for arm in ARMS:
            torch.manual_seed(49)
            models.append(conditioned_student(native, payload, arm))
        human = torch.tensor([[.1, .2, .3], [.2, .4, .6]])
        width = native.policy.cfg.latent
        human = human[:, :1].repeat(1, width)
        self.assertTrue(torch.equal(models[0].conditioning(human), torch.zeros_like(human)))
        self.assertTrue(torch.equal(models[1].conditioning(human), human))
        for model in models:
            with torch.no_grad():
                model.human_adapter[0].weight.fill_(.1)
                model.human_adapter[0].bias.zero_()
                model.human_adapter[-1].weight.fill_(.2)
        zero = models[0].human_adapter(models[0].conditioning(human))
        real = models[1].human_adapter(models[1].conditioning(human))
        self.assertTrue(torch.equal(zero[0], zero[1]))
        self.assertFalse(torch.equal(real[0], real[1]))
        values = origins[0]['train']
        goal, history = torch.from_numpy(values['f0']), torch.from_numpy(values['history'])
        with self.assertRaises(ValueError):
            models[1](history, goal[:, :9])
        with self.assertRaises(TypeError):
            models[1](history, goal, teacher_future_goal=object())

    def test_future_metadata_and_reassigned_splits_cannot_reach_inputs(self):
        _, _, origins, candidate, pure, jobs, summary = fixtures()
        for data in list(pure.values())+[v for o in origins for v in o.values()]:
            data['next_state'] = object()
            data['actual_state'] = object()
            data['teacher_future_goal'] = object()
        old, _, current, _, _ = sequence_data(origins, pure, candidate, jobs, summary)
        self.assertEqual(set(current['train']), {'f0', 'history', 'teacher_action', 'clip_id', 'origin_id'})
        self.assertEqual(set(old['validation']), set(current['train']))
        bad = copy.deepcopy(summary)
        bad['records'][0]['relative'] = jobs[1]['relative']
        with self.assertRaisesRegex(ValueError, 'source mapping'):
            sequence_data(origins, pure, candidate, jobs, bad)

    def test_actual_paired_adam_preserves_sources_objective_and_frozen_modules(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        frozen = module_sha(native.policy)
        settings = dict(seed=50, maximum_updates=2, windows_per_distribution=2, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1, retention_weight=20.)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, root,
                time.monotonic_ns()+120_000_000_000, dict(scope='synthetic_cpu_only'))
            self.assertEqual(result['actual_optimizer_updates'], 4)
            self.assertEqual(result['arms'][0]['schedule_sha256'], result['arms'][1]['schedule_sha256'])
            self.assertEqual(result['arms'][0]['source_sample_counts'], result['arms'][1]['source_sample_counts'])
            for arm in ARMS:
                last = torch.load(root/arm/'last.pt', map_location='cpu', weights_only=True)
                selected = torch.load(root/arm/'selected.pt', map_location='cpu', weights_only=True)
                trace = json.loads((root/arm/'training_trace.json').read_text())
                allowed = [r for r in trace if r['fixed_anchor_non_regression']]
                self.assertEqual(selected['selected_update'], min(allowed, key=lambda r:r['validation_risk'])['update'])
                self.assertEqual(last['actual_adam_steps'], [2]*8)
                self.assertEqual(last['schema'], 'q1_f0_human_conditioned_residual_v1')
                self.assertTrue(last['saved_state_open_loop_fitting'])
                self.assertEqual(last['human_real_future_frames'], 0)
                self.assertTrue(all(torch.equal(v, payload['base_encoder'][k]) for k,v in last['base_encoder'].items()))
                self.assertFalse(all(torch.equal(v, payload['adapter'][k]) for k,v in last['adapter'].items()))
                self.assertTrue(all(v.device.type == 'cpu' and torch.isfinite(v).all()
                    for values in (last['adapter'], last['human_adapter']) for v in values.values()))
                for row in trace[1:]:
                    values = row['metrics']
                    self.assertAlmostEqual(values['loss'], values['point_loss']+20.*values['retention_loss'], delta=1e-6)
            with self.assertRaises(FileExistsError):
                fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, root, 0, {})
        self.assertEqual(module_sha(native.policy), frozen)

    def test_exact_initial_validation_keeps_update_zero_and_structural_equivalence(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        reference = warm_student(native, payload)
        for data in [v for o in origins for v in o.values()]+list(pure.values()):
            with torch.no_grad():
                data['teacher_action'] = reference(torch.from_numpy(data['history']), torch.from_numpy(data['f0'])).numpy().copy()
        settings = dict(seed=51, maximum_updates=1, windows_per_distribution=1, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1, retention_weight=20.)
        with tempfile.TemporaryDirectory() as directory:
            result = fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, Path(directory),
                time.monotonic_ns()+120_000_000_000, dict(scope='synthetic_cpu_only'))
            self.assertTrue(all(r['selected_update'] == 0 and r['selected_policy_unchanged']
                and r['selected_human_branch_output_zero'] for r in result['arms']))

    def test_loss_change_or_short_wall_does_not_silently_make_an_unmatched_pair(self):
        native, payload, origins, candidate, pure, jobs, summary = fixtures()
        settings = dict(seed=52, maximum_updates=2, windows_per_distribution=1, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1, retention_weight=20., increment_weight=.5)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'settings'):
                fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, Path(directory), 0, {})
        settings.pop('increment_weight')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'truncate'):
                fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, Path(directory), 0, {})


if __name__ == '__main__':
    unittest.main()
