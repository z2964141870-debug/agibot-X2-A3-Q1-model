"""Standard-library D02 boundary controls; no numerical or simulation claims."""

import ast
import copy
import json
from pathlib import Path
import unittest

from script.a3.humanplus_replay_protocol import (
    CALL_BUDGET, COMMAND, ORIENTATION, forward_accounting, slices, validate_receipt, validate_selection,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'script/a3'


def function(path, name):
    return next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def layout():
    return [('encoder_index', 0, 4, (4,)), (COMMAND, 4, 584, (10, 58)),
            (ORIENTATION, 584, 644, (10, 6))]


def design():
    return json.loads((ROOT / 'data/manifests/a3_humanplus_replay_loss_design_20261018_D02.json').read_text())


class ReplayControls(unittest.TestCase):
    def test_grouped_feature_layout_and_route(self):
        result = slices(layout(), 644, ['g1', 'teleop', 'smpl', 'a3_fast'], ['actor_obs'])
        self.assertEqual(result[COMMAND], (4, 584, (10, 58)))
        self.assertEqual(result[ORIENTATION], (584, 644, (10, 6)))

    def test_layout_rejects_overlap_shape_and_unknown_tail(self):
        for index, replacement in [(1, (COMMAND, 3, 583, (10, 58))),
                                   (2, (ORIENTATION, 584, 644, (6, 10)))]:
            value = layout()
            value[index] = replacement
            with self.assertRaises(ValueError):
                slices(value, 644, ['g1', 'teleop', 'smpl', 'a3_fast'], ['actor_obs'])
        with self.assertRaises(ValueError):
            slices(layout(), 645, ['g1', 'teleop', 'smpl', 'a3_fast'], ['actor_obs'])

    def test_route_order_cannot_silently_switch_encoder(self):
        with self.assertRaises(ValueError):
            slices(layout(), 644, ['a3_fast', 'smpl', 'teleop', 'g1'], ['actor_obs'])
        with self.assertRaises(ValueError):
            slices(layout(), 644, ['g1', 'teleop', 'smpl', 'a3_fast'], ['critic_obs'])

    def test_uniform_selection_exact_and_tamper_rejected(self):
        value = design()
        rows = validate_selection(value)
        self.assertEqual([r['full_policy_steps'] for r in rows], [637, 882, 1652, 2604])
        value['loss_examples']['selection'][0]['selected_policy_ticks'][1] += 1
        with self.assertRaises(ValueError):
            validate_selection(value)

    def test_budget_counts_all_paths_and_refuses_extra_batch(self):
        self.assertEqual(forward_accounting(8, 32, 32, 288), CALL_BUDGET)
        self.assertEqual(forward_accounting(8, 32, 32, 64)['total'], 872)
        with self.assertRaises(ValueError):
            forward_accounting(8, 33, 32, 288)

    def receipt(self):
        value = design()
        sources = [dict(row, **value['read_only_sources'][row['source']])
                   for row in value['loss_examples']['selection']]
        receipt = dict(sources=sources, frames=384, input_order=value['loss_examples']['input_order'],
            preclip_labels=True, native_route=[0, 0, 0, 1], initial_calls=4, coefficient=1.,
            legacy_hard_guard_unchanged=True, label_tolerance=1e-5, fresh035_label_max_abs=1e-6,
            roles='DEVELOPMENT_REGULARIZATION_NOT_HELD_OUT',
            batches=[dict(commit_before=0, quarter=0, frames=96, loss=.01)])
        metric = dict(replay_quarter=0, replay_retention_loss=.01)
        rows = [dict(status='ACCEPTED_PENDING_COMMIT', before=metric, candidates=[metric])]
        return receipt, value, rows

    def test_receipt_binds_loss_to_live_candidate_and_original_labels(self):
        receipt, value, rows = self.receipt()
        self.assertEqual(validate_receipt(receipt, value, 1, rows)['replay_batches'], 1)
        for key, replacement in [('preclip_labels', False), ('fresh035_label_max_abs', float('nan')),
                                 ('legacy_hard_guard_unchanged', False), ('native_route', [1, 0, 0, 0])]:
            broken = copy.deepcopy(receipt)
            broken[key] = replacement
            with self.assertRaises(ValueError):
                validate_receipt(broken, value, 1, rows)

    def test_receipt_rejects_mismatched_quarter_and_live_loss(self):
        receipt, value, rows = self.receipt()
        for key, replacement in [('replay_quarter', 1), ('replay_retention_loss', .02)]:
            broken = copy.deepcopy(rows)
            broken[0]['before'][key] = replacement
            with self.assertRaises(ValueError):
                validate_receipt(receipt, value, 1, broken)

    def test_trainer_inherits_original_ownership_commit_and_rollback(self):
        tree = ast.parse((HERE / 'humanplus_replay_trainer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ReplayLossTrainer')
        self.assertEqual(ast.unparse(cls.bases[0]), 'HeadOnlyTrainer')
        overrides = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse(overrides & {'create_optimizer', 'train', '_before_optimizer_step',
                                     '_after_optimizer_step', '_record_candidate'})
        loss = ast.unparse(function(HERE / 'humanplus_replay_trainer.py', '_compute_loss'))
        self.assertIn('ConsistentPrecisionTrainer._compute_loss', loss)
        self.assertNotIn('super()._compute_loss', loss)
        self.assertNotIn('_cloth_targets', loss)

    def test_entry_and_verifier_bind_independent_trial_and_pure_probability(self):
        for name in ('humanplus_replay_loss_entry.py', 'humanplus_replay_loss_verification.py'):
            source = (HERE / name).read_text()
            self.assertIn('humanplus_replay_loss as trial', source)
            self.assertNotIn('humanplus_head_logged as trial', source)
            self.assertIn('validate_receipt(', source)
        self.assertIn('humanplus_saved_probability import initial_probability',
                      (HERE / 'humanplus_replay_loss_verification.py').read_text())

    def test_final_effect_formula_and_real_bundle_preserved(self):
        before = function(HERE / 'humanplus_head_logged.py', 'effect')
        after = function(HERE / 'humanplus_replay_loss.py', 'effect')
        self.assertEqual(ast.dump(before, include_attributes=False), ast.dump(after, include_attributes=False))
        source = ast.unparse(function(HERE / 'humanplus_replay_loss.py', 'evaluate'))
        self.assertIn('materialize_bundle(', source)
        self.assertIn('checkpoint_metadata(', source)
        check = ast.unparse(function(HERE / 'humanplus_replay_loss.py', 'verify_replays'))
        self.assertIn("np.clip(captured['actions'].astype(np.float64), -20.0, 20.0)", check)


if __name__ == '__main__':
    unittest.main()
