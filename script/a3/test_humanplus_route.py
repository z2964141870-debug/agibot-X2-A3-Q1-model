"""D03 exact-variable, route-budget, independent-binding controls; no model."""

import ast
import copy
from pathlib import Path
import unittest

from script.a3.humanplus_route_protocol import (
    OVERRIDE, PROBABILITIES, recipe, validate_config, validate_routes,
)

HERE = Path(__file__).resolve().parent


def config():
    return dict(manager_env=dict(commands=dict(motion=dict(
        encoder_sample_probs=dict(PROBABILITIES), coactivate_g1_when_a3_fast=True,
        motion_lib_cfg=dict(motion_file=['new']), reward=3))),
        algo=dict(trl=dict(output_dir='new', logging_dir='new'), config=dict(actor=dict(
            backbone=dict(active_encoders=['g1', 'a3_fast'],
                encoder_sample_probs='${manager_env.commands.motion.encoder_sample_probs}')),
            actor_learning_rate=2e-6)))


def old_config():
    value = config()
    value['manager_env']['commands']['motion']['encoder_sample_probs']['g1'] = 1.
    return value


class RouteControls(unittest.TestCase):
    def test_exact_single_variable_and_storage_allowed(self):
        new, old = config(), old_config()
        old['manager_env']['commands']['motion']['motion_lib_cfg']['motion_file'] = ['old']
        old['algo']['trl'].update(output_dir='old', logging_dir='old')
        before = copy.deepcopy(new)
        recipe(new, old)
        self.assertEqual(new, before)

    def test_loss_lr_reward_and_pairing_changes_rejected(self):
        for path, value in [(('algo', 'config', 'actor_learning_rate'), 1e-4),
                (('manager_env', 'commands', 'motion', 'reward'), 4),
                (('manager_env', 'commands', 'motion', 'coactivate_g1_when_a3_fast'), False)]:
            new = config()
            node = new
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.assertRaises(ValueError):
                recipe(new, old_config())

    def test_encoder_network_removal_or_probability_change_rejected(self):
        for key, value in [('active_encoders', ['a3_fast']),
                           ('encoder_sample_probs', dict(PROBABILITIES, g1=1.))]:
            new = config()
            new['algo']['config']['actor']['backbone'][key] = value
            with self.assertRaises(ValueError):
                validate_config(new)

    def test_all_routes_and_complete_budget(self):
        rows = [dict(update=i + 1, trainer_global_step=i, transitions=384,
                     routes={'a3_fast': 384}) for i in range(8)]
        self.assertEqual(validate_routes(rows, 8)['a3_fast_transitions'], 3072)
        self.assertEqual(validate_routes(rows[:3], 3)['a3_fast_transitions'], 1152)
        for broken in ([dict(rows[0], routes={'g1': 1, 'a3_fast': 383}), *rows[1:]],
                       rows[::-1], rows + [rows[-1]], []):
            with self.assertRaises(ValueError):
                validate_routes(broken, len(broken))

    def test_trainer_keeps_optimizer_loss_acceptance_and_cached_tokens(self):
        tree = ast.parse((HERE / 'humanplus_route_trainer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RouteTrainer')
        self.assertEqual(ast.unparse(cls.bases[0]), 'ReplayLossTrainer')
        self.assertEqual({n.name for n in cls.body if isinstance(n, ast.FunctionDef)},
                         {'load_checkpoint', '_capture_tokens', '_rollout_step'})
        source = ast.unparse(cls)
        self.assertIn('[1, 0, 0, 1]', source)
        self.assertIn('super()._capture_tokens(module, arguments, output)', source)
        self.assertNotIn('routes[', source)

    def test_independent_trial_bindings_and_no_old_execution(self):
        root = HERE.parents[1]
        names = [HERE / ('humanplus_route_' + n + '.py') for n in ('entry', 'verification')]
        names += [root / 'script/project_tools' / ('humanplus_route_' + n + '.py')
                  for n in ('audit', 'stop_audit')]
        for path in names:
            source = path.read_text()
            self.assertIn('humanplus_route_trial as trial', source)
            self.assertNotIn('humanplus_replay_loss as trial', source)
        source = (HERE / 'humanplus_route_trial.py').read_text()
        self.assertIn('OVERRIDE,', source)
        self.assertIn('R25_001_s0', source)
        self.assertIn('a3_humanplus_route_20261018_HP19', source)
        self.assertNotIn("'script.a3.humanplus_replay_loss'", source)

    def test_final_effect_and_capture_match_original(self):
        def function(name, key):
            return next(n for n in ast.walk(ast.parse((HERE / name).read_text()))
                        if isinstance(n, ast.FunctionDef) and n.name == key)
        for key in ('effect', 'verify_replays', 'evaluate'):
            self.assertEqual(ast.dump(function('humanplus_route_trial.py', key), include_attributes=False),
                ast.dump(function('humanplus_replay_loss.py', key), include_attributes=False))

    def test_no_new_forward_budget_and_exact_launch_override(self):
        self.assertEqual(OVERRIDE, 'manager_env.commands.motion.encoder_sample_probs.g1=0.0')
        from script.a3.humanplus_replay_protocol import CALL_BUDGET
        self.assertEqual(CALL_BUDGET['total'], 2440)
        source = (HERE / 'humanplus_route_entry.py').read_text()
        self.assertIn('validate_routes(routes, 8)', source)
        self.assertIn('validate_config(config)', source)
        self.assertIn('validate_receipt(replay, trial.read(trial.DESIGN), 32, rows)', source)


if __name__ == '__main__':
    unittest.main()
