"""Standard-library contract controls; no tensor or simulation success claim."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from script.a3.humanplus_specialist_entry import evaluation_tree
from script.a3.humanplus_specialist_protocol import (
    DECODER, INTERACTIONS, OPTIMIZER_STEPS, OUTER_CALLS, UPDATES,
    outcome, ownership, require_training_receipt, serialized_owners, validate_config,
)


class SpecialistControls(unittest.TestCase):
    def test_budget_and_incomplete_receipts(self):
        self.assertEqual((INTERACTIONS, OPTIMIZER_STEPS, OUTER_CALLS), (98304, 1024, 8192))
        receipt = dict(updates=256, interactions=98304, optimizer_steps=1024,
            rollout_actor_calls=6144, native_training_actor_calls=1024, cached_decoder_calls=1024,
            teacher_retention_calls=0, initial_probability=dict(max_abs_ratio_minus_one=0.),
            all_actor_buffers_unchanged=True)
        require_training_receipt(receipt)
        for key in ('updates', 'interactions', 'optimizer_steps', 'cached_decoder_calls'):
            bad = dict(receipt, **{key: receipt[key] - 1})
            with self.assertRaises(ValueError):
                require_training_receipt(bad)
        with self.assertRaises(ValueError):
            require_training_receipt(dict(receipt, teacher_retention_calls=1))

    def test_expert_is_not_unified_and_cannot_pass_local_window(self):
        before = dict(windows=dict(full_rollout=dict(joint_rmse_rad=.2,
            root_pos_error_m_mean=.4, wrists_world_mean_m=.5)))
        after = copy.deepcopy(before)
        after['fall'] = False
        after['windows']['full_rollout']['joint_rmse_rad'] = .19
        result = outcome(before, after)
        self.assertTrue(result['specialist_mujoco_gate_passed'])
        self.assertFalse(result['deploy_allowed'])
        self.assertFalse(result['unified_gate_passed'])
        after['fall'] = True
        self.assertFalse(outcome(before, after)['specialist_mujoco_gate_passed'])
        after['fall'] = False
        after['windows']['full_rollout']['root_pos_error_m_mean'] = .43
        self.assertFalse(outcome(before, after)['specialist_mujoco_gate_passed'])

    def test_ownership_cannot_expand_or_drop_critic(self):
        def parameter(trainable):
            return SimpleNamespace(requires_grad=trainable, shape=(2, 3))
        actor = {DECODER + 'weight': parameter(True),
                 'actor_module.encoders.a3_fast.weight': parameter(False), 'std': parameter(False)}
        critic = {'weight': parameter(True)}
        named = {'policy.' + n: p for n, p in actor.items()}
        named.update({'value_model.' + n: p for n, p in critic.items()})
        wrapper = SimpleNamespace(named_parameters=lambda: named.items(),
            policy=SimpleNamespace(named_parameters=lambda: actor.items()),
            value_model=SimpleNamespace(named_parameters=lambda: critic.items()))
        params = [actor[DECODER + 'weight'], critic['weight']]
        optimizer = SimpleNamespace(param_groups=[dict(params=params)])
        contract = ownership(wrapper, optimizer)
        self.assertEqual(serialized_owners([dict(params=[0, 1])], contract)[1], 'value_model.weight')
        for members in (params[:1], params + [params[0]], params + [actor['std']]):
            optimizer.param_groups = [dict(params=members)]
            with self.assertRaises(ValueError):
                ownership(wrapper, optimizer)
        with self.assertRaises(ValueError):
            serialized_owners([dict(params=[0, 0])], contract)

    def test_config_rejects_budget_motion_and_sampling_drift(self):
        motion = dict(motion_lib_cfg=dict(motion_file=['/registered/zzx'],
            adaptive_sampling=dict(enable=True)), encoder_sample_probs=dict(g1=0., teleop=0., smpl=0., a3_fast=1.),
            coactivate_g1_when_a3_fast=True)
        config = dict(num_envs=16, seed=0, manager_env=dict(commands=dict(motion=motion)),
            algo=dict(trl={}, config=dict(num_learning_iterations=UPDATES, num_steps_per_env=24,
                num_learning_epochs=1, num_mini_batches=4,
                actor=dict(backbone=dict(active_encoders=['g1', 'a3_fast'])))))
        validate_config(config, '/registered/zzx')
        for field, value in (('num_learning_iterations', 512), ('num_learning_epochs', 2)):
            bad = copy.deepcopy(config)
            bad['algo']['config'][field] = value
            with self.assertRaises(ValueError):
                validate_config(bad, '/registered/zzx')
        config['manager_env']['commands']['motion']['motion_lib_cfg']['adaptive_sampling']['enable'] = False
        with self.assertRaises(ValueError):
            validate_config(config, '/registered/zzx')

    def test_no_retention_or_head_trainer_inheritance(self):
        path = Path(__file__).with_name('humanplus_specialist_trainer.py')
        tree = ast.parse(path.read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpecialistTrainer')
        self.assertEqual([ast.unparse(n) for n in trainer.bases], ['ConsistentPrecisionTrainer'])
        methods = {n.name: ast.unparse(n) for n in trainer.body if isinstance(n, ast.FunctionDef)}
        self.assertNotIn('retention', methods['_compute_loss'])
        self.assertIn("batch['mb_sampled_tokens']", methods['_forward_model'])
        self.assertIn('restrict_policy', methods['create_optimizer'])

    def test_eval_arguments_can_be_composed_before_training_config_merge(self):
        from script.a3.humanplus_specialist_trial import launch_arguments
        for stage in ('isaac-baseline', 'isaac-final'):
            arguments = launch_arguments(stage)
            for key in ('headless', 'use_wandb', 'seed', 'num_envs'):
                self.assertEqual(len([v for v in arguments if v.startswith('++' + key + '=')]), 1)
            self.assertIn('++eval_callbacks=[]', arguments)

    def test_pinned_evaluation_adapter_changes_exact_sites(self):
        path = Path('script/vendor/sonic_for_a3/gear_sonic/eval_agent_trl.py')
        if not path.is_file():
            self.skipTest('Pinned vendor source not present on this host')
        tree = evaluation_tree(path.read_text(), str(path), Path('/registered/logs/isaac'))
        compile(tree, str(path), 'exec')
        text = ast.unparse(tree)
        self.assertIn('specialist_replay(env, model, config)', text)
        self.assertIn('specialist_close(simulation_app)', text)
        with self.assertRaises(ValueError):
            evaluation_tree(path.read_text().replace('base_kit_args =', 'other_kit_args ='),
                            str(path), Path('/registered/logs/isaac'))


if __name__ == '__main__':
    unittest.main()
