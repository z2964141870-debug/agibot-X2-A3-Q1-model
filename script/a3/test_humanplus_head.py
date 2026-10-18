"""Standard-library D01 controls; fakes test orchestration, never numerical training."""

import ast
import copy
import json
from pathlib import Path
import types
import unittest

from script.a3.humanplus_head_protocol import (
    CALL_BUDGET, HEAD_KEYS, HEAD_SHAPES, changed_actor, digest, forward_accounting,
    ownership, restrict_head, serialized_owners, transactions,
)

HERE = Path(__file__).parent


class Parameter:
    def __init__(self, shape):
        self.shape, self.requires_grad, self.grad = shape, True, None

    def requires_grad_(self, flag):
        self.requires_grad = flag


class Network:
    def __init__(self, values):
        self.values = values

    def named_parameters(self, **kwargs):
        return iter(self.values)


class Wrapper:
    def __init__(self):
        self.policy = Network([(n, Parameter(s)) for n, s in HEAD_SHAPES.items()] + [
            ('actor_module.encoders.g1.weight', Parameter((3, 2))),
            ('actor_module.decoders.g1_dyn.module.0.weight', Parameter((2, 3))),
            ('std', Parameter((29,)))])
        self.value_model = Network([('value.weight', Parameter((1, 2))), ('value.bias', Parameter((1,)))])

    def named_parameters(self, **kwargs):
        return iter([('policy.' + n, p) for n, p in self.policy.values]
                    + [('value_model.' + n, p) for n, p in self.value_model.values])


class Adam:
    def __init__(self, groups):
        self.param_groups = [dict(g, params=list(g['params'])) for g in groups]
        self.state = {}


class Finite(Adam):
    def __init__(self, groups, **callbacks):
        super().__init__(groups)
        self.callbacks = callbacks


def head_trainer():
    """Execute the actual override source with a tensor-free creator boundary."""
    class Parent:
        def create_optimizer(self):
            raise AssertionError('Cloth ancestor must never re-enable decoder')

        def _evaluate_candidate(self, *args):
            raise AssertionError('No numerical candidate in standard-library controls')

        def _record_candidate(self, *args):
            raise AssertionError('No numerical transaction in standard-library controls')

    class Generic:
        @staticmethod
        def create_optimizer(trainer):
            trainer.order.append('generic_creator')
            assert {n for n, p in trainer.model.policy.values if p.requires_grad} == set(HEAD_KEYS)
            selected = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
            options = dict(lr=2e-6, betas=(.9, .999), eps=1e-8, fused=True, foreach=None)
            trainer.optimizer = Adam([
                dict(options, params=[p for n, p in selected if n.endswith('weight')], weight_decay=.01),
                dict(options, params=[p for n, p in selected if n.endswith('bias')], weight_decay=0.)])
            return trainer.optimizer

    tree = ast.parse((HERE / 'humanplus_head_trainer.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ('HeadOnlyTrainer', 'group_options')]
    scope = dict(torch=types.SimpleNamespace(optim=types.SimpleNamespace(AdamW=Adam)),
        Trainer=Generic, FiniteRetentionTrainer=Parent, FiniteAdamW=Finite,
        restrict_head=restrict_head, ownership=ownership, digest=digest, HEAD_KEYS=HEAD_KEYS)
    exec(compile(tree, '<actual D01 trainer override>', 'exec'), scope)
    trainer = scope['HeadOnlyTrainer']()
    trainer.model, trainer.optimizer, trainer.optimizer_cls_and_kwargs = Wrapper(), None, None
    trainer.order = []
    return trainer


def transaction_rows(commits=2, rejected=False):
    rows = []
    for number in range(1, commits + 1 + int(rejected)):
        reject = rejected and number == commits + 1
        metrics = dict(objective=1., teacher_rmse=.01, teacher_max_abs=.1)
        candidates = [dict(metrics, scale=1., accepted=True)]
        if reject:
            from script.a3.finite_retention_protocol import SCALES
            candidates = [dict(metrics, scale=s, teacher_rmse=.06, accepted=False) for s in SCALES]
        row = dict(proposal=number, owner_sha256='owner', before=metrics,
            candidates=candidates, evaluations=1+len(candidates), nominal_group_lrs=[1e-6, 1e-6],
            head_gradients={n: dict(shape=list(s), sha256='a'*64, max_abs=.1) for n, s in HEAD_SHAPES.items()},
            parameters_and_optimizer_before_sha256='before', parameters_and_optimizer_now_sha256='before')
        if reject:
            rows.append(dict(row, status='REJECTED_NO_LIVE_UPDATE'))
        else:
            rows.extend([dict(row, status='ACCEPTED_PENDING_COMMIT'),
                         dict(row, status='COMMITTED', parameters_and_optimizer_now_sha256='after')])
    return rows


class HeadControls(unittest.TestCase):
    def test_actual_creator_freezes_before_generic_and_returns_final_finite(self):
        trainer = head_trainer()
        optimizer = trainer.create_optimizer()
        self.assertIs(type(optimizer), Finite)
        self.assertEqual(trainer.order, ['generic_creator'])
        self.assertEqual(trainer._cloth_scope['trainable'], list(HEAD_KEYS))
        self.assertEqual([g['weight_decay'] for g in optimizer.param_groups], [.01, 0.])
        self.assertTrue(all(g['fused'] for g in optimizer.param_groups))
        scheduler = types.SimpleNamespace(optimizer=optimizer)
        self.assertIs(trainer.create_optimizer(), scheduler.optimizer)
        self.assertEqual(trainer.order, ['generic_creator'])

    def test_creator_refuses_preexisting_optimizer_and_override(self):
        for field, value in [('optimizer', Adam([])), ('optimizer_cls_and_kwargs', (Adam, {}))]:
            trainer = head_trainer()
            setattr(trainer, field, value)
            with self.assertRaises(ValueError):
                trainer.create_optimizer()

    def test_missing_or_wrong_head_rejected_before_optimizer(self):
        for mutation in ('missing', 'shape'):
            wrapper = Wrapper()
            if mutation == 'missing':
                wrapper.policy.values.pop(0)
            else:
                wrapper.policy.values[0][1].shape = (29, 511)
            with self.assertRaises(ValueError):
                restrict_head(wrapper.policy)

    def test_reenabled_backbone_or_frozen_gradient_rejected(self):
        for mode in ('flag', 'gradient'):
            trainer = head_trainer()
            trainer.create_optimizer()
            parameter = trainer.model.policy.values[-1][1]
            if mode == 'flag':
                parameter.requires_grad = True
            else:
                parameter.grad = object()
            with self.assertRaises(ValueError):
                trainer._check_head(trainer.model)

    def test_missing_duplicate_foreign_and_frozen_owners_rejected(self):
        for mode in ('missing', 'duplicate', 'foreign', 'frozen'):
            trainer = head_trainer()
            optimizer = trainer.create_optimizer()
            group = optimizer.param_groups[0]['params']
            if mode == 'missing':
                group.pop()
            else:
                group.append(group[0] if mode == 'duplicate' else
                    Parameter((1,)) if mode == 'foreign' else trainer.model.policy.values[-1][1])
            with self.assertRaises(ValueError):
                ownership(trainer.model, optimizer)

    def test_critic_freezing_discriminator_and_alias_rejected(self):
        for mode in ('critic', 'discriminator', 'alias'):
            trainer = head_trainer()
            trainer.create_optimizer()
            if mode == 'critic':
                trainer.model.value_model.values[0][1].requires_grad = False
            elif mode == 'discriminator':
                trainer.model.disc_model = object()
            else:
                trainer.model.policy.values.append(('alias', trainer.model.policy.values[0][1]))
            with self.assertRaises(ValueError):
                trainer._check_head(trainer.model)

    def test_serialized_integer_ids_map_in_group_order(self):
        trainer = head_trainer()
        optimizer = trainer.create_optimizer()
        contract = ownership(trainer.model, optimizer)
        saved = [dict(params=[4, 8]), dict(params=[3, 7])]
        mapping = serialized_owners(saved, contract)
        self.assertEqual(list(mapping.values()), [n for g in contract['groups'] for n in g])
        for invalid in ([dict(params=[4]), dict(params=[3, 7])],
                        [dict(params=[4, 4]), dict(params=[3, 7])]):
            with self.assertRaises(ValueError):
                serialized_owners(invalid, contract)

    def test_saved_actor_changes_allow_one_head_tensor_but_no_backbone(self):
        changed_actor([], 0)
        changed_actor([HEAD_KEYS[0]], 8)
        for names, step in [([], 8), ([HEAD_KEYS[0]], 0), (['std'], 8)]:
            with self.assertRaises(ValueError):
                changed_actor(names, step)

    def test_general_transactions_at_first_middle_and_last_rejection(self):
        for commits in (0, 7, 31):
            proof = transactions(transaction_rows(commits, True), 'owner')
            self.assertEqual(proof['proposals'], commits + 1)
            self.assertEqual(len(proof['commits']), commits)
        self.assertIsNone(transactions(transaction_rows(32), 'owner')['rejected'])

    def test_transaction_corruption_and_extra_budget_rejected(self):
        for mode in ('owner', 'unpaired', 'live_write', 'scale', 'decision', 'gradient', 'extra'):
            rows = transaction_rows(1, True)
            if mode == 'owner':
                rows[0]['owner_sha256'] = 'other'
            elif mode == 'unpaired':
                rows.pop(1)
            elif mode == 'live_write':
                rows[-1]['parameters_and_optimizer_now_sha256'] = 'changed'
            elif mode == 'scale':
                rows[-1]['candidates'][0]['scale'] = .5
            elif mode == 'decision':
                rows[-1]['candidates'][0]['accepted'] = True
            elif mode == 'gradient':
                rows[-1]['head_gradients'] = {}
            else:
                rows = transaction_rows(33)
            with self.assertRaises(ValueError):
                transactions(rows, 'owner')

    def test_exact_call_budget_and_overflow(self):
        self.assertEqual(forward_accounting(8, 32, 32, 288), CALL_BUDGET)
        self.assertEqual(CALL_BUDGET['total'], 2148)
        with self.assertRaises(ValueError):
            forward_accounting(8, 32, 32, 289)

    def test_actual_checkpoint_callback_binds_saved_group_ids(self):
        trainer = head_trainer()
        optimizer = trainer.create_optimizer()
        optimizer.state_dict = lambda: dict(param_groups=[dict(params=[4, 8]), dict(params=[3, 7])])
        written = []

        class Base:
            def _save(self, args, state, **kwargs):
                self.saved_step = state.global_step

        tree = ast.parse((HERE / 'humanplus_head_trainer.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HeadCheckpointCallback']
        scope = dict(FiniteCheckpointCallback=Base, ownership=ownership, digest=digest,
            serialized_owners=serialized_owners, group_options=lambda opt: [],
            atomic_json=lambda path, data: written.append((path, data)), sha256=lambda path: 'checkpoint_sha')
        exec(compile(tree, '<actual D01 checkpoint callback>', 'exec'), scope)
        callback = scope['HeadCheckpointCallback']()
        callback.save_dir = Path('/nonexistent-d01-control-path')
        callback.saved_step = None
        callback._save(None, types.SimpleNamespace(global_step=2), model=trainer.model, optimizer=optimizer)
        path, record = written[0]
        self.assertEqual(str(path), '/nonexistent-d01-control-path/milestone_2/head_owners.json')
        self.assertEqual(record['serialized_id_names'][4], 'policy.' + HEAD_KEYS[0])
        self.assertEqual(record['step'], 2)
        self.assertEqual(record['checkpoint_sha256'], 'checkpoint_sha')

    def test_stop_auditor_has_no_tensor_load_or_fixed_hp12_counts(self):
        source = (HERE.parent / 'project_tools/humanplus_head_stop_audit.py').read_text()
        tree = ast.parse(source)
        self.assertNotIn('torch', {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names)
                             for n in ast.walk(tree)))
        self.assertNotIn('27', source)
        self.assertNotIn('28', source)
        self.assertIn('transactions(rows,', source)
        self.assertIn('head_stop[\'before_buffer_sha256\'] != head_stop[\'after_buffer_sha256\']', source)

    def test_independent_entry_and_audit_bindings_and_ast_abi(self):
        files = [HERE / n for n in ('humanplus_head.py', 'humanplus_head_entry.py', 'humanplus_head_verification.py')]
        files += [HERE.parent / 'project_tools' / n for n in ('humanplus_head_audit.py', 'humanplus_head_stop_audit.py')]
        for path in files:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('humanplus_finite', [n.name for n in node.names])
        source = (HERE / 'humanplus_head.py').read_text()
        self.assertIn('from script.a3.humanplus_head_entry import execute', source)
        self.assertNotIn('evaluation_base.OUT,', source)
        entry = ast.parse((HERE / 'humanplus_head_entry.py').read_text())
        for name in ('environment', 'finish'):
            function = next(n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertTrue(any(isinstance(n, ast.ImportFrom) and
                any(a.name == 'humanplus_head' and a.asname == 'trial' for a in n.names)
                for n in ast.walk(function)))


if __name__ == '__main__':
    unittest.main()
