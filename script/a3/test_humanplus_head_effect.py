"""Exercise the actual checkpoint metadata contract without loading tensors."""

import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.a3 import humanplus_head_effect as trial
from script.a3 import evaluation_common

HERE = Path(__file__).parent


def function(file, name):
    return next(n for n in ast.parse((HERE / file).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class EffectControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_bundle_satisfies_real_metadata_without_alias_resolution(self):
        data = self.root / 'data'
        data.mkdir()
        model = data / 'model_step_000008.pt'
        model.write_bytes(b'fixture only, not a tensor')
        config = data / 'official_config.yaml'
        config.write_text('fixture: true\n')
        sidecar = model.with_suffix('.json')
        record = dict(step=8, bytes=model.stat().st_size, sha256=trial.sha256(model))
        sidecar.write_text(json.dumps(record))
        bundle = data / 'bundle'
        target = trial.materialize_bundle(bundle, model, config, sidecar)
        self.assertEqual(target.stat().st_ino, model.stat().st_ino)
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.resolve().parent, bundle)
        self.assertFalse(model.with_name('config.yaml').exists())
        with patch.object(evaluation_common, 'ROOT', self.root):
            self.assertEqual(evaluation_common.checkpoint_metadata(target), record)
        self.assertEqual(trial.sha256(bundle / 'config.yaml'), trial.sha256(config))
        with self.assertRaises(FileExistsError):
            trial.materialize_bundle(bundle, model, config, sidecar)

    def test_bad_or_early_tensor_identity_rejected_before_bundle_created(self):
        model = self.root / 'model.pt'
        model.write_bytes(b'fixture')
        sidecar = model.with_suffix('.json')
        for record in (dict(step=4, bytes=7, sha256=trial.sha256(model)),
                       dict(step=8, bytes=7, sha256='changed')):
            sidecar.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                trial.materialize_bundle(self.root / 'new', model, self.root / 'unused', sidecar)
            self.assertFalse((self.root / 'new').exists())

    def source_fixture(self):
        root = self.root
        source = SimpleNamespace(NAME='a3_humanplus_head_assessment_20261018_HP15',
            OUT=root / 'out', LOG=root / 'log', PLAN=root / 'source_plan.json',
            RESULT=root / 'source_result.json', checked=lambda: {'input_sha256': {}})
        (source.OUT / 'cloth').mkdir(parents=True)
        (source.LOG / 'cloth').mkdir(parents=True)
        def save(path, data):
            path.write_text(json.dumps(data))
        save(source.PLAN, {})
        save(source.RESULT, dict(status='STOPPED', current_stage='cloth', reason='RuntimeError:WORKER_EXIT:1',
            stages=[dict(stage='verify-training', state=dict(status='COMPLETE', returncode=0))]))
        audit = source.PLAN.with_name(source.NAME + '_verify-training_audit.json')
        save(audit, dict(verified=True, plan_sha256=trial.sha256(source.PLAN), artifacts={}))
        save(source.PLAN.with_name(source.NAME + '_cloth_audit.json'), dict(verified=False,
            after_issues=[], remaining_own_group_processes=[], result_sha256=trial.sha256(source.RESULT)))
        save(source.OUT / 'training_verification.json', dict(status='COMPLETE',
            scope='VERIFIED_HEAD_TRAINING_NOT_EFFECT', optimizer_steps=32, updates=8, final_effect_step=8))
        (source.LOG / 'cloth/console.log').write_text('Traceback (most recent call last):\n'
            'ValueError: Missing matching checkpoint config.yaml\n')
        for name, value in (('source', source), ('ROOT', root)):
            p = patch.object(trial, name, value)
            p.start()
            self.addCleanup(p.stop)
        return source, audit

    def test_completed_model_verification_and_zero_replay_eligible(self):
        self.source_fixture()
        _, identities = trial.source_evidence()
        self.assertIn('source_result.json', identities)

    def test_any_prior_policy_replay_is_rejected(self):
        source, _ = self.source_fixture()
        for path in ('explicit_manifest.json', 'motion.metrics.json'):
            file = source.OUT / 'cloth' / path
            file.write_text('{}')
            with self.assertRaises(ValueError):
                trial.source_evidence()
            file.unlink()

    def test_unverified_model_never_eligible(self):
        _, audit = self.source_fixture()
        record = json.loads(audit.read_text())
        record['verified'] = False
        audit.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            trial.source_evidence()

    def test_replay_and_effect_formula_unchanged_no_training_or_verify_stage(self):
        for name in ('evaluate', 'verify_replays', 'effect'):
            self.assertEqual(ast.dump(function('humanplus_head_assessment.py', name), include_attributes=False),
                ast.dump(function('humanplus_head_effect.py', name), include_attributes=False))
        self.assertEqual(trial.STAGES, ('cloth', 'official', 'effect'))
        with self.assertRaises(ValueError):
            trial.checkpoint(4)
        compile((HERE / 'humanplus_head_effect.py').read_text(), 'effect.py', 'exec')


if __name__ == '__main__':
    unittest.main()
