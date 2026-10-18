"""Standard-library controls for provenance and the saved-only import repair."""

import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.a3 import humanplus_head_assessment as trial
from script.a3 import humanplus_saved_probability
from script.a3.humanplus_kit_log import SHUTDOWN, binding

HERE = Path(__file__).parent


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class AssessmentControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name).resolve()
        self.root = root
        self.source = SimpleNamespace(NAME='a3_humanplus_head_logged_20261018_HP14',
            OUT=root / 'out', LOG=root / 'log', PLAN=root / 'source_plan.json',
            RESULT=root / 'source_result.json', checked=lambda: {'input_sha256': {}})
        self.source.OUT.mkdir()
        (self.source.LOG / 'train').mkdir(parents=True)
        (self.source.LOG / 'verify-training').mkdir()
        self.save(self.source.PLAN, {})
        self.result = dict(status='STOPPED', current_stage='verify-training',
            reason='RuntimeError:WORKER_EXIT:1',
            stages=[dict(stage='train', state=dict(status='COMPLETE', returncode=0))])
        self.save(self.source.RESULT, self.result)
        self.save(self.source.PLAN.with_name(self.source.NAME + '_train_audit.json'),
            dict(verified=True, plan_sha256=trial.sha256(self.source.PLAN), artifacts={}))
        self.stop_audit = self.source.PLAN.with_name(self.source.NAME + '_verify-training_audit.json')
        self.refresh_stop()
        (self.source.LOG / 'verify-training/console.log').write_text(
            'Traceback (most recent call last):\n'
            'from script.a3.cloth_constrained_trainer import initial_probability\n'
            "ModuleNotFoundError: No module named 'gear_sonic'\n")
        directory = self.source.LOG / 'train'
        settings = {'/log/file': str(directory / 'kit.log'), '/log/level': 'info', '/log/fileLogLevel': 'info'}
        self.save(directory / 'kit_binding.json', binding(directory, settings, 123))
        (directory / 'kit.log').write_text('\n'.join(SHUTDOWN))
        from script.a3.humanplus_kit_log import verify_log
        self.save(directory / 'native_exit.json', dict(returncode=0, native_pid=123))
        self.save(directory / 'result.json', dict(native_shutdown_verified=True,
            optimizer_commits=32, updates=8, sampled_transitions=3072, **verify_log(directory, 123)))
        for name, value in (('source', self.source), ('ROOT', root), ('SOURCE_LOG', self.source.LOG)):
            patcher = patch.object(trial, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def refresh_stop(self):
        self.save(self.stop_audit, dict(verified=False, after_issues=[], remaining_own_group_processes=[],
            result_sha256=trial.sha256(self.source.RESULT)))

    def test_exact_completed_train_and_preload_import_error_eligible(self):
        plan, identities = trial.source_evidence()
        self.assertIn('source_result.json', identities)

    def test_scientific_stop_or_other_trial_never_eligible(self):
        for mutate in (lambda: self.result.update(current_stage='train'),
                       lambda: self.result.update(reason='CandidateRejected'),
                       lambda: setattr(self.source, 'NAME', 'HP13')):
            before = copy.deepcopy(self.result)
            name = self.source.NAME
            mutate()
            self.save(self.source.RESULT, self.result)
            with self.assertRaises(ValueError):
                trial.source_evidence()
            self.result = before
            self.source.NAME = name

    def test_partial_training_or_health_failure_rejected(self):
        self.result['stages'][0]['state']['status'] = 'STOPPED'
        self.save(self.source.RESULT, self.result)
        with self.assertRaises(ValueError):
            trial.source_evidence()
        self.result['stages'][0]['state']['status'] = 'COMPLETE'
        self.save(self.source.RESULT, self.result)
        self.refresh_stop()
        record = json.loads(self.stop_audit.read_text())
        record['after_issues'] = ['machine fault']
        self.save(self.stop_audit, record)
        with self.assertRaises(ValueError):
            trial.source_evidence()

    def test_probability_math_and_thresholds_are_identical(self):
        for name, file in (('compare', 'likelihood_probe.py'),
                           ('initial_probability', 'cloth_constrained_trainer.py')):
            old, new = function(HERE / file, name), function(HERE / 'humanplus_saved_probability.py', name)
            new.body = [n for n in new.body if not isinstance(n, ast.Import)]
            self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))
        self.assertNotIn('torch', humanplus_saved_probability.__dict__)

    def test_effect_protocol_and_saved_optimizer_check_unchanged(self):
        for name in ('evaluate', 'verify_replays', 'effect'):
            self.assertEqual(ast.dump(function(HERE / 'humanplus_head_logged.py', name), include_attributes=False),
                ast.dump(function(HERE / 'humanplus_head_assessment.py', name), include_attributes=False))
        self.assertEqual(ast.dump(function(HERE / 'humanplus_head_logged_verification.py', 'verify_optimizer'), include_attributes=False),
            ast.dump(function(HERE / 'humanplus_head_assessment_verification.py', 'verify_optimizer'), include_attributes=False))

    def test_assessment_has_no_training_or_native_entry_and_valid_handshake(self):
        self.assertEqual(trial.STAGES, ('verify-training', 'cloth', 'official', 'effect'))
        self.assertFalse(hasattr(trial, 'native'))
        tree = ast.parse((HERE / 'humanplus_head_assessment.py').read_text())
        self.assertEqual(sum(isinstance(n, ast.If) for n in tree.body), 1)
        self.assertIn('READY\n', [n.value for n in ast.walk(function(HERE / 'humanplus_head_assessment.py', 'worker'))
                                if isinstance(n, ast.Constant)])


if __name__ == '__main__':
    unittest.main()
