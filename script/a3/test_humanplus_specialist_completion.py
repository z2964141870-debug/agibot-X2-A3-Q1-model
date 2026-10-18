"""Completion is evidence adoption, never training or repeated source replay."""

import ast
from pathlib import Path
import sys
import unittest


class CompletionControls(unittest.TestCase):
    def test_no_trainer_dependency_in_saved_verifier(self):
        from script.a3 import humanplus_saved_probability
        path = Path(__file__).with_name('humanplus_specialist_completion_verification.py')
        tree = ast.parse(path.read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertIn('script.a3.humanplus_saved_probability', imports)
        self.assertFalse(any('trainer' in str(n) for n in imports))
        self.assertNotIn('gear_sonic', sys.modules)

    def test_no_training_or_baseline_stage(self):
        from script.a3 import humanplus_specialist_completion as trial
        self.assertEqual(trial.STAGES, ('verify-training', 'isaac-final', 'mujoco', 'effect'))
        self.assertTrue(trial.NAME.endswith('_HP22'))
        self.assertIn('HP21', str(trial.RUN))
        self.assertTrue(trial.SOURCE_NAME.endswith('_HP21'))
        self.assertEqual(trial.checkpoint(256).name, 'model_step_000256.pt')
        with self.assertRaises(ValueError):
            trial.checkpoint(8)

    def test_verifier_matches_completed_budget(self):
        path = Path(__file__).with_name('humanplus_specialist_completion_verification.py')
        text = path.read_text()
        self.assertIn("trial.SOURCE_LOG / 'train'", text)
        self.assertIn('trial.require_source()', text)
        self.assertIn('MINIBATCHES * step', text)
        self.assertIn("source[kind]", text)


if __name__ == '__main__':
    unittest.main()
