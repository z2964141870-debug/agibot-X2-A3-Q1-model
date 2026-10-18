"""Standard-library tests of the new host adapter, without model or simulation."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from script.a3.humanplus_training_entry import adapted_tree, validate_training_receipts

SOURCE = '''
def main(config):
    args_cli.kit_args = "--/log/fileLogLevel=error"
    env = create_manager_env(config, device, args_cli)
    if replay:
        os._exit(0)
    trainer.train()
    if simulator_type == "IsaacSim":
        os._exit(0)
'''


class AdapterTests(unittest.TestCase):
    def test_only_storage_logging_and_terminal_exit_change(self):
        code = ast.unparse(adapted_tree(SOURCE, 'fixture.py'))
        self.assertEqual(code.count('os._exit(0)'), 1)
        self.assertIn('trainer.train()', code)
        self.assertIn('humanplus_finish(simulation_app, trainer)', code)
        self.assertIn('humanplus_environment(create_manager_env, config, device, args_cli)', code)
        self.assertIn('fileLogLevel=info', code)

    def test_changed_tail_rejected(self):
        with self.assertRaises(ValueError):
            adapted_tree(SOURCE + '    cleanup()\n', 'fixture.py')

    def test_missing_environment_rejected(self):
        with self.assertRaises(ValueError):
            adapted_tree(SOURCE.replace('create_manager_env', 'different_env'), 'fixture.py')

    def test_completion_requires_exact_coverage(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            rows = [dict(update=i, transitions=384, routes=dict(g1=192, a3_fast=192)) for i in range(1, 9)]
            records = {'finite_contract.json': dict(proposals=32, committed=32, rows=[{}] * 32, evaluations=100),
                'reference_route_coverage.json': dict(rollouts=rows),
                'cloth_contract.json': dict(optimizer_steps=[{}] * 32)}
            for name, value in records.items():
                (directory / name).write_text(json.dumps(value))
            self.assertEqual(validate_training_receipts(directory)['sampled_transitions'], 3072)
            for ordinal in range(8):
                rows[ordinal]['transitions'] = 383
                (directory / 'reference_route_coverage.json').write_text(json.dumps(dict(rollouts=rows)))
                with self.assertRaises(ValueError):
                    validate_training_receipts(directory)
                rows[ordinal]['transitions'] = 384


if __name__ == '__main__':
    unittest.main()
