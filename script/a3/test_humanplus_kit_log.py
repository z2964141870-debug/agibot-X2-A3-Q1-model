"""Controls for the HP13 log-component confusion and explicit HP14 binding."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from script.a3.humanplus_kit_log import (
    SHUTDOWN, binding, logged_tree, log_path, require_fresh, verify_log,
)

SOURCE = '''
def main():
    args_cli.kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    env = create_manager_env(config, device, args_cli)
    if simulator_type == "IsaacSim":
        os._exit(0)
'''


class LogControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()

    def saved(self, pid=123):
        target = log_path(self.directory)
        values = {'/log/file': str(target), '/log/level': 'Info', '/log/fileLogLevel': 'info'}
        record = binding(self.directory, values, pid)
        (self.directory / 'kit_binding.json').write_text(json.dumps(record))
        target.write_text('\n'.join(SHUTDOWN))
        return target, record

    def test_real_adapter_changes_only_logging_setting_site(self):
        tree = logged_tree(SOURCE, 'vendor.py', self.directory)
        compile(tree, 'vendor.py', 'exec')
        scope = {}
        class Args:
            pass
        args = Args()
        scope.update(args_cli=args, config=1, device=2, simulator_type='IsaacSim',
                     create_manager_env=object(), humanplus_environment=lambda *a: 'environment',
                     simulation_app=object(), trainer=object(), humanplus_finish=lambda *a: None)
        exec(compile(tree, 'vendor.py', 'exec'), scope)
        scope['main']()
        self.assertEqual(args.kit_args, '--/log/level=info --/log/fileLogLevel=info '
                         '--/log/outputStreamLevel=error --/log/file=' + str(self.directory / 'kit.log'))

    def test_changed_or_duplicate_vendor_setting_rejected(self):
        for source in (SOURCE.replace('/log/level=error', '/log/level=warning'),
                       SOURCE.replace('    env =', '    args_cli.kit_args = "other"\n    env =')):
            with self.assertRaises(ValueError):
                logged_tree(source, 'vendor.py', self.directory)

    def test_explicit_log_passes_without_any_console_announcement(self):
        target, record = self.saved()
        (self.directory / 'native_console.log').write_text(
            'Logging to file: /tmp/isaaclab/logs/isaaclab_other.log\n')
        result = verify_log(self.directory, 123)
        self.assertEqual(result['kit_log'], str(target))
        self.assertEqual(result['kit_binding'], record)

    def test_native_pid_and_live_settings_must_match(self):
        self.saved()
        with self.assertRaises(ValueError):
            verify_log(self.directory, 124)
        with self.assertRaises(ValueError):
            binding(self.directory, {'/log/file': '/tmp/isaaclab/logs/other.log',
                                    '/log/level': 'info', '/log/fileLogLevel': 'info'}, 123)

    def test_error_level_cannot_hide_shutdown(self):
        for key in ('/log/level', '/log/fileLogLevel'):
            values = {'/log/file': str(self.directory / 'kit.log'),
                      '/log/level': 'info', '/log/fileLogLevel': 'info'}
            values[key] = 'error'
            with self.assertRaises(ValueError):
                binding(self.directory, values, 123)

    def test_missing_or_unordered_shutdown_rejected(self):
        target, _ = self.saved()
        for markers in (SHUTDOWN[:-1], SHUTDOWN[::-1]):
            target.write_text('\n'.join(markers))
            with self.assertRaises(ValueError):
                verify_log(self.directory, 123)

    def test_existing_and_symlink_log_rejected(self):
        target, _ = self.saved()
        with self.assertRaises(FileExistsError):
            require_fresh(self.directory)
        target.unlink()
        target.symlink_to(self.directory / 'absent')
        with self.assertRaises(ValueError):
            log_path(self.directory)

    def test_independent_trial_bindings_and_unchanged_science(self):
        here = Path(__file__).parent
        before = ast.parse((here / 'humanplus_head.py').read_text())
        after = ast.parse((here / 'humanplus_head_logged.py').read_text())
        functions = ('launch_arguments', 'evaluate', 'verify_replays', 'effect')
        for name in functions:
            left = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False))
        for name in ('humanplus_head_logged_entry.py', 'humanplus_head_logged_verification.py'):
            source = (here / name).read_text()
            self.assertIn('humanplus_head_logged as trial', source)
            self.assertNotIn('humanplus_head as trial', source)


if __name__ == '__main__':
    unittest.main()
