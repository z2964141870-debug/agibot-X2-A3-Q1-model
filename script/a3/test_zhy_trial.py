"""New-host memory accounting, launch ownership and stop-on-error contracts."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.a3 import zhy_trial as trial


class ZhyTrialTests(unittest.TestCase):
    def memory(self, usage, cache, rss=1, kills=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'memory.limit_in_bytes').write_text(str(16 * 1024**3))
            (root / 'memory.usage_in_bytes').write_text(str(usage * 1024**3))
            (root / 'memory.stat').write_text(
                f'total_inactive_file {cache * 1024**3}\ntotal_rss {rss * 1024**3}\n')
            (root / 'memory.oom_control').write_text(f'under_oom 0\noom_kill {kills}\n')
            return trial.container_memory(root)

    def test_reclaimable_cache_is_not_an_oom(self):
        value = self.memory(16, 14, rss=2)
        self.assertEqual(value['available_estimate_bytes'], 14 * 1024**3)
        self.assertEqual(value['rss_bytes'], 2 * 1024**3)

    def test_memory_pressure_is_not_hidden_by_host_memory(self):
        value = self.memory(14, 1, rss=13, kills=1)
        self.assertEqual(value['available_estimate_bytes'], 3 * 1024**3)
        self.assertEqual(value['oom_kill'], 1)

    def test_memory_estimate_is_bounded(self):
        self.assertEqual(self.memory(18, 1)['available_estimate_bytes'], 0)
        self.assertEqual(self.memory(1, 5)['available_estimate_bytes'], 16 * 1024**3)

    def test_only_ready_original_boot_can_launch(self):
        job = dict(boot_id='original')
        self.assertTrue(trial.launch_allowed(dict(status='READY'), job, 'original'))
        self.assertFalse(trial.launch_allowed(dict(status='READY'), job, 'new'))
        for status in ('RUNNING', 'STOPPED', 'TRAINING_COMPLETE'):
            self.assertFalse(trial.launch_allowed(dict(status=status), job, 'original'))

    def test_historical_checkpoint_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'escaped'):
            trial.choose_checkpoint([dict(run_dir='/tmp/a3_finetune_20261010/R09_001_s0')])

    def test_telemetry_failure_terminates_child(self):
        class Child:
            pid = 12345
            ended = False
            def poll(self):
                return -15 if self.ended else None
            def wait(self, timeout=None):
                self.ended = True
                return -15
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'registration.json').write_text(json.dumps(dict(kernel_baseline=[])))
            child = Child()
            with patch.object(trial, 'OUT', out), patch.object(trial, 'LOG', out), \
                 patch.object(trial, 'health', side_effect=OSError('sensor missing')), \
                 patch.object(trial.os, 'killpg') as kill:
                code, reason = trial.guarded_monitor(child, out, {})
            self.assertEqual((code, reason), (-15, 'TELEMETRY_UNREADABLE'))
            kill.assert_called_once_with(child.pid, trial.signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
