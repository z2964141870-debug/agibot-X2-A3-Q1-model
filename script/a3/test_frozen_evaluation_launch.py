"""Exercise the complete evaluation launch without loading policy or CUDA."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from script.a3 import evaluate_mujoco as evaluation
from script.a3.frozen_evaluation_launch import LaunchProxy, arguments


class FrozenLaunchTest(unittest.TestCase):
    def test_complete_evaluator_launch_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            checkpoint = root / 'data/model.pt'
            checkpoint.write_bytes(b'FAKE_TEST_CHECKPOINT_NO_MODEL')
            reference = root / 'data/unique.csv'
            reference.write_text('Frame\n0\n')
            calls = []

            def spawn(argv, **kwargs):
                calls.append((argv, kwargs))
                metric = Path(argv[argv.index('--metrics-out') + 1])
                trace = Path(argv[argv.index('--timeseries-out') + 1])
                metric.write_text(json.dumps(dict(fps=50, motions=[dict(num_policy_steps=1,
                    tracking=dict(all_29_rmse=0.), fall=False)])))
                trace.write_text('{}')
                return type('FakeChild', (), dict(wait=lambda self:0))()

            proxy = LaunchProxy(subprocess, spawn)
            args = arguments(root / 'data/evaluation', root / 'logs/evaluation', checkpoint, reference)
            with patch.object(evaluation, 'ROOT', root), patch.object(evaluation, 'subprocess', proxy), \
                    patch.object(evaluation, 'checkpoint_metadata', return_value=dict(sha256='FAKE_TEST_SHA')), \
                    patch.object(evaluation, 'preflight', return_value=dict(boot_id='FAKE_TEST_BOOT')), \
                    patch.object(evaluation, 'monitor_child', side_effect=lambda child, directory, config:(child.wait(), None)), \
                    patch.object(evaluation, 'summarize_explicit') as summarize:
                self.assertEqual(evaluation.evaluate_explicit(args), 0)
                summarize.assert_called_once_with(args.output)
            self.assertEqual(len(calls), 1)
            argv, kwargs = calls[0]
            self.assertIn('script.a3.smpl_frozen_entry', argv)
            self.assertNotIn('script.a3.sim_entry', argv)
            self.assertFalse(kwargs['start_new_session'])
            self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
            protocol = json.loads((args.output / 'explicit_manifest.json').read_text())['protocol']
            self.assertEqual(protocol['frame_stride'], 1)
            self.assertIsNone(protocol['max_policy_steps'])
            self.assertEqual(protocol['action_delay_ms'], 0.)

    def test_subprocess_constants_remain_available(self):
        proxy = LaunchProxy(subprocess)
        self.assertEqual(proxy.STDOUT, subprocess.STDOUT)
        self.assertIs(proxy.TimeoutExpired, subprocess.TimeoutExpired)


if __name__ == '__main__':
    unittest.main()
