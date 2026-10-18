"""Standard-library lifecycle controls; never imports Isaac or torch."""

import unittest
from unittest.mock import Mock

from script.a3.humanplus_physx_exit_validation import run_control


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = Mock()
        self.sim = Mock(device='cuda:0')
        self.sim.get_physics_dt.return_value = .01
        self.events = []
        self.saved = []

    def run_case(self, synchronize=lambda: None):
        return run_control(self.app, lambda: self.sim, synchronize,
            lambda phase, **fields: self.events.append((phase, fields)), self.saved.append)

    def test_steps_and_shutdown_both_required(self):
        result = self.run_case()
        self.assertEqual(self.sim.step.call_count, 10)
        self.sim.stop.assert_not_called()
        self.sim.clear.assert_not_called()
        self.app.close.assert_called_once_with()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual([fields['steps'] for phase, fields in self.events
                          if phase == 'step_returned'], list(range(1, 11)))
        self.assertLess([p for p, _ in self.events].index('control_complete'),
                        [p for p, _ in self.events].index('app_close_begin'))

    def test_step_failure_does_not_claim_control_complete(self):
        self.sim.step.side_effect = RuntimeError('step failure')
        with self.assertRaisesRegex(RuntimeError, 'step failure'):
            self.run_case()
        self.assertEqual(self.saved, [])
        self.app.close.assert_called_once_with()

    def test_shutdown_failure_keeps_separate_control_receipt(self):
        self.app.close.side_effect = RuntimeError('shutdown failure')
        with self.assertRaisesRegex(RuntimeError, 'shutdown failure'):
            self.run_case()
        self.assertEqual(self.saved[0]['steps'], 10)
        self.assertEqual(self.saved[0]['status'], 'CONTROL_COMPLETE')
        self.assertNotIn('app_close_returned', [p for p, _ in self.events])

    def test_cuda_failure_does_not_claim_control_complete(self):
        def failed_sync():
            raise RuntimeError('CUDA failure')
        with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
            self.run_case(failed_sync)
        self.assertEqual(self.saved, [])
        self.app.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
