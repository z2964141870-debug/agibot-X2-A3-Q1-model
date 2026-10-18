"""Native exit receipt acceptance uses no simulation packages."""

import json
from pathlib import Path
import tempfile
import unittest

from script.a3.humanplus_physx_native_validation import SHUTDOWN_MARKERS, validate_completion


class NativeReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'control.json').write_text(json.dumps(dict(status='CONTROL_COMPLETE',
            steps=10, device='cuda:0', dt=.01)))
        self.rows = [dict(phase='step_returned', steps=n) for n in range(1, 11)]
        self.rows += [dict(phase=phase) for phase in ('control_complete', 'app_close_begin',
                      'context_clear_returned', 'native_close_begin')]
        self.save_progress()

    def save_progress(self):
        (self.path / 'progress.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def test_only_zero_exit_plus_all_evidence_passes(self):
        receipt = validate_completion(self.path, 0, '\n'.join(SHUTDOWN_MARKERS))
        self.assertTrue(receipt['native_shutdown_verified'])
        self.assertFalse(receipt['shutdown_returned'])

    def test_native_error_never_passes(self):
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_EXIT:-11'):
            validate_completion(self.path, -11, '\n'.join(SHUTDOWN_MARKERS))

    def test_exit_zero_without_shutdown_log_rejected(self):
        with self.assertRaisesRegex(ValueError, 'shutdown evidence'):
            validate_completion(self.path, 0, '')

    def test_missing_step_rejected(self):
        self.rows.pop(4)
        self.save_progress()
        with self.assertRaisesRegex(ValueError, 'ten-step'):
            validate_completion(self.path, 0, '\n'.join(SHUTDOWN_MARKERS))

    def test_cleanup_after_close_rejected(self):
        self.rows[-2:] = reversed(self.rows[-2:])
        self.save_progress()
        with self.assertRaisesRegex(ValueError, 'shutdown evidence'):
            validate_completion(self.path, 0, '\n'.join(SHUTDOWN_MARKERS))


if __name__ == '__main__':
    unittest.main()
