"""Check callback cleanup ordering without importing simulation packages."""

import unittest
from unittest.mock import Mock

from script.a3.humanplus_physx_callback_validation import ContextClosingApp


class CallbackCleanupTests(unittest.TestCase):
    def test_unsubscribe_finishes_before_native_close(self):
        calls = []
        app = Mock()
        app.close.side_effect = lambda: calls.append('native_close')
        closer = ContextClosingApp(app, lambda: calls.append('clear_instance'),
                                   lambda phase: calls.append(phase))
        closer.close()
        self.assertEqual(calls, ['context_clear_begin', 'clear_instance',
                                 'context_clear_returned', 'native_close'])

    def test_cleanup_error_cannot_claim_native_close(self):
        app, cleanup = Mock(), Mock(side_effect=RuntimeError('unsubscribe failed'))
        with self.assertRaisesRegex(RuntimeError, 'unsubscribe failed'):
            ContextClosingApp(app, cleanup, lambda phase: None).close()
        app.close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
