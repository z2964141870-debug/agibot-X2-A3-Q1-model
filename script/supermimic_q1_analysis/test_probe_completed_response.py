"""Check process-local constructor binding is restored even when execution fails."""

from types import SimpleNamespace
import unittest

from probe_completed_response import execute_bound


class ConstructorBindingTests(unittest.TestCase):
    def test_success_restores_constructor(self):
        old, corrected = object(), object()
        observed = []
        delegate = SimpleNamespace(Q1Assets=old)
        delegate.execute = lambda args: observed.append((delegate.Q1Assets, args))
        execute_bound(delegate, corrected, "args")
        self.assertEqual(observed, [(corrected, "args")])
        self.assertIs(delegate.Q1Assets, old)

    def test_error_restores_constructor(self):
        old, corrected = object(), object()
        delegate = SimpleNamespace(Q1Assets=old)
        def fail(args):
            self.assertIs(delegate.Q1Assets, corrected)
            raise RuntimeError("fixture")
        delegate.execute = fail
        with self.assertRaises(RuntimeError):
            execute_bound(delegate, corrected, None)
        self.assertIs(delegate.Q1Assets, old)


if __name__ == "__main__":
    unittest.main()
