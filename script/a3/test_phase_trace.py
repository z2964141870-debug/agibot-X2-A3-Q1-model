"""Durability, synchronization ordering and diagnostic entry tests without CUDA."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script.a3.phase_trace import PhaseTrace, summarize_files
from script.a3.traced_entry import EntryTransformer


class PhaseTests(unittest.TestCase):
    def make(self, directory, sync):
        with patch.object(Path, "read_text", return_value="unit-test-boot"):
            return PhaseTrace(directory, sync=sync, context=lambda:dict(trainer_global_step=1))

    def test_sync_and_metadata_override(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make(directory, lambda: calls.append("sync") or True)
            with logger.span("BACKWARD", at_utc="duplicate"):
                calls.append("operation")
            rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
            self.assertEqual(calls, ["sync", "operation", "sync"])
            self.assertEqual([r["event"] for r in rows], ["BEGIN", "PRE_SYNC_BEGIN", "PRE_SYNC_END",
                "CALL_RETURNED", "POST_SYNC_BEGIN", "POST_SYNC_END", "END"])
            self.assertTrue(all(r["trainer_global_step"] == 1 and r["boot_id"] == "unit-test-boot" for r in rows))
            self.assertNotEqual(rows[0]["at_utc"], "duplicate")
            self.assertEqual(summarize_files(directory)["incomplete"], [])

    def test_error_keeps_unfinished_and_no_false_end(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make(directory, lambda: True)
            with self.assertRaises(ValueError):
                with logger.span("OPTIMIZER"):
                    raise ValueError("failure")
            result = summarize_files(directory)
            self.assertEqual(result["last_completed"], None)
            self.assertEqual(result["incomplete"][0]["last_event"]["event"], "ERROR")

    def test_sync_failure_cannot_claim_completion(self):
        count = 0
        def sync():
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("CUDA simulation")
            return True
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make(directory, sync)
            with self.assertRaises(RuntimeError):
                with logger.span("FORWARD"):
                    pass
            rows = [json.loads(x) for x in logger.path.read_text().splitlines()]
            self.assertIn("CALL_RETURNED", [r["event"] for r in rows])
            self.assertNotIn("END", [r["event"] for r in rows])

    def test_nested_span_and_partial_record(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = self.make(directory, lambda: False)
            with logger.span("PARENT", cuda_sync=False):
                with logger.span("CHILD", cuda_sync=False):
                    pass
            with logger.path.open("a") as stream:
                stream.write('{"unfinished"')
            result = summarize_files(directory)
            self.assertEqual(result["last_completed"]["phase"], "PARENT")
            self.assertEqual(len(result["partial_lines"]), 1)

    def test_entry_ast_uses_original_function_result(self):
        tree = EntryTransformer().visit(ast.parse("def create_manager_env(x):\n    return x+1\n"))
        calls = []
        def decorate(phase):
            def decorator(function):
                def wrapped(*args):
                    calls.append(phase)
                    return function(*args)
                return wrapped
            return decorator
        ns = {"_yuanqi_trace_decorate":decorate}
        exec(compile(ast.fix_missing_locations(tree), "test.py", "exec"), ns)
        self.assertEqual(ns["create_manager_env"](4), 5)
        self.assertEqual(calls, ["ENV_CREATE"])


if __name__ == "__main__":
    unittest.main()
