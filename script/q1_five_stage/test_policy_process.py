"""Bounded policy transport, error propagation, and child cleanup without Torch."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from policy_process import PolicyProcess


class PolicyProcessTests(unittest.TestCase):
    def test_float_roundtrip_and_graceful_reap(self):
        code = """import json,sys
print(json.dumps(dict(ok=True,value=dict(cuda_initialized=False))),flush=True)
for line in sys.stdin:
 value=json.loads(line)
 print(json.dumps(dict(ok=True,value=value)),flush=True)
 if value.get('op')=='quit': break
"""
        with tempfile.TemporaryDirectory() as directory:
            worker = PolicyProcess([sys.executable, "-u", "-c", code], Path(directory) / "worker.log", timeout=2)
            try:
                value = dict(op="echo", state=[0., -0., 1.2345678901234, 1e-30], text="a\nb")
                self.assertEqual(worker.request(value), value)
                with self.assertRaises(ValueError):
                    worker.request(dict(state=[float("nan")]))
            finally:
                worker.close()
            self.assertTrue(worker.closed)
            self.assertEqual(worker.child.returncode, 0)
            self.assertTrue(worker.child.stdin.closed and worker.child.stdout.closed)

    def test_worker_error_stops_without_restart(self):
        code = """import json,sys
print(json.dumps(dict(ok=True,value=dict(cuda_initialized=False))),flush=True)
sys.stdin.readline()
print(json.dumps(dict(ok=False,error='reference identity changed')),flush=True)
"""
        with tempfile.TemporaryDirectory() as directory:
            worker = PolicyProcess([sys.executable, "-u", "-c", code], Path(directory) / "worker.log", timeout=2)
            try:
                with self.assertRaisesRegex(RuntimeError, "reference identity changed"):
                    worker.request(dict(op="action"))
            finally:
                worker.close(graceful=False)
            self.assertIsNotNone(worker.child.poll())

    def test_startup_timeout_reaps_the_same_child(self):
        children = []
        original = subprocess.Popen

        def start(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child

        with tempfile.TemporaryDirectory() as directory, patch("policy_process.subprocess.Popen", side_effect=start):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                PolicyProcess([sys.executable, "-c", "import time; time.sleep(10)"],
                              Path(directory) / "worker.log", timeout=.1)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())

    def test_exit_or_non_json_output_is_not_a_policy_response(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, code in enumerate(("pass", "print('wrong output')")):
                with self.subTest(code=code), self.assertRaises((RuntimeError, json.JSONDecodeError)):
                    PolicyProcess([sys.executable, "-u", "-c", code], Path(directory) / f"worker{i}.log", timeout=2)


if __name__ == "__main__":
    unittest.main()
