"""Safety and exception-boundary evidence checks; no torch or CUDA import."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from script.a3.phase_trace import PhaseTrace, summarize_files
from script.diagnostics.gpu_operator_split import allowed


class SplitTests(unittest.TestCase):
    def test_original_boot_and_single_consumption(self):
        job = dict(boot_id="original")
        self.assertTrue(allowed(dict(status="READY"),job,"original"))
        self.assertFalse(allowed(dict(status="READY"),job,"new"))
        for status in ("RUNNING","OBSERVING","STOPPED","STOPPED_REBOOT","COMPLETED"):
            self.assertFalse(allowed(dict(status=status),job,"original"))

    def test_failure_preserves_exact_boundary_without_hidden_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path,"read_text",return_value="boot"):
                trace = PhaseTrace(directory,sync=lambda: self.fail("hidden synchronization"))
            with self.assertRaisesRegex(RuntimeError,"mock CUDA error"):
                with trace.span("MATMUL",cuda_sync=False):
                    raise RuntimeError("mock CUDA error")
            summary = summarize_files(directory)
            self.assertEqual(summary["incomplete"][0]["begin"]["phase"],"MATMUL")
            self.assertEqual(summary["incomplete"][0]["last_event"]["event"],"ERROR")
            self.assertIsNone(summary["last_completed"])


if __name__ == "__main__":
    unittest.main()
