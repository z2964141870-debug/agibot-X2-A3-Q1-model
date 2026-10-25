"""Check that baseline failure stops later training rather than retrying it."""

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pipeline_baseline as pipeline


class PipelineTests(unittest.TestCase):
    def test_failed_stage_is_recorded_without_starting_the_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            output = root / "run"
            output.mkdir()
            intake = root / "intake.json"
            backend = root / "backend.json"
            intake.write_text("{}\n")
            backend.write_text("{}\n")
            marker = root / "later_started"
            args = Namespace(root=root, output=output, intake=intake, backend_receipt=backend,
                             code_commit="1" * 40, expected_oom_kill=14)
            phases = [{"name": "broken_prepare", "output": str(root / "first"),
                       "command": [sys.executable, "-c", "raise SystemExit(7)"]},
                      {"name": "should_not_train", "output": str(root / "second"),
                       "command": [sys.executable, "-c",
                                   "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"]}]
            with patch.object(pipeline, "commands", return_value=(phases, root / "lib", root / "train")), \
                    patch.object(pipeline, "process_identity", return_value={"pid": 1, "start_ticks": 1}):
                with self.assertRaisesRegex(RuntimeError, "no automatic retry"):
                    pipeline.execute(args)
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["status"], "FAILED")
            self.assertEqual(len(state["events"]), 1)
            self.assertEqual(state["events"][0]["exit_code"], 7)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
