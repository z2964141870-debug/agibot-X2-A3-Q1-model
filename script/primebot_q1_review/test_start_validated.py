"""A failed validation or failed media check cannot launch fresh training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import start_validated


class SerialLaunch(unittest.TestCase):
    def exercise(self, validation_status, media_passed):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation"
            validation.mkdir()
            (validation / "pipeline.json").write_text(json.dumps(dict(status=validation_status)))
            config = root / "config.json"
            config.write_text(json.dumps(dict(graphics_dir=str(root), primebot_python="python", xml="q1.xml")))
            smoke = root / "smoke.json"
            smoke.write_text("{}")
            output = root / "new_launch"

            def media(*args, **kwargs):
                if media_passed:
                    result = output / "render_saved/media"
                    result.mkdir(parents=True)
                    (result / "result.json").write_text(json.dumps(dict(status="PASSED", videos=[{}, {}, {}])))
                return media_passed

            def train(*args):
                result = output / "formal"
                result.mkdir()
                (result / "pipeline.json").write_text("{}")

            with patch.object(start_validated, "phase", side_effect=media) as render, \
                    patch.object(start_validated, "run", side_effect=train) as formal, \
                    patch.dict(start_validated.os.environ):
                if validation_status != "VALIDATED_READY_FOR_FRESH_FORMAL" or not media_passed:
                    with self.assertRaises(RuntimeError):
                        start_validated.execute(config, validation, smoke, output, "a" * 40, 1)
                    self.assertEqual(formal.call_count, 0)
                    self.assertEqual(json.loads((output / "pipeline.json").read_text())["status"], "FAILED")
                else:
                    start_validated.execute(config, validation, smoke, output, "a" * 40, 1)
                    self.assertEqual(render.call_count, 1)
                    self.assertEqual(formal.call_count, 1)
                    self.assertEqual(formal.call_args.args[3], output / "formal")

    def test_failed_validation_never_launches(self):
        self.exercise("FAILED", True)

    def test_failed_video_check_never_launches(self):
        self.exercise("VALIDATED_READY_FOR_FRESH_FORMAL", False)

    def test_success_starts_exactly_one_fresh_formal(self):
        self.exercise("VALIDATED_READY_FOR_FRESH_FORMAL", True)


if __name__ == "__main__":
    unittest.main()
