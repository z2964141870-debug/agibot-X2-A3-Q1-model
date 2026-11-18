import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from . import live_morph_bridge


class BridgeStartupTest(unittest.TestCase):
    def test_start_and_clean_stop_emit_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "cal.json").write_text(json.dumps({"id": "startup_test"}))
            argv = ["bridge", "--clotho-root", folder, "--calibration", str(root / "cal.json"),
                    "--output", str(root / "bridge.jsonl")]
            output = io.StringIO()
            with patch("sys.argv", argv), patch.object(live_morph_bridge, "SmplAdapter", return_value=Mock(skeleton_id="test")), patch("script.x2_morph_human.artifacts.home", return_value=root), patch.object(live_morph_bridge, "get_json", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(output):
                live_morph_bridge.main()
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([r["status"] for r in rows], ["FGP_MORPH_BRIDGE_READY", "FGP_MORPH_BRIDGE_STOPPED"])
            self.assertEqual(rows[-1]["sent"], 0)
