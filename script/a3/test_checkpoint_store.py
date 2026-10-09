"""CPU fault tests; they do not simulate a physical power loss."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from script.a3.checkpoint_store import save_checkpoint, select_checkpoint


def payload(step):
    return {
        "policy_state_dict": {"layer.weight": torch.tensor([float(step)])},
        "value_state_dict": {"layer.weight": torch.tensor([float(step)])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(step)}}},
        "state": SimpleNamespace(global_step=step),
    }


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_roundtrip_and_retention(self):
        for step in [1, 2, 3]:
            save_checkpoint(self.directory, payload(step))
        path, loaded, failures = select_checkpoint(self.directory)
        self.assertEqual(loaded["state"].global_step, 3)
        self.assertEqual((self.directory / "last.pt").resolve(), path)
        self.assertFalse((self.directory / "model_step_000001.pt").exists())
        self.assertTrue((self.directory / "model_step_000002.pt").exists())
        self.assertEqual(failures, [])

    def test_failed_serialization_preserves_previous(self):
        save_checkpoint(self.directory, payload(1))
        with patch("script.a3.checkpoint_store.torch.save", side_effect=OSError("injected write failure")):
            with self.assertRaises(OSError):
                save_checkpoint(self.directory, payload(2))
        self.assertEqual(select_checkpoint(self.directory)[1]["state"].global_step, 1)

    def test_failed_fsync_preserves_previous(self):
        save_checkpoint(self.directory, payload(1))
        with patch("script.a3.checkpoint_store.os.fsync", side_effect=OSError("injected flush failure")):
            with self.assertRaises(OSError):
                save_checkpoint(self.directory, payload(2))
        self.assertEqual(select_checkpoint(self.directory)[1]["state"].global_step, 1)

    def test_corrupt_latest_falls_back(self):
        save_checkpoint(self.directory, payload(1))
        save_checkpoint(self.directory, payload(2))
        (self.directory / "model_step_000002.pt").write_bytes(b"")
        path, loaded, failures = select_checkpoint(self.directory)
        self.assertEqual(loaded["state"].global_step, 1)
        self.assertEqual(len(failures), 1)
        self.assertTrue(path.exists())
        save_checkpoint(self.directory, payload(3))
        self.assertTrue((self.directory / "model_step_000001.pt").exists())

    def test_corrupt_metadata_falls_back(self):
        save_checkpoint(self.directory, payload(1))
        save_checkpoint(self.directory, payload(2))
        (self.directory / "model_step_000002.json").write_text("{")
        self.assertEqual(select_checkpoint(self.directory)[1]["state"].global_step, 1)

    def test_nan_is_rejected(self):
        save_checkpoint(self.directory, payload(1))
        invalid = payload(2)
        invalid["policy_state_dict"]["layer.weight"][0] = float("nan")
        with self.assertRaises(ValueError):
            save_checkpoint(self.directory, invalid)
        self.assertEqual(select_checkpoint(self.directory)[1]["state"].global_step, 1)

    def test_repeated_step_is_not_overwritten(self):
        save_checkpoint(self.directory, payload(1))
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.directory, payload(1))

    def test_milestone_survives_retention(self):
        for step in [500, 501, 502, 503]:
            save_checkpoint(self.directory, payload(step))
        self.assertTrue((self.directory / "model_step_000500.pt").exists())
        self.assertFalse((self.directory / "model_step_000501.pt").exists())


if __name__ == "__main__":
    unittest.main()
