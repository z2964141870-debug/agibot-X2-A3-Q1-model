"""Check migration integrity and containment independently of the real dataset."""

import tempfile
from pathlib import Path
import unittest

from relocated_io import RelocatedView, identity


class RelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "ZHY"
        self.root.mkdir()
        self.root = self.root.resolve()
        self.file = self.root / "input.json"
        self.file.write_text('{"status": "UNCHANGED"}\n')
        self.old = Path("/old/q1/input.json")
        self.record = {**identity(self.file), "path": str(self.old)}
        self.view = RelocatedView(self.root, {"/old/q1": self.root})

    def test_rebind_retains_original_and_checks_bytes(self):
        actual = self.view.verify(self.record)
        self.assertEqual(actual, identity(self.file))
        self.assertEqual(self.record["path"], str(self.old))
        self.assertEqual(self.view.rebase(self.record), actual)

    def test_changed_content_is_rejected(self):
        self.file.write_text('{"status": "CORRUPTED"}\n')
        with self.assertRaises(ValueError):
            self.view.verify(self.record)

    def test_longest_prefix_and_other_project(self):
        special = self.root / "data"
        special.mkdir()
        view = RelocatedView(self.root, {"/old/q1": self.root, "/old/q1/input": special})
        self.assertEqual(view.path("/old/q1/input/item"), special / "item")
        self.assertEqual(view.path("/old/q1_other/item"), Path("/old/q1_other/item"))

    def test_escape_symlink_and_parent_are_rejected(self):
        (self.root / "escape").symlink_to(self.root.parent)
        for name in ("/old/q1/escape/out", "/old/q1/../out"):
            with self.assertRaises(ValueError):
                self.view.path(name)

    def test_rebase_does_not_modify_source(self):
        value = {"files": [self.record], "gate": False, "threshold": .15}
        result = self.view.rebase(value)
        self.assertEqual(result["files"][0], identity(self.file))
        self.assertEqual(value["files"][0]["path"], str(self.old))
        self.assertFalse(result["gate"])
        self.assertEqual(result["threshold"], .15)


if __name__ == "__main__":
    unittest.main()
