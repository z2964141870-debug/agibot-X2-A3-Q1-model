import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from .assets import audit, install


class AssetTests(unittest.TestCase):
    def check_bad_zip(self, rows, inventory, error=ValueError):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for key, data in rows:
                    archive.writestr(key, data)
            with self.assertRaises(error):
                audit(path, inventory)

    def test_missing_asset(self):
        self.check_bad_zip([], {"required.onnx": {"bytes": 1, "sha256": "x"}})

    def test_unknown_asset(self):
        self.check_bad_zip([("unknown.onnx", b"x")], {})

    def test_bad_sha(self):
        self.check_bad_zip([("model.onnx", b"x")], {"model.onnx": {"bytes": 1, "sha256": "x"}})

    def test_traversal(self):
        self.check_bad_zip([("../escape", b"x")], {"../escape": {"bytes": 1, "sha256": "x"}})

    def test_install_collision(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "model.onnx"
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                install(root / "unused.zip", root, [dict(path="model.onnx", sha256="different")])
            self.assertEqual(target.read_bytes(), b"existing")

    def test_install_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            (root / "models").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                install(root / "unused.zip", root, [dict(path="models/x", sha256="x")])


if __name__ == "__main__":
    unittest.main()
