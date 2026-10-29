"""Check that cache maintenance cannot select outside, symlinked or open files."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reclaim_q1_file_cache as cache


class CacheScopeTests(unittest.TestCase):
    def test_scope_excludes_outside_and_sibling_prefix(self):
        root = Path("/tmp/q1")
        self.assertTrue(cache.in_scope(root / "data/weights.pt", root))
        self.assertFalse(cache.in_scope(root / "data_other/weights.pt", root))
        self.assertFalse(cache.in_scope(Path("/tmp/other/data/weights.pt"), root))

    def test_symlinks_and_open_inodes_are_excluded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            (root / "data").mkdir()
            outside = root / "outside"
            outside.write_bytes(b"preserve")
            (root / "data/link").symlink_to(outside)
            active = root / "data/active"
            active.write_bytes(b"active")
            cold = root / "data/cold"
            cold.write_bytes(b"cold")
            info = active.stat()
            selected = cache.candidates(root, {(info.st_dev, info.st_ino)})
            self.assertEqual([path for path, _ in selected], [cold])
            self.assertEqual(outside.read_bytes(), b"preserve")

    def test_replaced_inode_does_not_receive_advice(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            (root / "data").mkdir()
            target = root / "data/checkpoint"
            target.write_bytes(b"saved weights")
            original = target.stat()
            target.rename(root / "data/original")
            target.write_bytes(b"new weights")
            with patch.object(cache, "resident_bytes") as observe:
                self.assertEqual(cache.release(target, original, set(), 0, 1, root), (0, False))
                observe.assert_not_called()
            self.assertEqual(target.read_bytes(), b"new weights")


if __name__ == "__main__":
    unittest.main()
