"""Check that cache maintenance cannot select outside, symlinked or open files."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reclaim_q1_file_cache as cache


class CacheScopeTests(unittest.TestCase):
    def test_default_identity_still_rejects_replaced_device(self):
        args = SimpleNamespace(once=True, trigger_gib=8, target_gib=4, batch_gib=8)
        with patch.object(cache, "ROOT", Path.cwd()), patch.object(Path, "read_text", return_value="new-boot"), \
                patch.object(cache.subprocess, "check_output") as gpu, patch.object(cache, "snapshot") as snapshot:
            with self.assertRaisesRegex(ValueError, "boot identity"):
                cache.maintain(args)
            gpu.assert_not_called()
            snapshot.assert_not_called()

    def test_explicit_verified_identity_allows_same_guarded_maintenance(self):
        args = SimpleNamespace(once=True, trigger_gib=8, target_gib=4, batch_gib=8,
                               expected_boot="new-boot", expected_gpu="new-gpu")
        with patch.object(cache, "ROOT", Path.cwd()), patch.object(Path, "read_text", return_value="new-boot"), \
                patch.object(cache.subprocess, "check_output", return_value="new-gpu\n"), \
                patch.object(cache, "snapshot", return_value={"cache_bytes": 0}):
            result = cache.maintain(args)
        self.assertEqual(result["status"], "SKIPPED_BELOW_CACHE_THRESHOLD")
        self.assertEqual((result["expected_boot"], result["expected_gpu"]), ("new-boot", "new-gpu"))
        self.assertEqual(result["maximum_advice_bytes"], 8 * cache.GIB)
        self.assertFalse(result["file_contents_read"])
        self.assertEqual(result["files_deleted"], 0)

    def test_wrong_gpu_is_rejected_before_file_scanning(self):
        args = SimpleNamespace(expected_boot="new-boot", expected_gpu="new-gpu")
        with patch.object(cache, "ROOT", Path.cwd()), patch.object(Path, "read_text", return_value="new-boot"), \
                patch.object(cache.subprocess, "check_output", return_value="another-gpu\n"), \
                patch.object(cache, "snapshot") as snapshot:
            with self.assertRaisesRegex(ValueError, "GPU identity"):
                cache.maintain(args)
            snapshot.assert_not_called()

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
