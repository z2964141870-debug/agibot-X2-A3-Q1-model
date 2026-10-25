"""Deletion-gate controls using disposable files only."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('cleanup', Path(__file__).with_name('cleanup_private_20261025.py'))
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.allowed = self.root / 'allowed'
        self.allowed.mkdir()
        self.file = self.allowed / 'sample'
        self.file.write_bytes(b'preserve unless explicitly selected')

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_identity_accepts_disposable_file(self):
        row = cleanup.file_record(self.file, 'test', {}, [self.allowed])
        self.assertEqual(cleanup.check_record(row, [self.allowed], {}), self.file)

    def test_changed_file_rejected(self):
        row = cleanup.file_record(self.file, 'test', {}, [self.allowed])
        self.file.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            cleanup.check_record(row, [self.allowed], {})

    def test_outside_scope_rejected(self):
        with self.assertRaises(ValueError):
            cleanup.validate_path(self.file, [self.root / 'elsewhere'], {})

    def test_protected_inode_rejected(self):
        st = self.file.stat()
        with self.assertRaises(ValueError):
            cleanup.validate_path(self.file, [self.allowed], {self.root / 'alias': (st.st_dev, st.st_ino)})

    def test_symlink_parent_rejected(self):
        alias = self.allowed / 'link'
        alias.symlink_to(self.allowed, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup.validate_path(alias / 'sample', [self.allowed], {})

    def test_hardlink_rejected(self):
        (self.allowed / 'hardlink').hardlink_to(self.file)
        with self.assertRaises(ValueError):
            cleanup.validate_path(self.file, [self.allowed], {})

    def test_current_process_open_file_visible(self):
        with self.file.open('rb'):
            refs, _ = cleanup.process_references()
            self.assertIn(str(self.file), refs)

    def test_complete_owned_hardlink_group_can_be_unlinked(self):
        alias = self.allowed / 'hardlink'
        alias.hardlink_to(self.file)
        owned = cleanup.owned_env_hardlinks([self.allowed])
        rows = [cleanup.file_record(p, 'test', {}, [self.allowed], owned)
                for p in [self.file, alias]]
        changes = {}
        for row in rows:
            cleanup.unlink_record(row, [self.allowed], {}, owned, changes)
        self.assertFalse(self.file.exists())
        self.assertFalse(alias.exists())

    def test_hardlink_with_name_outside_retired_environment_rejected(self):
        (self.root / 'outside').hardlink_to(self.file)
        self.assertEqual(cleanup.owned_env_hardlinks([self.allowed]), set())

    def test_hardlinked_symlink_group_preserves_target(self):
        first, second = self.allowed / 'first', self.allowed / 'second'
        first.symlink_to(self.file.name)
        os.link(first, second, follow_symlinks=False)
        owned = cleanup.owned_env_hardlinks([self.allowed])
        rows = [cleanup.file_record(p, 'test', {}, [self.allowed], owned)
                for p in [first, second]]
        changes = {}
        for row in rows:
            cleanup.unlink_record(row, [self.allowed], {}, owned, changes)
        self.assertTrue(self.file.is_file())
        self.assertFalse(first.is_symlink())
        self.assertFalse(second.is_symlink())


if __name__ == '__main__':
    unittest.main()
