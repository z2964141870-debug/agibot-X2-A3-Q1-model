"""Exercise rename, compatibility links, and prefix mapping in temporary trees."""

import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('consolidate', Path(__file__).with_name('consolidate_private_20261025.py'))
consolidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(consolidate)


class ConsolidationControls(unittest.TestCase):
    def test_most_specific_mapping_wins(self):
        mapping = {Path('/old'): Path('/new/script'), Path('/old/data'): Path('/new/data')}
        self.assertEqual(consolidate.mapped('/old/data/input.npz', mapping), Path('/new/data/input.npz'))
        self.assertEqual(consolidate.mapped('/old/script/tool.py', mapping), Path('/new/script/script/tool.py'))

    def test_prefix_does_not_capture_other_project(self):
        self.assertEqual(consolidate.mapped('/old_other/input', {Path('/old'): Path('/new')}),
                         Path('/old_other/input'))

    def test_rename_preserves_inode_and_old_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, target = root / 'source', root / 'target'
            source.mkdir()
            file = source / 'model'
            file.write_bytes(b'disposable data')
            inode, digest = file.stat().st_ino, consolidate.sha256(file)
            with (root / 'journal').open('w') as journal:
                consolidate.move_and_alias(source, target, journal)
            self.assertTrue(source.is_symlink())
            self.assertEqual((target / 'model').stat().st_ino, inode)
            self.assertEqual(consolidate.sha256(source / 'model'), digest)

    def test_existing_destination_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, target = root / 'source', root / 'target'
            source.mkdir()
            target.mkdir()
            with (root / 'journal').open('w') as journal:
                with self.assertRaises(ValueError):
                    consolidate.move_and_alias(source, target, journal)
            self.assertFalse(source.is_symlink())
            self.assertTrue(source.is_dir())


if __name__ == '__main__':
    unittest.main()
