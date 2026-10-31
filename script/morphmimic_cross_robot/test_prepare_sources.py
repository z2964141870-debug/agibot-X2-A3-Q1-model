"""Counterexamples for exact cross-robot SOMA source selection and extraction."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from prepare_sources import extract, source_plan


class SourceTests(unittest.TestCase):
    def row(self, name='walk'):
        return dict(filename=name, move_soma_uniform_path='soma_uniform/bvh/221012/' + name + '.bvh')

    def test_exact_metadata_and_date(self):
        plan = source_plan(['other_date/walk.npz'], [self.row('unselected'), self.row()])
        self.assertEqual(plan[0]['output_path'], '221012/walk.bvh')
        self.assertEqual(plan[0]['archive_member'], 'soma_uniform/bvh/221012/walk.bvh')

    def test_duplicate_stems_rejected(self):
        with self.assertRaises(ValueError):
            source_plan(['a/walk.npz', 'b/walk.npz'], [self.row()])

    def test_ambiguous_metadata_rejected(self):
        with self.assertRaises(ValueError):
            source_plan(['a/walk.npz'], [self.row(), self.row()])

    def test_missing_metadata_rejected(self):
        with self.assertRaises(ValueError):
            source_plan(['a/walk.npz'], [])

    def test_traversal_and_wrong_stem_rejected(self):
        for path in ('../bvh/221012/walk.bvh', '/soma_uniform/bvh/221012/walk.bvh',
                     'soma_uniform/bvh/../walk.bvh', 'soma_uniform/bvh/221012/other.bvh'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                source_plan(['a/walk.npz'], [dict(filename='walk', move_soma_uniform_path=path)])

    def archive(self, path, name, data=b'MOTION\nFrames: 1\n', link=False):
        with tarfile.open(path, 'w:gz') as stream:
            decoy = tarfile.TarInfo('unrelated/decoy')
            decoy.size = 1
            stream.addfile(decoy, io.BytesIO(b'x'))
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = '/unrelated'
                stream.addfile(member)
            else:
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))

    def test_selected_regular_file_only_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'source.tar.gz'
            plan = source_plan(['a/walk.npz'], [self.row()])
            self.archive(archive, plan[0]['archive_member'])
            result = extract(archive, plan, root / 'output')
            self.assertEqual(len(result['outputs']), 1)
            self.assertEqual((root / 'output/221012/walk.bvh').read_bytes(), b'MOTION\nFrames: 1\n')
            self.assertFalse((root / 'output/unrelated').exists())
            with self.assertRaises(FileExistsError):
                extract(archive, plan, root / 'output')

    def test_selected_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = source_plan(['a/walk.npz'], [self.row()])
            self.archive(root / 'source.tar.gz', plan[0]['archive_member'], link=True)
            with self.assertRaises(ValueError):
                extract(root / 'source.tar.gz', plan, root / 'output')

    def test_missing_archive_member_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = source_plan(['a/walk.npz'], [self.row()])
            self.archive(root / 'source.tar.gz', 'soma_uniform/bvh/221012/other.bvh')
            with self.assertRaises(ValueError):
                extract(root / 'source.tar.gz', plan, root / 'output')

    def test_resource_failure_stops_only_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = source_plan(['a/walk.npz'], [self.row()])
            self.archive(root / 'source.tar.gz', plan[0]['archive_member'])
            def stop():
                raise RuntimeError('protect existing training')
            with self.assertRaisesRegex(RuntimeError, 'protect existing training'):
                extract(root / 'source.tar.gz', plan, root / 'output', check=stop)
            self.assertFalse((root / 'output/221012/walk.bvh').exists())


if __name__ == '__main__':
    unittest.main()
