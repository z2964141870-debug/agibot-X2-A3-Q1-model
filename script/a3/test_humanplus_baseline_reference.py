"""Exercise the exact reference-directory adapter without loading tensor packages."""

from pathlib import Path
import tempfile
import unittest

from script.a3.humanplus_baseline_reference import stage_references


class ReferenceAdapterControls(unittest.TestCase):
    def fixture(self, root):
        selection = []
        for index in range(6):
            source = root / 'data' / str(index)
            source.mkdir(parents=True)
            (source / 'reference.csv').touch()
            selection.append(dict(source='data/' + str(index), filename=str(index) + '.csv'))
        return selection

    def test_exact_abi_lookup_paths_exist_before_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = self.fixture(root)
            output = root / 'validation'
            stage_references(root, output, selection)
            for row in selection:
                actual = output / 'references' / row['filename']
                self.assertTrue(actual.is_file())
                self.assertEqual(actual.resolve(), (root / row['source'] / 'reference.csv').resolve())
            with self.assertRaises(FileExistsError):
                stage_references(root, output, selection)

    def test_missing_source_and_alias_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = self.fixture(root)
            (root / selection[0]['source'] / 'reference.csv').unlink()
            with self.assertRaises(ValueError):
                stage_references(root, root / 'validation', selection)
            self.assertFalse((root / 'validation').exists())
            (root / selection[0]['source'] / 'reference.csv').touch()
            selection[0]['filename'] = '../outside.csv'
            with self.assertRaises(ValueError):
                stage_references(root, root / 'validation', selection)


if __name__ == '__main__':
    unittest.main()
