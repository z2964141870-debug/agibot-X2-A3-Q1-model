import tempfile
import unittest
import zipfile
from pathlib import Path
from script.teleop_research.pairing_candidates import candidates, REQUIRED

class Tests(unittest.TestCase):
    def make_zip(self, names):
        root = tempfile.TemporaryDirectory(); self.addCleanup(root.cleanup)
        p = Path(root.name) / 'data.zip'
        with zipfile.ZipFile(p, 'w') as z:
            for n in names: z.writestr(n, b'not-a-tensor')
        return p
    def test_candidates_not_ground_truth(self):
        rows = candidates(self.make_zip(['train/group/session/'+n for n in REQUIRED]))
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]['subject_id'])
        self.assertFalse(rows[0]['usable_as_independent_test'])
        self.assertFalse(rows[0]['source_alignment_verified'])
    def test_no_cross_session_pairing(self):
        names = ['a/'+n for n in REQUIRED if n != 'pose.pt'] + ['b/pose.pt']
        self.assertEqual(candidates(self.make_zip(names)), [])
    def test_missing_calibration(self):
        self.assertEqual(candidates(self.make_zip(['a/'+n for n in REQUIRED if n != 'device2bone.pt'])), [])
    def test_unsafe_path_rejected(self):
        with self.assertRaises(ValueError): candidates(self.make_zip(['../acc.pt']))

if __name__ == '__main__': unittest.main()
