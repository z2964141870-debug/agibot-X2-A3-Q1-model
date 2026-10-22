import unittest
from script.teleop_research.raw_timing_audit import timing
class Tests(unittest.TestCase):
    def test_fifty_hz(self):
        r=timing(['2026-01-01-00:00:00.000000,1', '2026-01-01-00:00:00.020000,1'])
        self.assertAlmostEqual(r['implied_recording_hz_from_median'],50)
        self.assertFalse(r['pt_timestamp_alignment_verified'])
    def test_disorder(self):
        r=timing(['2026-01-01-00:00:00.020000,1']*2+['2026-01-01-00:00:00.010000,1'])
        self.assertEqual(r['duplicate_intervals'],1);self.assertEqual(r['backward_intervals'],1)
    def test_invalid(self):
        r=timing(['bad,1','']);self.assertEqual(r['invalid_timestamp_rows'],1)
        self.assertIsNone(r['implied_recording_hz_from_median'])
if __name__=='__main__':unittest.main()
