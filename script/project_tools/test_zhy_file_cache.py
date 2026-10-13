"""Visibility evidence controls; no real process inspection or cache mutation."""

import unittest
from script.project_tools.zhy_file_cache import missing_process_evidence


class CacheVisibilityTests(unittest.TestCase):
    def test_zombie_needs_no_rss(self):
        self.assertEqual(missing_process_evidence(8, {'State':'Z (zombie)', 'Name':'python'})['state'], 'ZOMBIE')

    def test_small_known_service_allowed(self):
        self.assertEqual(missing_process_evidence(204, {'State':'S (sleeping)', 'Name':'ssh-agent', 'VmRSS':'1084 kB'})['rss_kib'],1084)

    def test_unreadable_unknown_or_large_service_still_rejected(self):
        for fields in ({'State':'S', 'Name':'python3', 'VmRSS':'10 kB'},
                       {'State':'S', 'Name':'pod', 'VmRSS':'16385 kB'},
                       {'State':'S', 'Name':'pod'}):
            with self.assertRaisesRegex(RuntimeError, '"pid": 77'):
                missing_process_evidence(77, fields)


if __name__ == '__main__':
    unittest.main()
