"""Filesystem controls for irreversible, explicitly scoped rejected-weight pruning."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script.project_tools import prune_humanplus_rejected_weights as prune
from script.a3.checkpoint_store import sha256


class PruneControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.weight = self.root / 'candidate.pt'
        self.weight.write_bytes(b'discarded test payload')
        self.sidecar = self.weight.with_suffix('.json')
        self.sidecar.write_text(json.dumps(dict(bytes=self.weight.stat().st_size, sha256=sha256(self.weight))))
        self.alias = self.root / 'last.pt'
        self.alias.symlink_to(self.weight.name)

    def tearDown(self):
        self.directory.cleanup()

    def enumerate(self, paths):
        with patch.object(prune, 'ROOT', self.root), patch.object(prune, 'expected_paths', return_value=paths):
            return prune.enumerate_files()

    def test_exact_regular_and_alias_and_all_hardlinks(self):
        link = self.root / 'bundle.pt'
        os.link(self.weight, link)
        link.with_suffix('.json').write_text(self.sidecar.read_text())
        rows, inodes = self.enumerate([self.weight, self.alias, link])
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(inodes), 1)
        self.assertEqual(sum(v[0]['bytes'] for v in inodes.values()), self.weight.stat().st_size)

    def test_unlisted_hardlink_refused(self):
        os.link(self.weight, self.root / 'unlisted.pt')
        with self.assertRaisesRegex(ValueError, 'Unlisted hardlink'):
            self.enumerate([self.weight, self.alias])

    def test_changed_sha_refused_before_delete(self):
        self.weight.write_bytes(b'new payload')
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            self.enumerate([self.weight, self.alias])
        self.assertTrue(self.weight.exists())

    def test_alias_escape_refused(self):
        other = self.root / 'other'
        other.mkdir()
        outside = other / 'protected.pt'
        outside.write_bytes(b'protected')
        self.alias.unlink()
        self.alias.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'alias target'):
            self.enumerate([self.weight, self.alias])

    def test_active_input_protected(self):
        manifests = self.root / 'data/manifests'
        manifests.mkdir(parents=True)
        plan = manifests / 'trial_HP99_plan.json'
        plan.write_text(json.dumps(dict(input_sha256={'candidate.pt': 'sha'})))
        with patch.object(prune, 'ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'Active or pending'):
                prune.active_input_check(['candidate.pt'])
            plan.with_name('trial_HP99_result.json').write_text(json.dumps(dict(status='COMPLETE_MATCHED_EFFECT')))
            prune.active_input_check(['candidate.pt'])

    def test_real_whitelist_excludes_baseline_and_input_tensors(self):
        paths = prune.expected_paths()
        self.assertEqual(len(paths), 32)
        self.assertFalse(any('/data/models/' in str(p) or '/a3_zhy_' in str(p) for p in paths))
        self.assertEqual(sum(p.name == 'last.pt' for p in paths), 15)

    def test_only_systemd_login_helper_is_excluded_from_auth_inspection(self):
        parent = self.root / '10'
        parent.mkdir()
        (parent / 'status').write_text('Name:\tsystemd\nPPid:\t1\n')
        self.assertTrue(prune.system_auth_helper('Name:\t(sd-pam)\nPPid:\t10\n', self.root))
        self.assertFalse(prune.system_auth_helper('Name:\tpython\nPPid:\t10\n', self.root))
        (parent / 'status').write_text('Name:\tpython\nPPid:\t1\n')
        self.assertFalse(prune.system_auth_helper('Name:\t(sd-pam)\nPPid:\t10\n', self.root))


if __name__ == '__main__':
    unittest.main()
