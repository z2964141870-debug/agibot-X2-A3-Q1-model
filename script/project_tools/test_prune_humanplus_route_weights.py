"""S02 scope and complete-rejection gates; temporary JSON only."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script.a3.checkpoint_store import sha256
from script.project_tools import prune_humanplus_route_weights as prune


class ScopeControls(unittest.TestCase):
    def test_only_hp19_generated_weights_selected(self):
        paths = prune.expected_paths()
        self.assertEqual(len(paths), 9)
        self.assertEqual(sum(p.name == 'last.pt' for p in paths), 4)
        self.assertTrue(all(prune.NAMES['HP19'] in p.parts for p in paths))
        self.assertFalse(set(prune.PROTECTED) & {str(p.relative_to(prune.ROOT)) for p in paths})

    def test_incomplete_unverified_or_accepted_candidate_never_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / 'data/manifests'
            directory.mkdir(parents=True)
            name = prune.NAMES['HP19']
            result = directory / (name + '_result.json')
            audit = directory / (name + '_effect_audit.json')
            value = dict(status='COMPLETE_MATCHED_EFFECT', stages=[dict(receipt=dict(
                matched_development_gate_passed=False))])
            result.write_text(json.dumps(value))
            proof = dict(verified=True, matched_development_gate_passed=False,
                         result_sha256=sha256(result), after_issues=[], remaining_own_group_processes=[])
            audit.write_text(json.dumps(proof))
            with patch.object(prune, 'ROOT', root):
                self.assertEqual(len(prune.eligibility()), 2)
                for key, replacement in [('verified', False), ('matched_development_gate_passed', True),
                        ('result_sha256', 'wrong'), ('after_issues', ['OOM']),
                        ('remaining_own_group_processes', [123])]:
                    audit.write_text(json.dumps(dict(proof, **{key: replacement})))
                    with self.assertRaises(ValueError):
                        prune.eligibility()
                audit.write_text(json.dumps(proof))
                value['status'] = 'STOPPED'
                result.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    prune.eligibility()


if __name__ == '__main__':
    unittest.main()
