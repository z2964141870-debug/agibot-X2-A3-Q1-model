"""Compare the final gate with identical metrics under reused-cloth provenance."""

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from script.a3 import humanplus_head_completion as current
from script.a3 import humanplus_head_effect as original


class CompletionControls(unittest.TestCase):
    def test_combined_effect_is_identical_for_pass_and_failure(self):
        for factor, fallen in ((.98, False), (1.131329674, True)):
            motions, after = [], []
            values = dict(joint_rmse_rad=1., root_pos_error_m_mean=1., wrists_world_mean_m=1.)
            for i in range(10):
                group = 'cloth' if i < 6 else 'official'
                key = str(i)
                motions.append(dict(key=key, group=group, fall=i == 5,
                                    windows=dict(full_rollout=values)))
                changed = dict(values, joint_rmse_rad=factor if group == 'cloth' else 1.01)
                after.append(dict(motion=key + '.csv', fall=fallen and i == 5,
                                  windows=dict(full_rollout=changed)))
            contract = dict(motions=motions, baseline_clothing_arithmetic_mean_rmse_rad=1.)
            cloth = dict(motions=after[:6])
            official = dict(stage='official', receipt=dict(motions=after[6:]))
            old_state = dict(stages=[dict(stage='cloth', receipt=cloth), official])
            new_state = dict(stages=[official])
            def old_read(path):
                return old_state if path == original.RESULT else contract
            def new_read(path):
                if path == current.SAVED_CLOTH:
                    return dict(accepted_cloth_summary=cloth)
                return new_state if path == current.RESULT else contract
            with patch.object(original, 'read', old_read), patch.object(current, 'read', new_read):
                plan = dict(matched_baseline_contract='fixture')
                self.assertEqual(original.effect(plan), current.effect(plan))

    def test_only_remaining_official_and_effect_exist(self):
        self.assertEqual(current.STAGES, ('official', 'effect'))
        self.assertFalse(hasattr(current, 'native'))
        self.assertEqual(current.checkpoint(8), original.checkpoint(8))
        with self.assertRaises(ValueError):
            current.checkpoint(4)

    def test_explicit_preclip_to_postclip_boundary(self):
        tree = ast.parse(Path(current.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_replays')
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'np.clip']
        self.assertEqual(len(calls), 1)
        self.assertEqual(ast.unparse(calls[0]),
            "np.clip(captured['actions'].astype(np.float64), -plan['action_clip'], plan['action_clip'])")


if __name__ == '__main__':
    unittest.main()
