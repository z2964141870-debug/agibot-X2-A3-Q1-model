"""New replay integration controls; no simulated or tensor execution."""

import ast
from pathlib import Path
import unittest

from script.a3.humanplus_specialist_bridge import correspondence
from script.a3.humanplus_specialist_repaired_entry import reference_joint_names, evaluation_tree
from script.a3.humanplus_specialist_repaired_trial import launch_arguments


class RepairedControls(unittest.TestCase):
    def test_native_selected_and_complete_joint_metadata(self):
        robot = ['passive'] + ['j' + str(i) for i in range(29)] + ['passive2']
        order = list(reversed(range(1, 30)))
        names = reference_joint_names(robot, 29, True, order)
        m = correspondence(selected_body_names=['root'], reference_body_count=1,
            robot_body_names=['aux', 'root'], reference_joint_names=names, reference_joint_count=29,
            robot_joint_names=robot, action_joint_names=robot[1:30])
        self.assertEqual(m['reference_joint_indices'], tuple(reversed(range(29))))
        self.assertEqual(reference_joint_names(robot, 31, False), robot)
        for ids in ([-1] * 29, [31] * 29, list(range(28))):
            with self.assertRaises(ValueError):
                reference_joint_names(robot, 29, True, ids)

    def test_independent_binding_and_replay_has_no_double_index(self):
        from script.a3 import humanplus_specialist_repaired_trial as trial
        self.assertTrue(trial.NAME.endswith('_HP21'))
        self.assertIn('R27_', str(trial.RUN))
        self.assertEqual(trial.guard.health.RECORD.name, 'a3_humanplus_recovery_20261018_RC01.json')
        path = Path(__file__).with_name('humanplus_specialist_repaired_entry.py')
        text = path.read_text()
        self.assertNotIn('command.body_indexes_data', text)
        self.assertIn("command.body_pos_w[0].clone()", text)
        self.assertIn('humanplus_specialist_repaired_trial', text)
        self.assertEqual(sum(isinstance(n, ast.FunctionDef) and n.name == 'replay'
            for n in ast.walk(ast.parse(text))), 1)
        for stage in ('isaac-baseline', 'isaac-final'):
            self.assertIn('++num_envs=1', launch_arguments(stage))
        self.assertIn('num_envs=16', launch_arguments('train'))

    def test_pinned_vendor_ast_still_matches_repaired_entry(self):
        path = Path('script/vendor/sonic_for_a3/gear_sonic/eval_agent_trl.py')
        if not path.is_file():
            self.skipTest('Vendor source absent on Mac')
        compile(evaluation_tree(path.read_text(), str(path), Path('/registered/log')), str(path), 'exec')


if __name__ == '__main__':
    unittest.main()
