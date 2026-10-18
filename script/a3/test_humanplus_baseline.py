"""Standard-library controls for HP10 process ownership and no-resume ordering."""

from pathlib import Path
import unittest
from unittest.mock import patch

from script.a3 import humanplus_baseline as target
from script.project_tools.humanplus_baseline_audit import verify_trace


class BaselineControls(unittest.TestCase):
    def test_native_descendants_stay_in_guarded_group(self):
        argv = [str(target.ROOT / 'data/environments/a3-sonic/bin/python'),
                '-m', 'script.a3.sim_entry', '--', '--checkpoint', '035.pt']
        with patch.object(target.subprocess, 'Popen') as spawn:
            target.EvaluationLaunch().Popen(argv, start_new_session=True)
        actual = spawn.call_args
        self.assertEqual(actual.args[0][:4], [str(target.ENV / 'bin/python'), '-B', '-m',
                                             'script.a3.smpl_frozen_entry'])
        self.assertFalse(actual.kwargs['start_new_session'])
        self.assertEqual(actual.args[0][4:], argv[3:])

    def test_unknown_launch_is_rejected(self):
        with patch.object(target.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                target.EvaluationLaunch().Popen(['other-python'])
            spawn.assert_not_called()

    def test_stage_order_and_no_replay(self):
        result = dict(status='REGISTERED', stages=[])
        target.require_next(result, 'validate')
        with self.assertRaises(ValueError):
            target.require_next(result, 'cloth')
        result.update(status='IN_PROGRESS', stages=[dict(stage='validate')])
        target.require_next(result, 'cloth')
        with self.assertRaises(ValueError):
            target.require_next(result, 'validate')

    def test_terminal_or_running_cannot_resume(self):
        for status in ('STOPPED', 'RUNNING', 'COMPLETE_MATCHED_BASELINE'):
            with self.subTest(status=status), self.assertRaises(ValueError):
                target.require_next(dict(status=status, stages=[]), 'validate')

    def histories(self):
        clothing = dict(selection=[dict(filename=str(n) + '.csv', source='data/clothing/' + str(n),
            sequence=str(n), partition='test') for n in range(6)])
        official = dict(diagnostic_motions=[dict(name=str(n) + '.csv', sha256='d' * 64,
            path='/old/host/' + str(n) + '.csv') for n in range(6, 10)])
        return clothing, official

    def test_explicit_new_root_paths_preserve_identity(self):
        clothing, official = self.histories()
        rows = target.motions_from_history(clothing, official)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(not Path(row['path']).is_absolute() for row in rows))
        self.assertTrue(all('/old/host/' not in row['path'] for row in rows))
        self.assertEqual([r['frame_stride'] for r in rows], [1] * 6 + [4] * 4)

    def test_duplicate_motion_and_path_traversal_rejected(self):
        clothing, official = self.histories()
        official['diagnostic_motions'][0]['name'] = '0.csv'
        with self.assertRaises(ValueError):
            target.motions_from_history(clothing, official)
        clothing, official = self.histories()
        clothing['selection'][0]['source'] = '../outside'
        with self.assertRaises(ValueError):
            target.motions_from_history(clothing, official)


class AuditControls(unittest.TestCase):
    def records(self):
        summary = dict(num_policy_steps=2, tracking=dict(all_29_rmse=0.), fall=True, fall_tick=1)
        trace = dict(policy_tick=[0, 1], q_state_29=[[0.] * 29] * 2,
            reference_q_29=[[0.] * 29] * 2, root_height=[.8, .4],
            root_roll_deg=[0., 0.], root_pitch_deg=[0., 0.])
        return summary, trace

    def test_independent_joint_and_fall_check(self):
        summary, trace = self.records()
        self.assertEqual(verify_trace(summary, trace, 2)['first_fall_s'], .02)
        summary['fall_tick'] = 0
        with self.assertRaises(ValueError):
            verify_trace(summary, trace, 2)

    def test_truncation_and_nonfinite_rejected(self):
        summary, trace = self.records()
        with self.assertRaises(ValueError):
            verify_trace(summary, trace, 3)
        trace['q_state_29'] = [[float('nan')] * 29] * 2
        with self.assertRaises(ValueError):
            verify_trace(summary, trace, 2)

    def test_wrong_reported_rmse_rejected(self):
        summary, trace = self.records()
        summary['tracking']['all_29_rmse'] = .1
        with self.assertRaises(ValueError):
            verify_trace(summary, trace, 2)


if __name__ == '__main__':
    unittest.main()
