"""Regression tests for diagnostic selection and coverage safeguards."""

import json
from pathlib import Path
import tempfile
import unittest

from script.a3.zhy_search import FIELDS, aggregate, paired


def row(name, error, steps):
    metric = dict(policy_steps=steps, joint_rmse_rad=error, **{f:1. for f in FIELDS})
    return dict(motion=name+'.csv', fall=False, first_fall_s=None,
                full_replay_completed=True, windows={'full_rollout':metric})


class SelectionTests(unittest.TestCase):
    def test_aggregate_weights_frames_not_motion_count(self):
        value = aggregate([row('short', 1., 10), row('long', 3., 90)])
        self.assertAlmostEqual(value['joint_rmse_rad'], 8.2**.5)

    def test_missing_motion_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            for path in (a, b):
                path.mkdir()
                (path/'explicit_summary.json').write_text(json.dumps({'motions':[row('one', .9, 3)]}))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                paired(a, b, ['one', 'missing'])

    def test_joint_improvement_does_not_hide_body_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp)/'a', Path(temp)/'b'
            current = row('one', .9, 3)
            current['windows']['full_rollout']['wrists_world_mean_m'] = 1.06
            for path, value in ((a, current), (b, row('one', 1., 3))):
                path.mkdir()
                (path/'explicit_summary.json').write_text(json.dumps({'motions':[value]}))
                (path/'one.timeseries.json').write_text(json.dumps({'motions':[{'raw_action_29':[[0.]*29]*3}]}))
            result = paired(a, b, ['one'])
            self.assertLess(result['joint_change_percent'], -1.)
            self.assertFalse(result['diagnostic_gate_passed'])
            self.assertIn('one:wrists_world_mean_m_regression', result['gate_failures'])


if __name__ == '__main__':
    unittest.main()
