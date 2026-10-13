"""Selection controls prevent missing coverage or splicing different policies."""

from copy import deepcopy
import unittest

from script.a3.cloth_curve_protocol import CHECKPOINTS, outcome


def result(passed=False):
    return dict(baseline_clothing_mean_rad=.14, candidate_clothing_mean_rad=.13 if passed else .15,
        development_gate_passed=passed, rows=[dict(kind='cloth' if i < 6 else 'official',
            motion=str(i), candidate_fall=False, baseline_fall=False,
            ratio=.98 if passed else 1.04, world_error_ratios={
                'root_pos_error_m_mean':1., 'wrists_world_mean_m':1.}) for i in range(10)])


class CurveProtocolTests(unittest.TestCase):
    def test_complete_rejection_does_not_extend_training(self):
        report = outcome({s: result() for s in CHECKPOINTS})
        self.assertEqual(report['status'], 'REJECTED_FIXED_EXPOSURE_BRANCH')
        self.assertIsNone(report['selected_development_step'])
        self.assertFalse(report['automatic_more_training_allowed'])

    def test_selected_checkpoint_still_needs_independent_validation(self):
        records = {s: result(s == 16) for s in CHECKPOINTS}
        report = outcome(records)
        self.assertEqual(report['selected_development_step'], 16)
        self.assertFalse(report['deploy_allowed'])
        self.assertFalse(report['unseen_generalization_proven'])

    def test_missing_duplicate_or_changed_motion_coverage_rejected(self):
        records = {s: result() for s in CHECKPOINTS}
        bad = deepcopy(records)
        del bad[16]
        with self.assertRaises(ValueError):
            outcome(bad)
        for motion in ('1', 'new'):
            replacement = deepcopy(records)
            replacement[16]['rows'][0]['motion'] = motion
            with self.assertRaises(ValueError):
                outcome(replacement)

    def test_false_success_flag_cannot_bypass_original_gates(self):
        for field, value in (('candidate_fall', True), ('ratio', 1.021)):
            records = {s: result(True) for s in CHECKPOINTS}
            records[8]['rows'][6][field] = value
            with self.assertRaises(ValueError):
                outcome(records)
        records = {s: result(True) for s in CHECKPOINTS}
        records[32]['rows'][0]['world_error_ratios']['wrists_world_mean_m'] = 1.051
        with self.assertRaises(ValueError):
            outcome(records)


if __name__ == '__main__':
    unittest.main()
