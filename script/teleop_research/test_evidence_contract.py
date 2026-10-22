import copy
import unittest
from script.teleop_research.evidence_contract import full_task_success, validate_pairs, validate_comparison, wilson

class ContractTests(unittest.TestCase):
    def row(self):
        return dict(session_id='s1', subject_id='person1', split='test',
                    fgp_input_sha256='a'*64, target_sha256='b'*64, policy_sha256='c'*64,
                    start_frame=100, end_frame=400, time_alignment_id='sync-a',
                    target_provenance='paired_label_audited', seed=0, method='035')
    def test_early_fall_is_failure(self):
        self.assertIs(full_task_success(dict(fall=True, fall_time_s=0.2, num_policy_steps=499), 499), False)
    def test_no_fall_complete(self):
        self.assertIs(full_task_success(dict(fall=False, num_policy_steps=499), 499), True)
    def test_incomplete_not_success(self):
        self.assertIsNone(full_task_success(dict(fall=False, num_policy_steps=10), 499))
    def test_missing_not_success(self):
        self.assertIsNone(full_task_success({}, 499))
    def test_string_false_rejected(self):
        self.assertIsNone(full_task_success(dict(fall='false', num_policy_steps=499), 499))
    def test_pairs_valid(self):
        self.assertEqual(validate_pairs([self.row()])['sessions'], 1)
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError): validate_pairs([self.row(), self.row()])
    def test_subject_leakage_rejected(self):
        other = self.row(); other.update(session_id='s2', split='train')
        with self.assertRaises(ValueError): validate_pairs([self.row(), other])
    def test_prediction_is_not_gt(self):
        row = self.row(); row['target_provenance'] = 'fgp_prediction'
        with self.assertRaises(ValueError): validate_pairs([row])
    def test_wilson_reference(self):
        lo, hi = wilson(8, 10)
        self.assertAlmostEqual(lo, 0.490162, places=5)
        self.assertAlmostEqual(hi, 0.943318, places=5)
    def test_zero_failures_nonzero_uncertainty(self):
        self.assertGreater(wilson(0, 10)[1], 0.27)

class ComparisonTests(unittest.TestCase):
    def matched(self):
        a = ContractTests().row()
        a.update(environment_id='huanyu-frozen-v1', retarget_config_sha256='d'*64,
                 reset_config_sha256='e'*64, control_dt_s=0.02,
                 expected_steps=499, candidate_budget=1, method='raw')
        b = copy.deepcopy(a); b['method'] = 'causal_filter'
        return [a, b]
    def test_matched(self):
        self.assertEqual(validate_comparison(self.matched(), ['raw', 'causal_filter'])['paired_tasks'], 1)
    def test_missing_arm(self):
        with self.assertRaises(ValueError): validate_comparison(self.matched()[:1], ['raw', 'causal_filter'])
    def test_mismatch_invariants(self):
        for key, value in [('target_sha256', 'f'*64), ('reset_config_sha256', 'f'*64),
                           ('retarget_config_sha256', 'f'*64), ('environment_id', 'another_gpu'),
                           ('policy_sha256', 'f'*64), ('candidate_budget', 3),
                           ('control_dt_s', 0.03), ('expected_steps', 582),
                           ('time_alignment_id', 'other-sync')]:
            with self.subTest(key=key):
                rows = self.matched(); rows[1][key] = value
                with self.assertRaises(ValueError): validate_comparison(rows, ['raw', 'causal_filter'])
    def test_policy_comparison_declared(self):
        rows = self.matched(); rows[1]['policy_sha256'] = 'f'*64
        self.assertEqual(validate_comparison(rows, ['raw', 'causal_filter'], varying='policy')['paired_tasks'], 1)
    def test_windows_do_not_inflate_sessions(self):
        rows = self.matched()
        more = copy.deepcopy(rows)
        for r in more: r.update(start_frame=200, end_frame=500)
        out = validate_comparison(rows + more, ['raw', 'causal_filter'])
        self.assertEqual(out['paired_tasks'], 2)
        self.assertEqual(out['independent_session_count'], 1)

if __name__ == '__main__': unittest.main()
