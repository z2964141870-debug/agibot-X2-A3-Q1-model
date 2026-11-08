"""Causal confirmation, censoring and label-sensitivity contracts."""

import unittest

import numpy as np

from saved_load_timing import compare_loads, confirm_modes, load_modes


class SavedLoadTimingTests(unittest.TestCase):
    def test_exact_inclusive_threshold_and_four_modes(self):
        np.testing.assert_array_equal(load_modes(np.array([[0,0],[5,0],[0,5],[5,5],[4.99,5]]),5),[0,1,2,3,2])

    def test_confirmation_never_backdates_online_emission(self):
        result = confirm_modes(np.array([0,1,1,2,2]),2)
        self.assertEqual(result['visible_modes'],[0,0,1,1,2])
        self.assertEqual([e['emission_index'] for e in result['events']],[2,4])
        self.assertEqual([e['retrospective_candidate_first_index'] for e in result['events']],[1,3])

    def test_future_change_cannot_alter_prior_visible_labels(self):
        raw = np.array([0,1,1,2,2,3])
        before = confirm_modes(raw,2)
        raw[4:] = 0
        after = confirm_modes(raw,2)
        self.assertEqual(before['visible_modes'][:4],after['visible_modes'][:4])

    def test_final_unconfirmed_candidate_remains_censored(self):
        value = confirm_modes(np.array([3,3,1]),2)
        self.assertEqual(value['events'],[])
        self.assertEqual(value['visible_modes'],[3,3,3])
        self.assertFalse(value['pending_final_candidate']['confirmed'])
        self.assertEqual(value['pending_final_candidate']['first_index'],2)

    def test_return_to_accepted_label_resets_candidate(self):
        value = confirm_modes(np.array([0,1,0,1,1]),2)
        self.assertEqual(len(value['events']),1)
        self.assertEqual(value['events'][0]['emission_index'],4)

    def test_no_difference_is_missing_time_not_zero_latency(self):
        loads = np.zeros((5,2))
        result = compare_loads([loads,loads],5.,2)
        self.assertIsNone(result['first_difference_observed_ms'])
        self.assertTrue(result['no_difference_right_censored'])
        self.assertFalse(result['response_latency_measured'])

    def test_short_excursion_is_retrospective_not_confirmed_event(self):
        loads = np.array([[0,0],[5,0],[0,0],[5,0]])
        result = compare_loads([loads,loads.copy()],5.,2)
        self.assertEqual(result['streams'][0]['retrospective_single_sample_excursions'],2)
        self.assertEqual(result['streams'][0]['events'],[])
        self.assertIsNotNone(result['streams'][0]['pending_final_candidate'])

    def test_invalid_labels_loads_or_initial_state_are_rejected(self):
        for labels in (np.array([0,4]),np.array([0.,1.])):
            with self.assertRaises(ValueError): confirm_modes(labels,2)
        with self.assertRaises(ValueError): load_modes(np.array([[0,0],[np.nan,0]]),5)
        with self.assertRaises(ValueError): compare_loads([np.zeros((3,2)),np.ones((3,2))],5,2)


if __name__ == '__main__':
    unittest.main()
