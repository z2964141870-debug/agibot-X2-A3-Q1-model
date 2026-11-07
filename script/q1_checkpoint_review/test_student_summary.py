"""Regressions for failure-aware R1 artifact comparisons and software timing."""

import unittest

import numpy as np

from student_summary import FIXED, KEYS, differences, full_group, software_summary
from stream_events import EventSpec


class StudentSummaryTests(unittest.TestCase):
    def timings(self, fallback):
        return [dict(frame=i, horizon_frames=4, fallback=kind, input_ready=kind is None,
            observed_prefix_frames=0 if kind else 3, gpu_policy_wall_compute_ns=t,
            worker_action_wall_ns=t+1000, local_pipe_rpc_ns=t+2000)
            for i, (kind, t) in enumerate(zip(fallback, (500000000, 150000000, 25000000, 8000000)))]

    def test_failed_clip_cannot_produce_a_full_group_mean_or_difference(self):
        rows = [dict(clip=c, complete=i != 2, **{k:1. for k in KEYS}) for i, c in enumerate(FIXED)]
        value = full_group(rows)
        self.assertEqual(value["complete_cases"], 2)
        self.assertIsNone(value["full_fixed_set_means"])
        self.assertIsNone(differences(rows[2], rows[0]))

    def test_missing_fixed_clip_is_not_silently_replaced_by_a_successful_subset(self):
        with self.assertRaises(RuntimeError):
            full_group([dict(clip=FIXED[0], complete=True)])

    def test_nested_scopes_and_stable_software_spike_remain_separate(self):
        timings = self.timings([None]*4)
        result = software_summary(timings, np.ones((4, 29)), EventSpec("nominal", 50, 10, 0))
        compute = result["gpu_policy_wall_compute_ms"]
        self.assertEqual(compute["first_two_ms"], [500., 150.])
        self.assertEqual(compute["without_first_two"]["calls_over_20ms"], 1)
        self.assertEqual(compute["without_first_two"]["maximum_ms"], 25.)
        self.assertIn("local_pipe_rpc_ms", result)
        self.assertIn("worker_action_wall_ms", result)

    def test_truncated_hold_has_no_invented_recovery(self):
        timings = self.timings([None, "previous_action_hold", "previous_action_hold"])
        result = software_summary(timings, np.ones((3, 29)), EventSpec("dropout", 50, 10, 0))
        interval = result["fallback_intervals"][0]
        self.assertIsNone(interval["recovered_origin_frame"])
        self.assertIsNone(interval["recovery_after_packet_window_ms"])

    def test_fallback_claim_must_match_saved_action_and_recovery_jump_is_retained(self):
        timings = self.timings(["initial_zero_delta", "previous_action_hold", None])
        actions = np.zeros((3, 29))
        actions[2] = 2.
        value = software_summary(timings, actions, EventSpec("nominal", 50, 10, 0))
        self.assertEqual(value["fallback_intervals"][0]["recovery_action_jump_max"], 2.)
        actions[1] = 1.
        with self.assertRaisesRegex(RuntimeError, "declared hold"):
            software_summary(timings, actions, EventSpec("nominal", 50, 10, 0))


if __name__ == "__main__":
    unittest.main()
