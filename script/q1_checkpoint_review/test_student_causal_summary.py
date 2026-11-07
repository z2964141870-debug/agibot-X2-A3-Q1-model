"""Failure, provenance and clock regressions for saved f0 comparisons."""

import copy
import unittest

import numpy as np

from student_causal_summary import (completed_gate, history_software, match_controls, review_cases, video_contract)
from student_summary import KEYS
from stream_events import EventSpec


class CausalSummaryTests(unittest.TestCase):
    def comparisons(self):
        rows, controls = [], []
        for event, clip in review_cases():
            common = dict(event=event, clip=clip, complete=True, derived_reference_sha256=event+clip,
                **{key:1. for key in KEYS})
            rows.append(dict(common, variant="f0_student", horizon_frames=0, human_real_future_frames=0))
            controls.extend(dict(common, variant=v, horizon_frames=4) for v in ("teacher", "student"))
        return rows, controls

    def timing(self, frame, fallback=None, age=20_000_000):
        return dict(frame=frame, fallback=fallback, input_ready=fallback is None,
            gpu_policy_wall_compute_ns=(500_000_000, 150_000_000, 25_000_000, 8_000_000)[frame],
            worker_action_wall_ns=1_000_000, local_pipe_rpc_ns=2_000_000,
            current_capture_age_ns=age, used_sequences=[None]*9+[frame])

    def test_f0_has_history_age_and_keeps_stable_spikes_without_a_prefix_claim(self):
        timings = [self.timing(i) for i in range(4)]
        value = history_software(timings, np.ones((4, 29)), EventSpec("nominal", 50, 10, 0))
        self.assertNotIn("actual_prefix_histogram", value)
        self.assertEqual(value["history_valid_slots_histogram"], {1:4})
        self.assertEqual(value["gpu_policy_wall_compute_ms"]["without_first_two"]["calls_over_20ms"], 1)
        self.assertEqual(value["current_capture_age"]["input_ready"]["maximum_ms"], 20.)

    def test_fallback_reports_retained_old_support_without_invented_recovery(self):
        timings = [self.timing(0), self.timing(1, "previous_action_hold", 220_000_000)]
        value = history_software(timings, np.ones((2, 29)), EventSpec("dropout", 50, 10, 0))
        self.assertEqual(value["current_capture_age"]["fallback"]["maximum_ms"], 220.)
        self.assertIsNone(value["fallback_intervals"][0]["recovered_origin_frame"])
        self.assertIsNone(value["fallback_intervals"][0]["recovery_after_packet_window_ms"])

    def test_zero_preview_recovery_uses_origin_clock_and_saved_jump(self):
        timings = [self.timing(0, "initial_zero_delta"), self.timing(1, "previous_action_hold"), self.timing(2)]
        actions = np.zeros((3, 29))
        actions[-1] = 2.
        value = history_software(timings, actions, EventSpec("nominal", 50, 10, 0))
        self.assertEqual(value["fallback_intervals"][0]["recovered_synthetic_control_ms"], 40)
        self.assertEqual(value["fallback_intervals"][0]["recovery_action_jump_max"], 2.)
        actions[1] = 1.
        with self.assertRaisesRegex(RuntimeError, "saved previous-action hold"):
            history_software(timings, actions, EventSpec("nominal", 50, 10, 0))

    def test_missing_duplicate_changed_target_or_preview_cannot_enter_comparison(self):
        rows, controls = self.comparisons()
        self.assertEqual(len(match_controls(rows, controls)), 7)
        for invalid in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaisesRegex(RuntimeError, "seven unique"):
                match_controls(invalid, controls)
        wrong = copy.deepcopy(controls)
        wrong[0]["derived_reference_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "different targets"):
            match_controls(rows, wrong)
        with self.assertRaisesRegex(RuntimeError, "fourteen unique"):
            match_controls(rows, controls[:-1]+[controls[0]])
        invalid = copy.deepcopy(rows)
        invalid[0]["horizon_frames"] = 1
        with self.assertRaisesRegex(RuntimeError, "f0 comparison interface"):
            match_controls(invalid, controls)

    def test_failed_new_case_has_no_full_motion_differences(self):
        rows, controls = self.comparisons()
        rows[0]["complete"] = False
        value = match_controls(rows, controls)
        self.assertIsNone(value[0]["change_from_teacher_h4"])
        self.assertIsNone(value[0]["change_from_r1_h4"])

    def test_full_duration_failure_video_does_not_invent_physical_completion(self):
        row = dict(reference_steps=1031, attempted_steps=53, complete=False)
        video = dict(status="PASSED", original_speed=True, probe=dict(nb_frames=1032),
            actual_frames_before_hold=54, failure_preserved=True)
        video_contract(video, row)
        self.assertFalse(row["complete"])
        for changed in (dict(video, actual_frames_before_hold=1032), dict(video, failure_preserved=False)):
            with self.assertRaisesRegex(RuntimeError, "labelled failure hold"):
                video_contract(changed, row)

    def test_completion_requires_all_fourteen_unique_phases_and_reaped_controller(self):
        auth = dict(stage="R2_F0_MATCHED_CLOSED_LOOP", new_physical_cases=7, reused_baseline_cases=14,
            new_optimizer_updates=0, human_real_future_frames=0, cases=review_cases())
        value = dict(stage=auth["stage"], status="COMPLETED", source_integrity="PASSED",
            new_physical_cases=7, reused_baseline_cases=14, total_comparison_cases=21,
            new_optimizer_updates=0, controller=dict(pid=2147483647),
            cases=[dict(path=str(i)) for i in range(7)],
            phases=[dict(name=f"{e}_f0_student_h0_{c}_{p}")
                for e, c in review_cases() for p in ("replay", "media")])
        completed_gate(value, auth)
        for changed in (dict(value, status="RUNNING"), dict(value, source_integrity="PENDING"),
                dict(value, phases=value["phases"][:-1]), dict(value, cases=value["cases"][:-1])):
            with self.assertRaises(RuntimeError):
                completed_gate(changed, auth)


if __name__ == "__main__":
    unittest.main()
