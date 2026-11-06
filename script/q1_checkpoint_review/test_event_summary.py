"""Check clock, truncation and failure rules for the saved-event report."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from event_summary import CLIPS, analyze, capture_window, full_group, telemetry_summary
from stream_contract import DT_NS
from stream_events import EventSpec


def timing(horizon, count=100):
    return [dict(frame=i, simulated_control_ns=(i+horizon)*DT_NS,
        gpu_policy_wall_compute_ns=1_000_000, fallback=None, used_pose_age_ns=0) for i in range(count)]


class EventSummaryTests(unittest.TestCase):
    def test_software_spikes_remain_visible_after_first_call_is_removed(self):
        rows = timing(4, 3)
        for row, ms in zip(rows, (1000, 30, 1)):
            row["gpu_policy_wall_compute_ns"] = ms*1_000_000
        value = telemetry_summary(rows, np.zeros((3, 29)), EventSpec("stop", first_frame=50), 4)
        self.assertEqual(value["gpu_goal_call_all"]["calls_over_20ms"], 2)
        self.assertEqual(value["gpu_goal_call_without_first"]["maximum_ms"], 30)
        self.assertEqual(value["gpu_goal_call_without_first"]["calls_over_20ms"], 1)

    def test_recovery_is_on_capture_clock_and_retains_preview_wait(self):
        spec = EventSpec("dropout", first_frame=50, duration_frames=10)
        delays = []
        for h in (4, 10):
            rows = timing(h, 70)
            for i in range(49, 59):
                rows[i]["fallback"] = "visible_past_pose_hold"
            value = telemetry_summary(rows, np.zeros((70, 29)), spec, h)
            interval, = value["fallback_intervals"]
            self.assertEqual(interval["observed_fallback_ms"], 200)
            self.assertEqual(interval["recovered_origin_frame"], 59)
            delays.append(interval["recovery_after_packet_window_ms"])
        self.assertEqual(delays, [60, 180])

    def test_truncated_outage_does_not_invent_a_recovery(self):
        rows = timing(4, 55)
        for i in range(49, 55):
            rows[i]["fallback"] = "visible_past_pose_hold"
        value = telemetry_summary(rows, np.zeros((55, 29)), EventSpec("dropout", first_frame=50), 4)
        interval, = value["fallback_intervals"]
        self.assertIsNone(interval["recovered_origin_frame"])
        self.assertIsNone(interval["recovery_after_packet_window_ms"])

    def test_capture_window_uses_control_timestamps_and_next_saved_state(self):
        rows = timing(4, 120)
        actions = np.zeros((120, 29))
        reference = np.zeros((121, 79))
        actual = reference.copy()
        actual[47:97, 7] = 2.
        value = capture_window(rows, actions, actual, reference, [0], EventSpec("stop", first_frame=50))
        self.assertTrue(value["complete"])
        self.assertEqual(value["observed_samples"], 50)
        self.assertEqual(value["upper_rmse_rad"], 2.)
        short = capture_window(rows[:60], actions[:60], actual[:61], reference, [0], EventSpec("stop", first_frame=50))
        self.assertFalse(short["complete"])
        self.assertIsNone(short["upper_rmse_rad"])

    def test_one_failed_clip_prevents_full_set_means(self):
        rows = [dict(clip=clip, complete=i != 2) for i, clip in enumerate(CLIPS)]
        self.assertEqual(full_group(rows)["complete_cases"], 2)
        self.assertIsNone(full_group(rows)["full_fixed_set_means"])

    def test_live_campaign_is_rejected_before_output_or_trace_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, output = root / "campaign", root / "analysis"
            source.mkdir()
            (source / "result.json").write_text(json.dumps(dict(status="RUNNING")))
            with self.assertRaisesRegex(ValueError, "not formally complete"):
                analyze(source, output)
            self.assertFalse(output.exists())

    def test_wrong_clock_or_nonfinite_action_is_rejected(self):
        rows = timing(4, 2)
        rows[1]["simulated_control_ns"] += DT_NS
        with self.assertRaisesRegex(ValueError, "timestamp"):
            telemetry_summary(rows, np.zeros((2, 29)), EventSpec("stop", first_frame=50), 4)
        bad = np.full((2, 29), np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            telemetry_summary(timing(4, 2), bad, EventSpec("stop", first_frame=50), 4)


if __name__ == "__main__":
    unittest.main()
