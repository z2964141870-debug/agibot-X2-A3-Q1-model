"""Protect bounded scope, original failure provenance and untrimmed timing tails."""

import copy
import unittest

from event_profile import CASES, CLIP, FRAMES, selected_rows, statistics, summarize


def parent():
    return dict(stage="JOINED84_SAVED_EVENT_ANALYSIS", status="PASSED",
        joined84_sources_verified=True, total_cases=84, complete_cases=84,
        original_campaign_status="FAILED", original_full84_campaign_gate_passed=False,
        rows=[dict(event=e,method=m,horizon_frames=h,clip=CLIP,complete=True) for e,m,h in CASES])


class ProfileTests(unittest.TestCase):
    def test_only_four_saved_prefixes_are_selected(self):
        rows=selected_rows(parent())
        self.assertEqual([(r["event"],r["method"],r["horizon_frames"]) for r in rows],list(CASES))
        self.assertEqual(len(rows)*FRAMES,512)

    def test_duplicate_or_incomplete_saved_stream_is_rejected(self):
        value=parent()
        value["rows"].append(copy.deepcopy(value["rows"][0]))
        with self.assertRaises(RuntimeError):selected_rows(value)
        value=parent()
        value["rows"][0]["complete"]=False
        with self.assertRaises(RuntimeError):selected_rows(value)

    def test_original_failure_cannot_gain_the_original_complete_gate(self):
        value=parent()
        value["original_full84_campaign_gate_passed"]=True
        with self.assertRaises(RuntimeError):selected_rows(value)

    def test_second_call_and_later_tail_are_both_retained(self):
        frames=[]
        for frame in range(FRAMES):
            wall=500_000_000 if frame==1 else 35_000_000 if frame==70 else 5_000_000
            frames.append(dict(frame=frame,local_pipe_rpc_ns=wall+1_000_000,original_goal_scope_ns=wall,
                profile=dict(worker_action_wall_ns=wall,stage_wall_ns={"one":wall},
                             cuda_stream_interval_ms={"forward":wall/1e6})))
        value=summarize(frames)
        self.assertEqual(value["all"]["worker_action"]["maximum_ms"],500.)
        self.assertEqual(value["without_first_two"]["worker_action"]["maximum_ms"],35.)
        self.assertEqual(value["without_first_two"]["worker_action"]["calls_over_20ms"],1)
        self.assertEqual(value["capture_event_window"]["worker_action"]["samples"],10)
        self.assertEqual(value["slowest_worker_call"]["frame"],1)
        frames[70]["frame"]=69
        with self.assertRaises(RuntimeError):summarize(frames)

    def test_nonfinite_or_negative_timing_does_not_get_a_summary(self):
        for values in ([],[float("nan")],[float("inf")],[-1]):
            with self.assertRaises(RuntimeError):statistics(values)


if __name__ == "__main__":
    unittest.main()
