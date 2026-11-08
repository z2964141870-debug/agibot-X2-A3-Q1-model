"""Raw-support and hidden-future checks for the independent completed preview."""

from copy import deepcopy
import unittest

import numpy as np

from causal_tail import forecast_window
from dependency_preview import DependencyPreviewReceiver, validate_dependency_timing
from dependency_preview_audit import phase_counts, prefix_at_phase
from event_inputs import EventReceiver
from forecast_data import source_support
from stream_contract import DT_NS
from stream_events import EventSpec


def receiver(kind="nominal", first=20, duration=5, transport=0, poison_after=None):
    positions = np.zeros((70, 24, 3))
    positions[:, :, 0] = np.arange(70)[:, None] * .01
    positions[:, :, 2] = 1.
    rotations = np.zeros((70, 24, 4))
    rotations[:, :, 0] = 1.
    if poison_after is not None:
        positions[poison_after:] += 1000.
    return DependencyPreviewReceiver(positions, rotations,
        EventSpec(kind, first, duration, transport), "a" * 64, "synthetic_contract_only")


def at_frame(value, frame, horizon=4, method="damped_cv"):
    for index in range(frame + 1):
        result = value.step(index, horizon, method)
    return result


class DependencyPreviewTests(unittest.TestCase):
    def test_cycle_aggregation_matches_independent_per_anchor_support(self):
        for frames in (11, 12, 14, 15, 16, 53, 1042):
            counts = phase_counts(frames)
            self.assertEqual(sum(counts.values()), frames-10)
            for phase in range(5):
                self.assertEqual(counts[phase], sum(i % 5 == phase for i in range(frames-10)))
            for horizon in (1, 2, 4, 5, 10):
                for anchor in range(min(20, frames-10)):
                    ready = sum(source_support(i)["dependency_end_ns"] <= (anchor+horizon)*DT_NS
                                for i in range(anchor+1, anchor+horizon+1))
                    self.assertEqual(prefix_at_phase(anchor % 5, horizon), ready)

    def test_h4_uses_three_or_four_real_slots_by_raw_support_phase(self):
        value = receiver()
        counts = []
        for frame in range(5):
            result = value.step(frame, 4, "damped_cv")
            counts.append(sum(result.timing["measured_mask"]))
            self.assertLessEqual(result.timing["max_used_dependency_ns"], (frame + 4) * DT_NS)
            self.assertEqual(result.policy_frames, 10)
            self.assertIsNone(result.timing["measured_end_to_end_latency_ns"])
            self.assertFalse(result.timing["upstream_capture_causality_verified"])
        self.assertEqual(counts, [3, 4, 3, 3, 3])

    def test_changed_unready_last_interpolant_cannot_change_any_goal_pose(self):
        original = at_frame(receiver(), 0)
        poisoned = at_frame(receiver(poison_after=4), 0)
        self.assertEqual(original.poses, poisoned.poses)
        self.assertEqual(original.timing, poisoned.timing)

    def test_completion_reuses_original_forecast_on_the_same_available_prefix(self):
        value = receiver()
        observed = value.stream.student(12, mode="short_preview", horizon=4)
        expected = forecast_window(observed, "damped_cv")
        actual = at_frame(value, 12)
        self.assertEqual(actual.poses, expected.poses)
        self.assertEqual(actual.timing["measured_mask"], expected.measured_mask)
        self.assertEqual(actual.timing["generated_mask"], expected.generated_mask)

    def test_full_raw_ready_h4_matches_legacy_goal_but_not_a_quality_claim(self):
        value = receiver()
        stream = value.stream
        legacy = EventReceiver(stream.positions, stream.quaternions, stream.events.spec, "a" * 64, "legacy")
        for frame in range(2):
            old = legacy.step(frame, 4, "damped_cv")
            current = value.step(frame, 4, "damped_cv")
        self.assertEqual(current.poses, old.poses)
        self.assertEqual(sum(current.timing["measured_mask"]), 4)

    def test_stop_reverse_dropout_and_late_burst_respect_both_cutoffs(self):
        for kind in ("stop", "reverse", "dropout", "late_burst"):
            with self.subTest(kind=kind):
                value = receiver(kind, first=23, duration=7, transport=2)
                for frame in range(40):
                    result = value.step(frame, 4, "damped_cv")
                    for row in result.timing["used_packets"]:
                        self.assertLessEqual(row["dependency_end_ns"], (frame + 4) * DT_NS)
                        self.assertLessEqual(row["arrival_ns"], (frame + 4) * DT_NS)
                        self.assertLessEqual(row["source_index"], row["sequence"])

    def test_one_measured_slot_uses_hold_and_never_relabels_generated_as_measured(self):
        result = at_frame(receiver(), 0, horizon=2)
        self.assertEqual(result.timing["effective_tail_method"], "hold")
        self.assertEqual(result.timing["measured_mask"], (True,) + (False,) * 9)
        self.assertEqual(result.timing["generated_mask"], (False,) + (True,) * 9)
        self.assertTrue(all(pose == result.poses[0] for pose in result.poses))

    def test_h1_missing_first_slot_holds_only_fresh_raw_ready_past_pose(self):
        result = at_frame(receiver(), 0, horizon=1)
        self.assertEqual(result.timing["fallback"], "visible_past_pose_hold")
        self.assertEqual([row["sequence"] for row in result.timing["used_packets"]], [0])
        self.assertFalse(any(result.timing["measured_mask"]))
        stale = receiver("dropout", first=10, duration=30)
        with self.assertRaisesRegex(ValueError, "No fresh"):
            at_frame(stale, 25, horizon=4)

    def test_parent_rejects_forged_dependency_mask_and_source_identity(self):
        value = receiver()
        result = value.step(0, 4, "damped_cv")
        for key, changed in (("max_used_dependency_ns", 81_000_000),
                             ("measured_mask", [True] * 10),
                             ("upstream_capture_causality_verified", True),
                             ("terminal_padding_applied", True),
                             ("damping_tau_s", .2)):
            bad = deepcopy(result.timing)
            bad[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_dependency_timing(bad, 0, 4, "damped_cv", value.stream.events)
        bad = deepcopy(result.timing)
        bad["used_packets"][0]["dependency_end_ns"] = 0
        with self.assertRaises(ValueError):
            validate_dependency_timing(bad, 0, 4, "damped_cv", value.stream.events)

    def test_masked_and_completed_modes_share_exact_raw_supported_prefix(self):
        for horizon in (1, 2, 4, 5, 10):
            for method in ("masked", "hold", "damped_cv"):
                with self.subTest(horizon=horizon, method=method):
                    value = receiver()
                    result = at_frame(value, 6, horizon, method)
                    observed = value.stream.student(6, mode="short_preview", horizon=horizon)
                    count = observed.prefix_frames
                    self.assertEqual(result.poses[:count], observed.poses[:count])
                    if count:
                        self.assertEqual(result.policy_frames, count if method == "masked" else 10)
                    with self.assertRaises(ValueError):
                        value.step(6, horizon, method)


if __name__ == "__main__":
    unittest.main()
