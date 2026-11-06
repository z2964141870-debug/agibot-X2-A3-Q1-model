"""Leakage, event availability and geometry checks, without policy/physics runs."""

from dataclasses import replace
import random
import unittest

import numpy as np

from forecast_geometry import product, yaw_matrix
from stream_contract import DT_NS, HumanPose, PoseFrame, SourceIdentity
from stream_events import EventSpec, EventTimeline
from student_inputs import (FRAME_DIM, GEOMETRY_DIM, current_features,
    current_history, fit_current_statistics, short_preview)


SOURCE = SourceIdentity("synthetic_student", "episode1", "common_replay_ns",
                        "smpl24_world_m_wxyz", "declared_causal")


def frame(index, *, marker=None, arrival=None, source=SOURCE):
    t = index*DT_NS
    value = index*.01 if marker is None else marker
    positions = tuple((value+i*.02, .1+i*.01, .8+i*.005) for i in range(24))
    pose = HumanPose(positions, ((1., 0., 0., 0.),)*24)
    return PoseFrame(source, index, t, t, t, t if arrival is None else arrival, pose)


def current(rows, *, anchor=12, control=None, start=0, age=5):
    return current_history(rows, SOURCE, anchor_ns=anchor*DT_NS,
        control_ns=(anchor if control is None else control)*DT_NS,
        episode_start_ns=start*DT_NS, max_capture_age_ns=age*DT_NS)


def preview(rows, horizon, *, anchor=12, control=22):
    return short_preview(rows, SOURCE, anchor_ns=anchor*DT_NS,
        control_ns=control*DT_NS, horizon=horizon, episode_start_ns=0,
        max_capture_age_ns=20*DT_NS)


def event_frames(kind):
    timeline = EventTimeline(40, EventSpec(kind, first_frame=12, duration_frames=4))
    return [frame(packet.sequence, marker=packet.source_index*.01, arrival=packet.arrival_ns)
            for packet in timeline.packets]


class StudentInputTests(unittest.TestCase):
    def test_h1_is_future_and_f0_uses_current_even_when_control_is_delayed(self):
        rows = [frame(i) for i in range(25)]
        self.assertEqual(preview(rows, 1).used_sequences, (13,))
        value = current(rows, control=22, age=10)
        self.assertEqual(value.samples[-1].sequence, 12)
        self.assertEqual(value.timing()["future_preview_ns"], 0)
        self.assertEqual(value.timing()["current_capture_age_ns"], 10*DT_NS)
        for invalid in (0, True):
            with self.assertRaises(ValueError):
                preview(rows, invalid)

    def test_future_pose_heading_and_root_changes_cannot_change_student_features(self):
        rows = [frame(i) for i in range(25)]
        changed = [row if row.sequence <= 12 else replace(row,
            pose=HumanPose(((999., -123., 4.),)*24, ((0., 0., 0., 1.),)*24)) for row in rows]
        left = current_features(current(rows, control=22, age=10), (1., 0., 0., 0.))
        right = current_features(current(changed, control=22, age=10), (1., 0., 0., 0.))
        np.testing.assert_array_equal(left, right)
        for horizon in (1, 2, 4, 5, 10):
            mutated = [row if row.sequence <= 12+horizon else frame(row.sequence, marker=1e6)
                       for row in rows]
            self.assertEqual(preview(rows, horizon), preview(mutated, horizon))

    def test_preprocessing_lookahead_and_unknown_cutoff_cannot_claim_f0(self):
        offline = replace(SOURCE, preprocessing="offline_unverified")
        rows = [replace(frame(i), source=offline, dependency_end_ns=(i+1)*DT_NS,
                        ready_ns=(i+1)*DT_NS, arrival_ns=(i+1)*DT_NS) for i in range(15)]
        value = current_history(rows, offline, anchor_ns=12*DT_NS, control_ns=15*DT_NS,
                                episode_start_ns=0, max_capture_age_ns=5*DT_NS)
        self.assertEqual(value.samples[-1].sequence, 11)
        self.assertEqual(value.timing()["max_used_dependency_ns"], 12*DT_NS)
        self.assertEqual(value.timing()["upstream_preprocessing"], "offline_unverified")
        with self.assertRaisesRegex(ValueError, "Unknown preprocessing"):
            current_history([replace(rows[-1], dependency_end_ns=None)], offline,
                anchor_ns=14*DT_NS, control_ns=15*DT_NS, episode_start_ns=0,
                max_capture_age_ns=DT_NS)

    def test_episode_boundary_padding_does_not_import_prior_episode_history(self):
        rows = [frame(i) for i in range(13)]
        value = current(rows, start=10)
        self.assertEqual([None if row is None else row.sequence for row in value.samples],
                         [None]*7+[10, 11, 12])
        features = current_features(value, (1., 0., 0., 0.))
        np.testing.assert_array_equal(features[:7], np.zeros((7, FRAME_DIM)))
        mixed = replace(frame(11), source=replace(SOURCE, session_id="episode2"))
        with self.assertRaisesRegex(ValueError, "Mixed source/session"):
            current(rows+[mixed])

    def test_dropout_uses_only_previous_support_and_reports_held_age(self):
        rows = event_frames("dropout")
        value = current(rows, anchor=15)
        self.assertEqual(value.samples[-1].sequence, 11)
        features = current_features(value, (1., 0., 0., 0.))
        self.assertEqual(features[-1, GEOMETRY_DIM], 1.)
        self.assertEqual(features[-1, GEOMETRY_DIM+1], 0.)
        self.assertAlmostEqual(features[-1, -1], .08)
        self.assertIsNone(value.timing()["measured_end_to_end_latency_ns"])
        self.assertFalse(current(rows, anchor=15, age=2).can_step)
        self.assertEqual(current(rows, anchor=16).samples[-1].sequence, 16)

    def test_late_burst_recovers_only_after_arrival_without_relabeling_old_capture(self):
        rows = event_frames("late_burst")
        before = current(rows, anchor=15)
        self.assertEqual(before.samples[-1].sequence, 11)
        after = current(rows, anchor=16)
        self.assertEqual(after.samples[-1].sequence, 16)
        self.assertEqual(after.samples[-2].capture_ns, 15*DT_NS)
        self.assertEqual(after.samples[-2].arrival_ns, 16*DT_NS)
        # At an earlier control instant these newly arrived poses are unavailable.
        self.assertTrue(all(row is None or row.arrival_ns <= 15*DT_NS for row in before.samples))

    def test_stop_and_reverse_do_not_warp_time_or_rewrite_the_history(self):
        for kind in ("stop", "reverse"):
            rows = event_frames(kind)
            first, later = current(rows, anchor=12), current(rows, anchor=14)
            self.assertEqual(first.samples[-1].pose.positions[0][0], .12)
            self.assertAlmostEqual(later.samples[-1].pose.positions[0][0],
                                   .12 if kind == "stop" else .10)
            self.assertEqual(later.targets_ns[-1]-first.targets_ns[-1], 2*DT_NS)
            self.assertEqual(first.samples[-2].pose, frame(11).pose)

    def test_current_geometry_preserves_height_and_common_world_yaw_invariance(self):
        rows = [frame(i) for i in range(13)]
        angle = .7
        yaw = np.array((np.cos(angle/2), 0., 0., np.sin(angle/2)))
        changed = [replace(row, pose=HumanPose(
            np.asarray(row.pose.positions)@yaw_matrix(angle).T+np.array((5., -3., 0.)),
            product(yaw, np.asarray(row.pose.orientations)))) for row in rows]
        before = current_features(current(rows), (1., 0., 0., 0.))
        after = current_features(current(changed), yaw)
        np.testing.assert_allclose(before, after, atol=2e-15)
        raised = [replace(row, pose=HumanPose(
            np.asarray(row.pose.positions)+np.array((0., 0., .2)), row.pose.orientations)) for row in rows]
        higher = current_features(current(raised), (1., 0., 0., 0.))
        np.testing.assert_allclose(higher[:, 216]-before[:, 216], .2)
        np.testing.assert_allclose(higher[:, :216], before[:, :216], atol=3e-16)

    def test_randomized_arrival_selection_matches_an_independent_prefix_oracle(self):
        rng = random.Random(2026110701)
        for _ in range(35):
            rows = [frame(i, arrival=(i+rng.randrange(6))*DT_NS) for i in range(30)
                    if rng.random() > .15]
            value = current(rows, anchor=20)
            for target, actual in zip(value.targets_ns, value.samples):
                expected = [row for row in rows if row.capture_ns <= target
                            and row.arrival_ns <= 20*DT_NS and target-row.capture_ns <= 5*DT_NS]
                expected = max(expected, key=lambda row: row.capture_ns) if expected else None
                self.assertEqual(actual, expected)

    def test_statistics_ignore_padding_and_cannot_fit_validation_or_test(self):
        values = current_features(current([frame(12)]), (1., 0., 0., 0.))
        params = fit_current_statistics([values], split="train")
        self.assertEqual(params["count"], 1)
        np.testing.assert_array_equal(params["mean"], values[-1, :GEOMETRY_DIM])
        for split in ("validation", "test"):
            with self.assertRaises(ValueError):
                fit_current_statistics([values], split=split)

    def test_no_support_or_stale_support_requires_explicit_fallback(self):
        for value in (current([]), current([frame(12)], control=18)):
            self.assertFalse(value.can_step)
            with self.assertRaisesRegex(ValueError, "caller must handle fallback"):
                current_features(value, (1., 0., 0., 0.))

    def test_modified_window_cannot_bypass_the_timestamp_gate(self):
        value = current([frame(i) for i in range(13)])
        with self.assertRaisesRegex(ValueError, "visibility boundary"):
            replace(value, samples=value.samples[:-1]+(frame(13),))
        with self.assertRaisesRegex(ValueError, "visibility boundary"):
            replace(value, samples=value.samples[:-1]+(frame(12, arrival=13*DT_NS),))


if __name__ == "__main__":
    unittest.main()
