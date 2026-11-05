"""Behavioral checks for the preview receiver clock and causal visibility boundary."""

from dataclasses import replace
import random
import unittest

from stream_contract import (DT_NS, HumanPose, PoseFrame, PoseTimeline, SourceIdentity,
                             command_timing)


SOURCE = SourceIdentity("synthetic_smpl24", "test_session", "test_receiver_ns",
                        "smpl24_world_m_wxyz", "declared_causal")


def sample(index, arrival_ns=None, marker=None):
    capture = index * DT_NS
    pose = HumanPose([[index if marker is None else marker, 0., 1.]] * 24,
                     [[1., 0., 0., 0.]] * 24)
    return PoseFrame(SOURCE, index, capture, capture, capture,
                     capture if arrival_ns is None else arrival_ns, pose)


def window(timeline, frames, control_ns=None, max_age=100_000_000):
    return timeline.window(origin_ns=0, control_ns=frames*DT_NS if control_ns is None else control_ns,
                           preview_frames=frames, max_capture_age_ns=max_age)


class StreamContractTests(unittest.TestCase):
    def test_each_horizon_has_only_its_visible_prefix_and_masked_tail(self):
        for horizon in (10, 5, 4, 2, 1):
            with self.subTest(horizon=horizon):
                timeline = PoseTimeline(SOURCE)
                for i in range(horizon+1):
                    timeline.ingest(sample(i))
                result = window(timeline, horizon)
                self.assertEqual(result.status, "READY")
                self.assertEqual(result.used_sequences, tuple(range(1, horizon+1)))
                self.assertEqual(result.measured_mask, (True,)*horizon + (False,)*(10-horizon))
                self.assertEqual(result.policy_frame_mask, result.measured_mask)
                self.assertEqual([pose.positions[0][0] for pose in result.poses],
                                 list(range(1, horizon+1)) + [horizon]*(10-horizon))
                self.assertEqual(result.timing()["scheduled_origin_to_control_ns"], horizon*DT_NS)
                self.assertIsNone(result.timing()["measured_end_to_end_latency_ns"])

    def test_preloaded_future_and_receiver_rewind_are_rejected(self):
        timeline = PoseTimeline(SOURCE)
        timeline.ingest(sample(5, marker=1e6))
        with self.assertRaisesRegex(ValueError, "preload future"):
            window(timeline, 4)
        timeline = PoseTimeline(SOURCE)
        timeline.ingest(sample(0))
        window(timeline, 1)
        with self.assertRaisesRegex(ValueError, "rewind"):
            timeline.ingest(sample(0, arrival_ns=DT_NS-1))

    def test_transport_loss_and_reordering_reduce_prefix_without_reading_ahead(self):
        timeline = PoseTimeline(SOURCE)
        for i, arrival in ((0, 0), (1, DT_NS), (3, 3*DT_NS), (2, 4*DT_NS)):
            timeline.ingest(sample(i, arrival_ns=arrival))
        result = window(timeline, 5, control_ns=5*DT_NS)
        self.assertEqual(result.status, "GAP_AFTER_PREFIX")
        self.assertEqual(result.prefix_frames, 3)
        self.assertTrue(result.can_step)
        self.assertEqual(timeline.counters["reordered"], 1)
        self.assertEqual(result.poses[-1].positions[0][0], 3)
        lost_first = PoseTimeline(SOURCE)
        lost_first.ingest(sample(2))
        result = window(lost_first, 4)
        self.assertFalse(result.can_step)
        self.assertEqual(result.status, "MISSING_FIRST_TARGET")
        self.assertEqual(result.poses, (None,)*10)

    def test_budget_cutoff_still_applies_when_later_captures_have_arrived(self):
        timeline = PoseTimeline(SOURCE)
        for i in range(11):
            timeline.ingest(sample(i, marker=1e6 if i > 4 else i))
        result = window(timeline, 4, control_ns=10*DT_NS, max_age=6*DT_NS)
        self.assertEqual(result.prefix_frames, 4)
        self.assertEqual(result.newest_capture_ns, 4*DT_NS)
        self.assertEqual(result.poses[-1].positions[0][0], 4)
        self.assertEqual(result.timing()["latest_capture_age_ns"], 6*DT_NS)
        self.assertTrue(result.can_step)
        stale = window(timeline, 4, control_ns=10*DT_NS+1, max_age=6*DT_NS)
        self.assertEqual(stale.status, "STALE")
        self.assertFalse(stale.can_step)
        self.assertEqual(stale.policy_frame_mask, (False,)*10)
        with self.assertRaisesRegex(ValueError, "unavailable/stale"):
            command_timing(stale, policy_start_ns=10*DT_NS+1,
                           policy_end_ns=10*DT_NS+2, command_sent_ns=10*DT_NS+3)

    def test_recent_pose_beyond_a_gap_cannot_refresh_the_used_prefix(self):
        timeline = PoseTimeline(SOURCE)
        timeline.ingest(sample(1))
        timeline.ingest(sample(5))
        result = window(timeline, 5, max_age=DT_NS)
        self.assertEqual(result.newest_capture_ns, 5*DT_NS)
        self.assertEqual(result.latest_used_capture_ns, DT_NS)
        self.assertEqual(result.status, "STALE")
        self.assertFalse(result.can_step)

    def test_bounded_storage_duplicates_and_conflicting_identity(self):
        timeline = PoseTimeline(SOURCE, capacity=3)
        first = sample(0)
        timeline.ingest(first)
        self.assertEqual(timeline.ingest(replace(first, arrival_ns=1)), "DUPLICATE")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            timeline.ingest(replace(first, pose=sample(0, marker=99).pose, arrival_ns=2))
        for i in range(1, 4):
            timeline.ingest(sample(i))
        self.assertEqual(len(timeline.frames), 3)
        self.assertEqual(len(timeline.by_sequence), 3)
        self.assertEqual(timeline.counters["overflow_evictions"], 1)
        self.assertEqual(timeline.ingest(replace(first, arrival_ns=4*DT_NS)), "TOO_OLD")
        other = replace(SOURCE, session_id="different_session")
        with self.assertRaisesRegex(ValueError, "Wrong source/session/clock"):
            timeline.ingest(replace(sample(5), source=other))
        self.assertLessEqual(len(timeline.frames), timeline.capacity)

    def test_pose_copy_and_timestamp_dependencies_are_explicit(self):
        positions, quats = [[0., 0., 1.]]*24, [[1., 0., 0., 0.]]*24
        pose = HumanPose(positions, quats)
        positions[0][0] = 9.
        self.assertEqual(pose.positions[0][0], 0.)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            HumanPose([[float("nan"), 0., 1.]]*24, quats)
        with self.assertRaisesRegex(ValueError, "normalized"):
            HumanPose([[0., 0., 1.]]*24, [[2., 0., 0., 0.]]*24)
        with self.assertRaisesRegex(ValueError, "Declared causal"):
            replace(sample(0), dependency_end_ns=DT_NS, ready_ns=DT_NS, arrival_ns=DT_NS)
        with self.assertRaisesRegex(ValueError, "capture <="):
            replace(sample(1), arrival_ns=0)
        with self.assertRaises(ValueError):
            replace(sample(0), capture_ns=True)

    def test_randomized_visibility_matches_independent_arrival_oracle(self):
        rng = random.Random(13)
        for _ in range(40):
            horizon = rng.choice((1, 2, 4, 5, 10))
            control = horizon*DT_NS + rng.randrange(4)*DT_NS
            rows = [sample(i, arrival_ns=i*DT_NS+rng.randrange(5)*DT_NS)
                    for i in range(18) if rng.random() > .15]
            arrived = sorted((row for row in rows if row.arrival_ns <= control),
                             key=lambda row: row.arrival_ns)
            timeline = PoseTimeline(SOURCE)
            for row in arrived:
                timeline.ingest(row)
            result = window(timeline, horizon, control_ns=control, max_age=10*DT_NS)
            allowed = {row.capture_ns: row for row in arrived
                       if row.capture_ns <= horizon*DT_NS}
            expected = []
            for i in range(1, horizon+1):
                if i*DT_NS not in allowed:
                    break
                expected.append(allowed[i*DT_NS].sequence)
            self.assertEqual(result.used_sequences, tuple(expected))
            self.assertEqual(result.prefix_frames, len(expected))
            self.assertTrue(all(value <= control for value in
                                (row.arrival_ns for row in arrived)))

    def test_software_latency_does_not_claim_physical_latency(self):
        timeline = PoseTimeline(SOURCE)
        for i in range(5):
            timeline.ingest(sample(i))
        result = window(timeline, 4)
        timing = command_timing(result, policy_start_ns=80_000_100,
                                policy_end_ns=82_000_100, command_sent_ns=83_000_100)
        self.assertEqual(timing["policy_compute_ns"], 2_000_000)
        self.assertEqual(timing["scheduled_origin_to_send_ns"], 83_000_100)
        self.assertIsNone(timing["physical_actuation_time_ns"])
        with self.assertRaisesRegex(ValueError, "control <= policy"):
            command_timing(result, policy_start_ns=79_000_000,
                           policy_end_ns=82_000_000, command_sent_ns=83_000_000)

    def test_offline_clip_end_is_explicit_and_not_a_measured_future(self):
        offline = replace(SOURCE, preprocessing="offline_unverified")
        timeline = PoseTimeline(offline)
        for i in range(4):
            timeline.ingest(replace(sample(i), source=offline, dependency_end_ns=None,
                                    terminal=i == 3))
        result = window(timeline, 10, max_age=7*DT_NS)
        self.assertTrue(result.can_step)
        self.assertEqual(result.prefix_frames, 10)
        self.assertEqual(result.measured_mask, (True,)*3 + (False,)*7)
        self.assertEqual(result.policy_frame_mask, (True,)*10)
        self.assertEqual(result.provenance[-1], "declared_clip_terminal_hold")
        self.assertEqual(result.timing()["observed_prefix_frames"], 3)
        self.assertEqual(result.timing()["policy_prefix_frames"], 10)
        self.assertIsNone(result.max_used_dependency_ns)
        with self.assertRaisesRegex(ValueError, "follows a declared end"):
            timeline.ingest(replace(sample(11), source=offline))
        with self.assertRaisesRegex(ValueError, "restricted to offline"):
            replace(sample(1), terminal=True)


if __name__ == "__main__":
    unittest.main()
