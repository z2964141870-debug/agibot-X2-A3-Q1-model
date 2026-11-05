"""Independent analytic and event-boundary checks for arrival-clock resampling."""

from dataclasses import replace
import math
import unittest

import numpy as np

from causal_resampler import ArrivedPose, CausalResampler, IntakeSource
from stream_contract import HumanPose


MS = 1_000_000
SOURCE = IntakeSource("synthetic", "one", "sensor_epoch", "receiver_monotonic")


def pose(x=0., angle=0.):
    quaternion = (math.cos(angle/2), 0., 0., math.sin(angle/2))
    return HumanPose([(x, 0., 0.)]*24, [quaternion]*24)


def planar_angle(pose):
    w, x, y, z = pose.orientations[0]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def packet(sequence, arrival_ms, x=0., angle=0., **overrides):
    row = ArrivedPose(SOURCE, sequence, 1_700_000_000_000_000_000+sequence*30*MS,
                      arrival_ms*MS, pose(x, angle))
    return replace(row, **overrides)


def sample(resampler, target_ms, control_ms=None, method="hold"):
    return resampler.sample(target_ns=target_ms*MS,
        control_ns=(target_ms if control_ms is None else control_ms)*MS,
        receiver_clock_id=SOURCE.receiver_clock_id, method=method)


class CausalResamplerTests(unittest.TestCase):
    def test_unrelated_clock_origins_are_retained_never_subtracted(self):
        timelines = [CausalResampler(SOURCE), CausalResampler(SOURCE)]
        for resampler, shift in zip(timelines, (0, 9_000_000_000_000)):
            for seq, ms in enumerate((0, 30, 90)):
                original = packet(seq, ms, ms/1000)
                resampler.ingest(replace(original, payload_ns=original.payload_ns+shift))
        a, b = (sample(row, 100, method="cv") for row in timelines)
        self.assertEqual(a.pose, b.pose)
        self.assertNotEqual(a.timing()["used_payload_ns"], b.timing()["used_payload_ns"])
        self.assertIsNone(a.timing()["measured_end_to_end_latency_ns"])
        self.assertFalse(a.timing()["is_exact_measured_capture"])
        self.assertEqual(a.timing()["latest_support_arrival_age_ns"], 10*MS)

    def test_actual_arrivals_only_and_no_time_rewind_or_future_target(self):
        resampler = CausalResampler(SOURCE)
        resampler.ingest(packet(0, 0))
        before = sample(resampler, 20)
        self.assertEqual(before.provenance, "held_arrival_pose")
        self.assertEqual(before.timing()["used_sequences"], [0])
        resampler.ingest(packet(1, 30, 999.))
        with self.assertRaisesRegex(ValueError, "rewind"):
            sample(resampler, 20)
        with self.assertRaisesRegex(ValueError, "no later"):
            sample(resampler, 40, 30)
        with self.assertRaisesRegex(ValueError, "same-receiver"):
            resampler.sample(target_ns=30*MS, control_ns=30*MS,
                             receiver_clock_id=SOURCE.payload_clock_id)

    def test_buffered_interpolation_declares_right_dependency_and_slerp(self):
        resampler = CausalResampler(SOURCE)
        resampler.ingest(packet(0, 0, 0., np.deg2rad(170)))
        held = sample(resampler, 20, method="delayed_linear")
        self.assertEqual(held.pose.positions[0][0], 0.)
        resampler.ingest(packet(1, 40, .04, np.deg2rad(-170)))
        interpolated = sample(resampler, 20, 40, "delayed_linear")
        self.assertAlmostEqual(interpolated.pose.positions[0][0], .02)
        self.assertAlmostEqual(abs(planar_angle(interpolated.pose)), math.pi)
        self.assertEqual(interpolated.timing()["interpolation_right_support_ns"], 20*MS)
        self.assertEqual(interpolated.timing()["scheduled_buffer_ns"], 20*MS)
        self.assertEqual(interpolated.provenance, "arrived_linear_slerp")
        self.assertTrue(interpolated.timing()["locally_causal"])
        self.assertFalse(interpolated.timing()["is_exact_measured_capture"])

    def test_cv_recovers_analytic_motion_using_irregular_arrival_intervals(self):
        resampler = CausalResampler(SOURCE)
        for sequence, ms in enumerate((0, 11, 49, 90)):
            resampler.ingest(packet(sequence, ms, 2*ms/1000, 3*ms/1000))
        row = sample(resampler, 130, method="cv")
        self.assertAlmostEqual(row.pose.positions[0][0], .26)
        self.assertAlmostEqual(planar_angle(row.pose), .39)
        self.assertTrue(row.generated)
        self.assertFalse(row.speed_cap_applied)
        self.assertEqual(row.timing()["used_sequences"], [0, 1, 2, 3])

    def test_damping_caps_and_dropout_are_bounded(self):
        resampler = CausalResampler(SOURCE, max_linear_speed_m_s=1.,
                                    max_angular_speed_rad_s=1.)
        resampler.ingest(packet(0, 0))
        resampler.ingest(packet(1, 30, 3., 1.))
        row = sample(resampler, 80, method="damped_cv")
        travel = .12*(-np.expm1(-.05/.12))
        self.assertAlmostEqual(row.pose.positions[0][0], 3.+travel)
        self.assertTrue(row.speed_cap_applied)
        limited = sample(resampler, 111, method="damped_cv")
        self.assertEqual(limited.status, "EXTRAPOLATION_LIMIT")
        self.assertFalse(limited.can_step)
        self.assertIsNone(limited.pose)
        stale = sample(resampler, 240, method="hold")
        self.assertEqual(stale.status, "STALE_SUPPORT")
        self.assertFalse(stale.can_step)

    def test_newer_nonused_sample_does_not_refresh_old_hold(self):
        resampler = CausalResampler(SOURCE, max_arrival_age_ns=100*MS)
        resampler.ingest(packet(0, 0))
        resampler.ingest(packet(1, 200, .2))
        row = sample(resampler, 20, 200)
        self.assertFalse(row.can_step)
        self.assertEqual(row.timing()["used_sequences"], [0])
        self.assertEqual(row.timing()["latest_support_arrival_age_ns"], 200*MS)

    def test_gap_guard_and_missing_history_request_explicit_behavior(self):
        resampler = CausalResampler(SOURCE)
        self.assertFalse(sample(resampler, 0).can_step)
        resampler.ingest(packet(0, 0))
        first = sample(resampler, 10, method="cv")
        self.assertEqual(first.status, "INSUFFICIENT_HISTORY_HELD")
        self.assertFalse(first.generated)
        resampler.ingest(packet(1, 300, .3))
        gap = sample(resampler, 150, 300, "delayed_linear")
        self.assertEqual(gap.status, "SUPPORT_GAP_LIMIT")
        self.assertFalse(gap.can_step)
        after = sample(resampler, 310, method="cv")
        self.assertEqual(after.status, "INSUFFICIENT_HISTORY_HELD")
        self.assertEqual(after.pose.positions[0][0], .3)

    def test_duplicate_reorder_conflict_and_capacity_do_not_regress_pose(self):
        resampler = CausalResampler(SOURCE, capacity=2, history_frames=2)
        resampler.ingest(packet(0, 0))
        resampler.ingest(packet(2, 30, .2))
        self.assertEqual(resampler.ingest(packet(1, 40, .1)), "REORDERED_DROPPED")
        self.assertEqual(resampler.ingest(packet(2, 50, .2)), "DUPLICATE")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resampler.ingest(packet(2, 60, .3))
        resampler.ingest(packet(3, 60, .3))
        self.assertEqual(sample(resampler, 60).pose.positions[0][0], .3)
        self.assertEqual(resampler.counters, dict(received=5, duplicates=1, reordered_drops=1, evicted=1))
        self.assertEqual(len(resampler.frames), 2)

    def test_invalid_clock_reset_parameters_and_dependency_are_rejected(self):
        resampler = CausalResampler(SOURCE)
        resampler.ingest(packet(0, 0))
        with self.assertRaisesRegex(ValueError, "clock discontinuity"):
            resampler.ingest(packet(1, 30, payload_ns=1))
        with self.assertRaisesRegex(ValueError, "session"):
            resampler.ingest(packet(1, 30, source=replace(SOURCE, session_id="two")))
        with self.assertRaisesRegex(ValueError, "exact payload dependency"):
            packet(0, 0, source=replace(SOURCE, preprocessing="declared_causal"))
        with self.assertRaisesRegex(ValueError, "Receiver-ready"):
            packet(0, 0, receiver_ready_ns=1)
        for kwargs in (dict(tau_s=float("nan")), dict(capacity=1),
                       dict(max_linear_speed_m_s=0.), dict(max_extrapolation_ns=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CausalResampler(SOURCE, **kwargs)

    def test_same_arrival_batch_uses_newest_and_never_divides_by_zero(self):
        resampler = CausalResampler(SOURCE)
        resampler.ingest(packet(0, 0))
        resampler.ingest(packet(1, 0, .1))
        exact = sample(resampler, 0)
        self.assertEqual(exact.pose.positions[0][0], .1)
        self.assertEqual(exact.timing()["used_sequences"], [1])
        held = sample(resampler, 10, method="cv")
        self.assertEqual(held.status, "INSUFFICIENT_HISTORY_HELD")
        self.assertEqual(held.pose.positions[0][0], .1)


if __name__ == "__main__":
    unittest.main()
