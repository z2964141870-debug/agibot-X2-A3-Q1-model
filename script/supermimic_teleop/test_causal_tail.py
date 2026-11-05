"""Analytic and visibility checks for position/SO(3) causal tail baselines."""

from dataclasses import replace
import math
import unittest

from causal_tail import forecast_window, validate_forecast
from stream_contract import DT_NS, HumanPose, PoseTimeline
from test_stream_contract import SOURCE, sample, window


def linear_window(horizon=4, yaw0=0., flip=False):
    timeline = PoseTimeline(SOURCE)
    for index in range(horizon+1):
        seconds = index*.02
        yaw = yaw0+seconds*2
        q = (math.cos(yaw/2), 0., 0., math.sin(yaw/2))
        if flip and index % 2:
            q = tuple(-value for value in q)
        pose = HumanPose([(seconds, seconds*2, 1.)]*24, [q]*24)
        timeline.ingest(replace(sample(index), pose=pose))
    return window(timeline, horizon)


class CausalTailTests(unittest.TestCase):
    def test_constant_velocity_and_rotation_keep_original_ten_slots(self):
        observed = linear_window()
        forecast = validate_forecast(forecast_window(observed, "cv"))
        self.assertEqual(forecast.observed.requested_frames, 4)
        self.assertEqual(forecast.measured_mask, (True,)*4+(False,)*6)
        self.assertEqual(forecast.generated_mask, (False,)*4+(True,)*6)
        for index, pose in enumerate(forecast.poses, 1):
            seconds = index*.02
            self.assertAlmostEqual(pose.positions[0][0], seconds, places=12)
            self.assertAlmostEqual(pose.positions[0][1], seconds*2, places=12)
            self.assertAlmostEqual(pose.orientations[0][0], math.cos(seconds), places=12)
            self.assertAlmostEqual(pose.orientations[0][3], math.sin(seconds), places=12)
        self.assertEqual(forecast.timing()["requested_preview_ns"], 80_000_000)
        self.assertEqual(forecast.timing()["generated_slots"], 6)
        self.assertIsNone(forecast.timing()["measured_end_to_end_latency_ns"])

    def test_quaternion_sign_and_pi_boundary_do_not_create_spurious_spin(self):
        ordinary = forecast_window(linear_window(yaw0=math.pi-.05), "cv")
        flipped = forecast_window(linear_window(yaw0=math.pi-.05, flip=True), "cv")
        for a, b in zip(ordinary.poses, flipped.poses):
            self.assertAlmostEqual(abs(sum(x*y for x, y in zip(a.orientations[0], b.orientations[0]))), 1., places=12)
        expected = (math.cos((math.pi-.05+.4)/2), 0., 0., math.sin((math.pi-.05+.4)/2))
        self.assertAlmostEqual(abs(sum(x*y for x, y in zip(ordinary.poses[-1].orientations[0], expected))), 1., places=12)

    def test_hidden_tail_poison_does_not_change_any_forecast(self):
        observed = linear_window()
        poisoned = replace(observed, poses=observed.poses[:4] + (sample(999).pose,)*6)
        for method in ("hold", "cv", "damped_cv"):
            with self.subTest(method=method):
                self.assertEqual(forecast_window(observed, method).poses,
                                 forecast_window(poisoned, method).poses)

    def test_damped_displacement_and_linear_angular_caps(self):
        observed = linear_window()
        forecast = forecast_window(observed, "damped_cv", tau_s=.12)
        travel = .12*(1-math.exp(-.12/.12))
        self.assertAlmostEqual(forecast.poses[-1].positions[0][0], .08+travel, places=12)
        self.assertAlmostEqual(forecast.poses[-1].orientations[0][3], math.sin((.16+2*travel)/2), places=12)
        limited = forecast_window(observed, "cv", max_linear_speed_m_s=.1, max_angular_speed_rad_s=.2)
        self.assertAlmostEqual(math.dist(limited.poses[-1].positions[0], observed.poses[3].positions[0]), .1*.12, places=12)
        dot = abs(sum(a*b for a, b in zip(limited.poses[-1].orientations[0], observed.poses[3].orientations[0])))
        self.assertAlmostEqual(2*math.acos(dot), .2*.12, places=10)

    def test_hold_h1_and_h10_are_truthful_and_prefix_is_immutable(self):
        one = linear_window(1)
        held = forecast_window(one, "hold")
        self.assertEqual(held.poses, (one.poses[0],)*10)
        with self.assertRaisesRegex(ValueError, "two measured"):
            forecast_window(one, "cv")
        full = linear_window(10)
        self.assertEqual(forecast_window(full).poses, full.poses)
        self.assertEqual(forecast_window(full).timing()["generated_slots"], 0)
        self.assertEqual(one.requested_frames, 1)

    def test_terminal_event_holds_without_claiming_predicted_or_measured_future(self):
        source = replace(SOURCE, preprocessing="offline_unverified")
        timeline = PoseTimeline(source)
        for index in range(4):
            timeline.ingest(replace(sample(index), source=source, dependency_end_ns=None, terminal=index == 3))
        forecast = validate_forecast(forecast_window(window(timeline, 4)))
        self.assertEqual(forecast.method, "terminal_hold")
        self.assertEqual(forecast.generated_mask, (False,)*10)
        self.assertEqual(sum(forecast.measured_mask), 3)
        self.assertEqual(forecast.provenance[-1], "declared_clip_terminal_hold")

    def test_terminal_exactly_at_preview_boundary_does_not_extrapolate_past_clip_end(self):
        source = replace(SOURCE, preprocessing="offline_unverified")
        timeline = PoseTimeline(source)
        for index in range(5):
            timeline.ingest(replace(sample(index), source=source,
                                    dependency_end_ns=None, terminal=index == 4))
        observed = window(timeline, 4)
        self.assertEqual(observed.provenance[:4], ("measured_exact",)*4)
        self.assertEqual(observed.terminal_capture_ns, 4*DT_NS)
        for method in ("hold", "cv", "damped_cv"):
            forecast = validate_forecast(forecast_window(observed, method))
            self.assertEqual(forecast.method, "terminal_hold")
            self.assertEqual(forecast.poses[4:], (observed.poses[3],)*6)
            self.assertEqual(sum(forecast.generated_mask), 0)

    def test_invalid_parameters_stale_goals_and_false_provenance_rejected(self):
        observed = linear_window()
        for kwargs in (dict(method="unknown"), dict(tau_s=0), dict(max_linear_speed_m_s=-1),
                       dict(max_angular_speed_rad_s=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                forecast_window(observed, **kwargs)
        with self.assertRaisesRegex(ValueError, "fresh available"):
            forecast_window(replace(observed, can_step=False))
        forecast = forecast_window(observed)
        for bad in (replace(forecast, measured_mask=(True,)*10),
                    replace(forecast, generated_mask=(False,)*10),
                    replace(forecast, method="terminal_hold"),
                    replace(forecast, tau_s=float("nan")),
                    replace(forecast, poses=(sample(999).pose,)*10)):
            with self.assertRaises(ValueError):
                validate_forecast(bad)


if __name__ == "__main__":
    unittest.main()
