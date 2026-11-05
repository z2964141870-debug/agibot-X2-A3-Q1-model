"""Causal baseline and evaluation aggregation checks."""

import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from forecast_data import window_items
from forecast_eval import baseline_targets, evaluate, execute, _new_accumulator, _add, _finish, _validate_payload
from forecast_geometry import decode_targets, encode_history, encode_targets, exp_rotation
from forecast_model import ForecastSpec, NormalizationFit
from test_forecast_model import training_window
from test_forecast_geometry import motion


class ForecastEvaluationTests(unittest.TestCase):
    def test_hold_baseline_is_exactly_the_visible_pose(self):
        positions, quaternions = motion()
        labels = baseline_targets(positions[:25], quaternions[:25], "hold")
        decoded = decode_targets(labels, encode_history(positions[:25], quaternions[:25]).anchor)
        np.testing.assert_allclose(decoded.positions, np.repeat(positions[24][None], 10, axis=0), atol=3e-15)
        np.testing.assert_allclose(decoded.quaternions, np.repeat(quaternions[24][None], 10, axis=0), atol=3e-15)

    def test_constant_velocity_baseline_reconstructs_linear_motion(self):
        positions = np.zeros((40, 24, 3))
        positions[:, :, 0] = np.arange(40)[:, None] * .02
        positions[:, :, 1] = np.arange(24)[None] * .03
        positions[:, :, 2] = .8
        quaternions = np.repeat(exp_rotation(np.arange(40)[:, None]*.02*
                                            np.array([.2, -.1, .3]))[:, None], 24, axis=1)
        labels = baseline_targets(positions[:25], quaternions[:25], "cv",
                                  max_linear_speed_m_s=None, max_angular_speed_rad_s=None)
        decoded = decode_targets(labels, encode_history(positions[:25], quaternions[:25]).anchor)
        np.testing.assert_allclose(decoded.positions, positions[25:35], atol=3e-13)
        np.testing.assert_allclose(decoded.quaternions, quaternions[25:35], atol=3e-13)
        damped = decode_targets(baseline_targets(positions[:25], quaternions[:25], "damped_cv"),
                                 encode_history(positions[:25], quaternions[:25]).anchor)
        travel = .12*(-np.expm1(-np.arange(1, 11)*.02/.12))
        np.testing.assert_allclose(damped.positions[:, 0, 0], positions[24, 0, 0]+travel, atol=3e-13)

    def test_aggregation_reports_zero_for_exact_prediction(self):
        positions, quaternions = motion(35)
        prediction = decode_targets(
            encode_targets(
                positions[25:35], quaternions[25:35],
                encode_history(positions[:25], quaternions[:25]).anchor),
            encode_history(positions[:25], quaternions[:25]).anchor)
        accumulator = _new_accumulator(10)
        _add(accumulator, prediction, positions[25:35], quaternions[25:35], "fixture")
        result = _finish(accumulator, "fixture", "test")
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["sources"], 1)
        self.assertLess(max(result["position_rmse_m"]), 3e-14)
        self.assertLess(max(result["orientation_geodesic_rmse_rad"]), 3e-14)

    def test_known_three_cm_error_uses_euclidean_not_per_coordinate_rmse(self):
        positions, quaternions = motion(35)
        anchor = encode_history(positions[:25], quaternions[:25]).anchor
        expected = positions[25:35]
        prediction = decode_targets(encode_targets(expected+[.01, .02, .02], quaternions[25:35], anchor), anchor)
        accumulator = _new_accumulator(10)
        _add(accumulator, prediction, expected, quaternions[25:35], "fixture")
        result = _finish(accumulator, "known_offset", "test")
        np.testing.assert_allclose(result["position_rmse_m"], .03, atol=3e-15)
        np.testing.assert_allclose(result["root_position_rmse_m"], .03, atol=3e-15)
        with self.assertRaisesRegex(ValueError, "No evaluation windows"):
            _finish(_new_accumulator(10), "hold", "test")

    def test_bounded_sampling_covers_sources_and_each_entire_clip_time_range(self):
        plan = dict(history_frames=25, future_frames=10, clips=[
            dict(relative=f"{i}/clip.npz", split="validation", frames=135+i*100) for i in range(4)])
        selected = window_items(plan, "validation", 24)
        self.assertEqual(len(selected), 24)
        for row in plan["clips"]:
            cutoffs = [cutoff for relative, cutoff in selected if relative == row["relative"]]
            self.assertEqual(len(cutoffs), 6)
            self.assertEqual(cutoffs[0], 24)
            self.assertEqual(cutoffs[-1], row["frames"]-11)
        plan["clips"].reverse()
        self.assertEqual(selected, window_items(plan, "validation", 24))
        self.assertEqual(len(window_items(plan, "validation", 2)), 2)
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                window_items(plan, "validation", limit)

    def test_selection_is_frozen_before_test_and_never_follows_test_preference(self):
        plan = dict(history_frames=25, future_frames=10, clips=[
            dict(relative="v/clip.npz", split="validation", frames=35),
            dict(relative="t/clip.npz", split="test", frames=35)])
        selected, reads = [], []
        times = np.arange(35)*.02
        positions = np.zeros((35, 24, 3))
        positions[:, :, 0] = times[:, None]
        quaternions = np.zeros((35, 24, 4))
        quaternions[:, :, 0] = 1.

        class Corpus:
            def window(self, relative, cutoff, *, split):
                if split == "test":
                    self_test.assertTrue(selected, "Test was read before validation selection was frozen")
                reads.append(split)
                targets = (np.repeat(positions[24:25], 10, axis=0) if split == "validation"
                           else positions[25:35])
                return SimpleNamespace(input_positions=positions[:25], input_quaternions=quaternions[:25],
                    target_positions=targets, target_quaternions=quaternions[25:35])

        self_test = self
        result = evaluate(plan, Corpus(), on_selection=selected.append)
        self.assertEqual(reads, ["validation", "test"])
        self.assertEqual(selected[0]["method"], "hold")
        self.assertEqual(result["selected_method"], "hold")
        test_best = min(result["test"], key=lambda row: sum(row["position_rmse_m"]))
        self.assertEqual(test_best["method"], "cv")
        self.assertFalse(result["test_used_for_selection"])
        self.assertIsNone(result["robot_naturalness_conclusion"])
        self.assertEqual(result["sampling"]["splits"]["validation"]["windows"], 1)

    def test_baselines_reject_bad_caps_and_noncausal_history_length(self):
        positions, quaternions = motion()
        for kwargs in (dict(max_linear_speed_m_s=-1), dict(max_angular_speed_rad_s=float("nan")),
                       dict(tau_s=float("inf")), dict(horizon=0)):
            with self.assertRaises(ValueError):
                baseline_targets(positions[:25], quaternions[:25], **kwargs)
        with self.assertRaisesRegex(ValueError, "25 visible"):
            baseline_targets(positions[:26], quaternions[:26])

    def test_payload_rejects_wrong_time_schema_split_heldout_normalization_and_zero_update(self):
        window = training_window()
        fit = NormalizationFit()
        fit.update(window)
        plan = dict(clips=[dict(relative=window.relative, split="train")])
        plan_id = dict(bytes=123, sha256="a"*64)
        payload = dict(spec=ForecastSpec().to_dict(), plan_identity=plan_id,
            status="COMPLETED", optimizer_updates=10, normalization=fit.freeze().to_dict())
        _validate_payload(payload, plan, plan_id)
        for field, value in (("optimizer_updates", 0), ("status", "RUNNING"),
                             ("plan_identity", dict(bytes=124, sha256="b"*64))):
            altered = deepcopy(payload)
            altered[field] = value
            with self.assertRaises(ValueError):
                _validate_payload(altered, plan, plan_id)
        altered = deepcopy(payload)
        altered["spec"]["horizon_frames"] = 5
        with self.assertRaises(ValueError):
            _validate_payload(altered, plan, plan_id)
        altered = deepcopy(payload)
        altered["normalization"]["fitted_sources"] = ["test/fixture.npz"]
        with self.assertRaises(ValueError):
            _validate_payload(altered, plan, plan_id)

    def test_resource_failure_precedes_loading_data_or_gpu_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(project_root=root, output=root/"data/supermimic_for_q1/eval/result.json",
                max_windows=2, max_seconds=10., expected_gpu="gpu", expected_boot="boot",
                plan=root/"plan.json", device="cpu", checkpoint=None)
            with patch("forecast_eval.identity", return_value=dict(bytes=1, sha256="a"*64)), \
                    patch("forecast_eval.ForecastBudget") as budget, \
                    patch("forecast_eval.load_plan") as load, patch("forecast_eval._load_gru") as model:
                budget.return_value.check.side_effect = RuntimeError("memory reserve")
                with self.assertRaisesRegex(RuntimeError, "memory reserve"):
                    execute(args)
                load.assert_not_called()
                model.assert_not_called()
            marker = args.output.parent/"status.json"
            before = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, "new independent"):
                execute(args)
            self.assertEqual(marker.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
