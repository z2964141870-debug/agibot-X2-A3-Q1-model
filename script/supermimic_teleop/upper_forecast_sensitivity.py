"""Exact iid input-noise propagation for the already selected linear forecast.

This explains an existing pressure result without fitting, changing selection
or reading raw held-out trajectories. Correlated sensor errors, SO(3), contact
and real robot response are outside this scalar-angle linear analysis.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from forecast_data import identity
from upper_forecast import BASELINES, CHANNELS, LocalBudget, save


MODEL_SHA = "be36ef81c0c9564d3b9e0b249d6e5031637dcf9ad86c2310ec4a500dc1e988fd"
RESULT_SHA = "36c391a64ccc28fc5c5eae45d866f94ebe339a82c9d2f8f925a38ecef0f16864"
PLAN_SHA = "f40c1d5a856a08d89f1a8c23dfe6d990fc7439232b9da9a658d9860456353145"


def checked_model(payload):
    if (payload.get("schema") != "q1_upper_linear_model_v1"
            or payload.get("channels") != list(CHANNELS)
            or payload.get("history_frames") != 25 or payload.get("future_frames") != 10
            or payload.get("plan_identity", {}).get("sha256") != PLAN_SHA
            or payload.get("selection", {}).get("best_learned_method") != "ridge_0.01"):
        raise ValueError("Require the previously selected model and exact feature/label schema")
    model = payload["model"]
    if (model.get("fitted_split") != "train" or model.get("training_windows") != 8192
            or model.get("lambda_mean_loss") != .01):
        raise ValueError("Model training provenance differs from the existing experiment")
    arrays = {key: np.asarray(model[key], dtype=np.float64) for key in ("weights", "mean", "scale")}
    if (arrays["weights"].shape != (201, 80) or arrays["mean"].shape != (200,)
            or arrays["scale"].shape != (200,)
            or any(not np.isfinite(value).all() for value in arrays.values())
            or (arrays["scale"] < 1e-6).any()):
        raise ValueError("Invalid physical input normalization or forecast coefficients")
    return dict(model, **arrays)


def learned_affine_map(model):
    weights, mean, scale = (np.asarray(model[key], dtype=np.float64)
                             for key in ("weights", "mean", "scale"))
    if (weights.shape != (201, 80) or mean.shape != (200,) or scale.shape != (200,)
            or not np.isfinite(weights).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0.).any()):
        raise ValueError("Invalid linear forecast dimensions or normalization")
    normalized_weights = weights[:-1]/scale[:, None]
    jacobian = normalized_weights.copy()
    jacobian[192:] -= normalized_weights[:192].reshape(24, 8, 80).sum(axis=0)
    for joint in range(8):
        jacobian[192+joint, np.arange(10)*8+joint] += 1.
    intercept = weights[-1] - np.einsum("i,ij->j", mean/scale, weights[:-1], optimize=False)
    return jacobian, intercept


def baseline_affine_map(method):
    if method not in BASELINES:
        raise ValueError("Unknown fixed baseline")
    jacobian = np.zeros((25, 8, 10, 8))
    for joint in range(8):
        jacobian[24, joint, :, joint] = 1.
    if method != "hold":
        times = np.arange(-4, 1)*.02
        offsets = np.arange(1, 11)*.02
        travel = .12*(-np.expm1(-offsets/.12)) if method == "damped_cv" else offsets
        slopes = times/np.sum(times*times)
        for joint in range(8):
            for index in range(5):
                jacobian[20+index, joint, :, joint] += travel*slopes[index]
            jacobian[24, joint, :, joint] -= travel*slopes.sum()
    return jacobian.reshape(200, 80), np.zeros(80)


def noise_gains(jacobian):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (200, 80) or not np.isfinite(jacobian).all():
        raise ValueError("Require a finite raw-history-to-future Jacobian")
    variance = np.sum(jacobian*jacobian, axis=0).reshape(10, 8)
    return dict(per_joint_per_horizon_std_gain=np.sqrt(variance).tolist(),
        per_horizon_rms_std_gain=np.sqrt(np.mean(variance, axis=1)).tolist(),
        covariance_model="iid zero-mean input errors, all25*8 history coordinates",
        capture_or_robot_latency_measured=False)


def explain(model, previous):
    selected = previous["selection"]["method"]
    if selected != "ridge_0.01" or previous["schema"] != "q1_saved_upper_forecast_v1":
        raise ValueError("Previously frozen selection or original result schema differs")
    maps = {method: baseline_affine_map(method)[0] for method in BASELINES}
    maps[selected] = learned_affine_map(model)[0]
    rows = []
    for method, jacobian in maps.items():
        clean = next(row for row in previous["test"] if row["method"] == method and row["noise_std_rad"] == 0.)
        noisy = next(row for row in previous["test"] if row["method"] == method and row["noise_std_rad"] == .01)
        gains = noise_gains(jacobian)
        profile = np.asarray(gains["per_horizon_rms_std_gain"])
        expected = np.sqrt(np.asarray(clean["per_horizon_rmse_rad"])**2 + (.01*profile)**2)
        observed = np.asarray(noisy["per_horizon_rmse_rad"])
        rows.append(dict(method=method, **gains,
            iid_noise_std_rad=.01, predicted_noise_only_rmse_rad=(.01*profile).tolist(),
            previous_clean_test_rmse_rad=clean["per_horizon_rmse_rad"],
            expected_noisy_test_rmse_rad=expected.tolist(),
            previous_observed_noisy_test_rmse_rad=noisy["per_horizon_rmse_rad"],
            relative_rmse_difference_from_expectation=((observed-expected)/expected).tolist()))
    return dict(schema="q1_upper_linear_noise_sensitivity_v1", status="COMPLETED",
        methods=rows, selected_method_unchanged=selected,
        test_result_reused_for_explanation_only=True, test_used_for_new_selection=False,
        raw_reference_or_actual_trajectories_read=False, new_model_fits=0,
        new_optimizer_updates=0, robot_policy_inference=0, physics_steps=0, cuda_jobs=0,
        model_or_policy_changed=False, measured_robot_latency_ms=None,
        naturalness_conclusion=None,
        limitation="Exact only for this affine predictor and iid additive angle noise; not a sensor-noise calibration")


def execute(args):
    if not args.local_cpu_study or sys.platform != "darwin":
        raise ValueError("Explicit local Mac saved-model coefficient analysis only")
    budget = LocalBudget()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.project_root.resolve() / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent output root")
    model_id, result_id = identity(args.model), identity(args.result)
    if (model_id["sha256"] != MODEL_SHA or result_id["sha256"] != RESULT_SHA
            or model_id["bytes"] > 1024**2 or result_id["bytes"] > 2*1024**2):
        raise ValueError("Require exact bounded artifacts from the completed upper forecast study")
    model = checked_model(json.loads(args.model.read_text()))
    previous = json.loads(args.result.read_text())
    budget.check()
    result = explain(model, previous)
    if identity(args.model) != model_id or identity(args.result) != result_id:
        raise ValueError("Completed source artifacts changed during coefficient analysis")
    result.update(model_identity=model_id, previous_result_identity=result_id,
        code_identity=identity(__file__), resources=budget.check(),
        original_training_modified=False, user_accepted_checkpoint=None)
    output.mkdir(parents=True)
    save(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "result", "project-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--local-cpu-study", action="store_true")
    result = execute(parser.parse_args())
    print(json.dumps(dict(status=result["status"], resources=result["resources"],
        methods=[dict(method=row["method"], gains=row["per_horizon_rms_std_gain"])
                 for row in result["methods"]]), allow_nan=False))
