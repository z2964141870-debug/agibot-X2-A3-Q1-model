"""Bounded, history-only forecast comparisons in physical units.

Freeze the choice using validation before reading test labels. These are pose
forecast errors, not measurements of robot latency, contact or naturalness.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from forecast_data import ForecastCorpus, identity, load_plan, window_items
from forecast_geometry import (DT_NS, checked_poses, decode_targets, encode_history,
    encode_targets, exp_rotation, inverse, log_rotation, product)
from forecast_model import ForecastNormalization, ForecastSpec, make_model
from train_forecast import (DEFAULT_BOOT, DEFAULT_GPU, FORMAL_TRAINING,
                            ForecastBudget, gpu_lock, write_json)


DT_S = DT_NS / 1e9
METHODS = ("hold", "cv", "damped_cv")
METRICS = ("position_rmse_m", "root_position_rmse_m", "orientation_geodesic_rmse_rad")


def _anchored_slope(values):
    times = (np.arange(len(values), dtype=np.float64) - (len(values)-1)) * DT_S
    return np.sum(times[:, None, None] * (values-values[-1]), axis=0) / np.sum(times*times)


def _cap(values, maximum):
    if maximum is None:
        return values
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values * np.minimum(1., maximum / np.maximum(norms, 1e-12))


def baseline_targets(history_positions, history_quaternions, method="damped_cv",
                     *, horizon=10, tau_s=.12, max_linear_speed_m_s=3.,
                     max_angular_speed_rad_s=8.):
    """Predict from the latest five visible samples, using the GRU label schema."""
    if method not in METHODS or type(horizon) is not int or not 1 <= horizon <= 10:
        raise ValueError("Require a known method and horizon in 1..10")
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("Require a finite positive damping time")
    for maximum in (max_linear_speed_m_s, max_angular_speed_rad_s):
        if maximum is not None and (not math.isfinite(maximum) or maximum <= 0):
            raise ValueError("Speed caps must be finite and positive")
    positions, quaternions = checked_poses(history_positions, history_quaternions)
    if len(positions) != 25:
        raise ValueError("Require exactly the frozen 25 visible history frames")
    history = encode_history(positions, quaternions)
    last_positions, last_quaternions = positions[-1], quaternions[-1]
    if method == "hold":
        predicted_positions = np.repeat(last_positions[None], horizon, axis=0)
        predicted_quaternions = np.repeat(last_quaternions[None], horizon, axis=0)
    else:
        linear = _cap(_anchored_slope(positions[-5:]), max_linear_speed_m_s)
        relative = product(inverse(last_quaternions), quaternions[-5:])
        angular = _cap(_anchored_slope(log_rotation(relative)), max_angular_speed_rad_s)
        offsets = np.arange(1, horizon+1, dtype=np.float64) * DT_S
        travel = tau_s * (-np.expm1(-offsets/tau_s)) if method == "damped_cv" else offsets
        predicted_positions = last_positions[None] + travel[:, None, None] * linear[None]
        predicted_quaternions = product(last_quaternions[None],
                                         exp_rotation(travel[:, None, None]*angular[None]))
    return encode_targets(predicted_positions, predicted_quaternions, history.anchor)


def _new_accumulator(horizon):
    return dict(samples=0, position_sq=np.zeros(horizon), root_sq=np.zeros(horizon),
                orientation_sq=np.zeros(horizon), sources=set())


def _add(accumulator, prediction, target_positions, target_quaternions, source):
    positions, quaternions = checked_poses(target_positions, target_quaternions)
    if prediction.positions.shape != positions.shape:
        raise ValueError("Forecast and labels must have identical, unshifted time slots")
    position = prediction.positions-positions
    angles = np.linalg.norm(log_rotation(product(inverse(quaternions),
                                                 prediction.quaternions)), axis=-1)
    accumulator["samples"] += 1
    accumulator["position_sq"] += np.sum(position*position, axis=(1, 2))
    accumulator["root_sq"] += np.sum(position[:, 0]**2, axis=1)
    accumulator["orientation_sq"] += np.sum(angles*angles, axis=1)
    accumulator["sources"].add(source)


def _finish(accumulator, method, split):
    count = accumulator["samples"]
    if not count:
        raise ValueError("No evaluation windows; cannot report zero error")
    return dict(method=method, split=split, samples=count, sources=len(accumulator["sources"]),
                position_rmse_m=np.sqrt(accumulator["position_sq"]/(count*24)).tolist(),
                root_position_rmse_m=np.sqrt(accumulator["root_sq"]/count).tolist(),
                orientation_geodesic_rmse_rad=np.sqrt(accumulator["orientation_sq"]/(count*24)).tolist(),
                position_metric="RMS Euclidean distance per joint; root uses Euclidean distance",
                target_offsets_ns=[(i+1)*DT_NS for i in range(len(accumulator["position_sq"]))],
                temporal_realignment_applied=False, bone_contact_constraints_applied=False,
                measured_robot_latency_ns=None)


def _validate_payload(payload, plan, plan_identity):
    spec = ForecastSpec(hidden_size=payload.get("spec", {}).get("hidden_size", 128))
    if (payload.get("spec") != spec.to_dict() or payload.get("plan_identity") != plan_identity
            or payload.get("status") != "COMPLETED"
            or type(payload.get("optimizer_updates")) is not int or payload["optimizer_updates"] <= 0):
        raise ValueError("Checkpoint schema, frozen split or completed training identity differs")
    allowed = {row["relative"] for row in plan["clips"] if row["split"] == "train"}
    normalization = ForecastNormalization.from_dict(payload["normalization"], allowed)
    if set(normalization.fitted_sources) != allowed:
        raise ValueError("Checkpoint normalization did not cover the declared training sources")
    return spec, normalization


def _load_gru(checkpoint, plan, plan_identity, device):
    import torch
    before = identity(checkpoint)
    if before["bytes"] > 32*1024**2:
        raise ValueError("Forecast checkpoint exceeds its bounded model size")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if identity(checkpoint) != before:
        raise ValueError("Checkpoint changed while loading")
    spec, normalization = _validate_payload(payload, plan, plan_identity)
    state = payload["model"]
    if not state or any(not torch.is_tensor(tensor) or not torch.isfinite(tensor).all().item()
                        for tensor in state.values()):
        raise ValueError("Checkpoint has missing or nonfinite forecast tensors")
    model = make_model(spec)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    def predict(history):
        inputs = torch.from_numpy(normalization.inputs(history.features)[None]).to(device)
        with torch.inference_mode():
            values = model(inputs).cpu().numpy()[0]
        return normalization.physical(values)

    return predict, before


def _evaluate_split(corpus, plan, split, items, learned_predictor, budget_check):
    methods = (*METHODS, "gru") if learned_predictor is not None else METHODS
    totals = {method: _new_accumulator(plan["future_frames"]) for method in methods}
    clips = {method: {} for method in methods}
    for relative, cutoff in items:
        budget_check()
        window = corpus.window(relative, cutoff, split=split)
        history = encode_history(window.input_positions, window.input_quaternions)
        for method in methods:
            labels = (learned_predictor(history) if method == "gru" else
                      baseline_targets(window.input_positions, window.input_quaternions, method))
            prediction = decode_targets(labels, history.anchor)
            clip = clips[method].setdefault(relative, _new_accumulator(plan["future_frames"]))
            for accumulator in (totals[method], clip):
                _add(accumulator, prediction, window.target_positions, window.target_quaternions, relative)
    reports = []
    for method in methods:
        row = _finish(totals[method], method, split)
        row["per_clip"] = [dict(relative=relative, **_finish(value, method, split))
                           for relative, value in clips[method].items()]
        reports.append(row)
    return reports


def evaluate(plan, corpus, *, learned_predictor=None, max_windows=2048,
             selection_metric="position_rmse_m", budget_check=lambda: None,
             on_selection=lambda value: None):
    if selection_metric not in METRICS:
        raise ValueError("Unknown validation selection metric")
    items = {split: window_items(plan, split, max_windows) for split in ("validation", "test")}
    validation = _evaluate_split(corpus, plan, "validation", items["validation"],
                                 learned_predictor, budget_check)
    selected = min(validation, key=lambda row: float(np.mean(row[selection_metric])))
    selection = dict(method=selected["method"], metric=selection_metric,
                     value=float(np.mean(selected[selection_metric])), split="validation",
                     test_labels_read=False, test_used_for_selection=False)
    on_selection(selection)
    test = _evaluate_split(corpus, plan, "test", items["test"], learned_predictor, budget_check)
    sampling = {split: dict(windows=len(rows), sources=len({row[0] for row in rows}),
                    sha256=hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
                for split, rows in items.items()}
    return dict(schema="q1_causal_forecast_evaluation_v2", selection=selection,
                selected_method=selected["method"], test_used_for_selection=False,
                validation=validation, test=test, max_windows_per_split=max_windows,
                sampling=dict(scheme="source_balanced_uniform_cutoffs_v1", splits=sampling),
                baseline_parameters=dict(history_frames=25, slope_frames=5, tau_s=.12,
                    max_linear_speed_m_s=3., max_angular_speed_rad_s=8.),
                reference_preprocessing=plan.get("preprocessing"),
                policy_inference=0, physics_steps=0, optimizer_updates=0,
                measured_robot_latency_ns=None, robot_naturalness_conclusion=None)


def execute(args):
    project_root = args.project_root.resolve()
    output = args.output.resolve()
    if (output.name != "result.json" or output.parent.exists()
            or not output.parent.is_relative_to(project_root / "data/supermimic_for_q1")
            or output.parent.is_relative_to(FORMAL_TRAINING)):
        raise ValueError("Use result.json in a new independent Q1 output directory outside formal V04")
    if args.max_windows is not None and not 1 <= args.max_windows <= 8192:
        raise ValueError("Bounded evaluation accepts 1..8192 windows per split")
    args.output.parent.mkdir(parents=True)
    args._owns_output = True
    budget = ForecastBudget(output.parent, max_seconds=args.max_seconds,
                            expected_gpu=args.expected_gpu, expected_boot=args.expected_boot,
                            require_idle=args.device == "cuda")
    plan_id = identity(args.plan)
    status = dict(status="PREFLIGHT", plan_identity=plan_id, device=args.device,
                  max_seconds=args.max_seconds, max_windows=args.max_windows,
                  training_signals_sent=0, memory_reclaim_attempted=False)
    write_json(output.parent / "status.json", status)
    lock = gpu_lock(project_root) if args.device == "cuda" else nullcontext()
    with lock:
        budget.check(start=True)
        plan = load_plan(args.admission, args.sources, args.plan)
        corpus = ForecastCorpus(plan, args.human_root)
        predictor, checkpoint_id = (None, None)
        if args.checkpoint is not None:
            predictor, checkpoint_id = _load_gru(args.checkpoint, plan, plan_id, args.device)
        budget.check()
        status["status"] = "RUNNING"
        write_json(output.parent / "status.json", status)
        result = evaluate(plan, corpus, learned_predictor=predictor, max_windows=args.max_windows,
                          selection_metric=args.selection_metric, budget_check=budget.check,
                          on_selection=lambda value: write_json(output.parent / "validation_selection.json", value))
        result.update(plan_identity=plan_id, checkpoint_identity=checkpoint_id,
                      status="COMPLETED", final_resources=budget.check(force=True))
        if identity(args.plan) != plan_id or (args.checkpoint is not None
                                             and identity(args.checkpoint) != checkpoint_id):
            raise ValueError("Evaluation input changed during the experiment")
        write_json(output, result)
        write_json(output.parent / "status.json", {**status, "status": "COMPLETED"})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("plan", "admission", "sources", "human-root", "output", "project-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-windows", type=int, default=2048)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--expected-gpu", default=DEFAULT_GPU)
    parser.add_argument("--expected-boot", default=DEFAULT_BOOT)
    parser.add_argument("--selection-metric", choices=METRICS, default=METRICS[0])
    args = parser.parse_args()
    if args.device == "cuda" and args.checkpoint is None:
        parser.error("CUDA is for GRU inference; NumPy-only baselines do not need GPU")
    try:
        result = execute(args)
    except BaseException as error:
        if getattr(args, "_owns_output", False):
            write_json(args.output.parent / "status.json", dict(status="FAILED",
                       error=f"{type(error).__name__}: {error}", training_signals_sent=0))
        raise
    print(json.dumps(dict(output=str(args.output), identity=identity(args.output),
                          selected_method=result["selected_method"]), allow_nan=False))


if __name__ == "__main__":
    main()
