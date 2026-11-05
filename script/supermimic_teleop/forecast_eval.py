"""Causal forecast evaluation with physical, per-horizon error reports.

The evaluator is independent of the formal V04 trainer.  Every prediction is
constructed from one visible history window only.  Validation and test results
are emitted together, but model selection is performed on validation only and
the test split is never used to choose a method.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from forecast_data import identity, load_plan
from forecast_geometry import (DT_NS, decode_targets, encode_history,
    encode_targets, inverse, log_rotation, product)
from forecast_model import ForecastNormalization, ForecastSpec, make_model


DT_S = DT_NS / 1e9
METHODS = ("hold", "cv", "damped_cv")


def _anchored_slope(values):
    """Estimate a velocity with the latest visible value as fixed intercept."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or len(values) < 2:
        raise ValueError("Require at least two visible samples")
    times = (np.arange(len(values), dtype=np.float64) - (len(values) - 1)) * DT_S
    denominator = float(np.sum(times * times))
    return np.sum(times.reshape((-1,) + (1,) * (values.ndim - 1)) *
                  (values - values[-1]), axis=0) / denominator


def _cap(values, maximum):
    if maximum is None:
        return values
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.minimum(1., maximum / np.maximum(norms, 1e-12))
    return values * scale


def baseline_targets(history_positions, history_quaternions, method="damped_cv",
                     *, horizon=10, tau_s=.12, max_linear_speed_m_s=3.,
                     max_angular_speed_rad_s=8.):
    """Return physical target labels for a causal hold/CV/damped-CV baseline.

    The returned labels use the same anchored SE(3) representation as the GRU.
    The last five visible frames are the only samples used to estimate velocity.
    """
    if method not in METHODS or type(horizon) is not int or not 1 <= horizon <= 10:
        raise ValueError("Require a known method and horizon in 1..10")
    if method == "damped_cv" and (not np.isfinite(tau_s) or tau_s <= 0):
        raise ValueError("Require a finite positive damping time")
    positions = np.asarray(history_positions, dtype=np.float64)
    quaternions = np.asarray(history_quaternions, dtype=np.float64)
    if len(positions) < 2:
        raise ValueError("Require at least two visible history frames")
    history = encode_history(positions, quaternions)
    last_positions, last_quaternions = positions[-1], quaternions[-1]
    if method == "hold":
        predicted_positions = np.repeat(last_positions[None], horizon, axis=0)
        predicted_quaternions = np.repeat(last_quaternions[None], horizon, axis=0)
    else:
        visible_positions = positions[-min(5, len(positions)):]
        linear = _cap(_anchored_slope(visible_positions), max_linear_speed_m_s)
        visible_quaternions = quaternions[-min(5, len(quaternions)):]
        relative = product(inverse(last_quaternions), visible_quaternions)
        angular = _cap(_anchored_slope(log_rotation(relative)), max_angular_speed_rad_s)
        offsets = np.arange(1, horizon + 1, dtype=np.float64) * DT_S
        if method == "damped_cv":
            travel = tau_s * (-np.expm1(-offsets / tau_s))
        else:
            travel = offsets
        predicted_positions = last_positions[None] + travel[:, None, None] * linear[None]
        deltas = travel[:, None, None] * angular[None]
        from forecast_geometry import exp_rotation
        predicted_quaternions = product(last_quaternions[None], exp_rotation(deltas))
    return encode_targets(predicted_positions, predicted_quaternions, history.anchor)


def _new_accumulator(horizon):
    return dict(samples=0, position_sq=np.zeros(horizon), position_count=np.zeros(horizon),
                root_sq=np.zeros(horizon), root_count=np.zeros(horizon),
                orientation_sq=np.zeros(horizon), orientation_count=np.zeros(horizon),
                sources=set())


def _add(accumulator, prediction, target_positions, target_quaternions, source):
    position = prediction.positions - target_positions
    angles = np.linalg.norm(log_rotation(product(inverse(target_quaternions),
                                                prediction.quaternions)), axis=-1)
    accumulator["samples"] += 1
    accumulator["position_sq"] += np.sum(position * position, axis=(1, 2))
    accumulator["position_count"] += position.shape[1] * position.shape[2]
    root = position[:, 0]
    accumulator["root_sq"] += np.sum(root * root, axis=1)
    accumulator["root_count"] += root.shape[1]
    accumulator["orientation_sq"] += np.sum(angles * angles, axis=1)
    accumulator["orientation_count"] += angles.shape[1]
    accumulator["sources"].add(source)


def _finish(accumulator, method, split):
    def rms(values, counts):
        return np.sqrt(np.divide(values, counts, out=np.zeros_like(values), where=counts > 0))
    return dict(method=method, split=split, samples=accumulator["samples"],
                sources=len(accumulator["sources"]),
                position_rmse_m=rms(accumulator["position_sq"], accumulator["position_count"]).tolist(),
                root_position_rmse_m=rms(accumulator["root_sq"], accumulator["root_count"]).tolist(),
                orientation_geodesic_rmse_rad=rms(accumulator["orientation_sq"],
                                                   accumulator["orientation_count"]).tolist(),
                target_offsets_ns=[(i + 1) * DT_NS for i in range(len(accumulator["position_sq"]))],
                temporal_realignment_applied=False, bone_contact_constraints_applied=False,
                measured_robot_latency_ns=None)


def _items(plan, split, limit=None):
    count = 0
    for row in plan["clips"]:
        if row["split"] != split:
            continue
        for cutoff in range(plan["history_frames"] - 1,
                            row["frames"] - plan["future_frames"]):
            yield row["relative"], cutoff
            count += 1
            if limit is not None and count >= limit:
                return


def _load_gru(checkpoint, plan, device):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("A GRU checkpoint requires the admitted Torch runtime") from error
    payload = torch.load(checkpoint, map_location=device)
    spec_record = payload.get("spec", {})
    spec = ForecastSpec(hidden_size=spec_record.get("hidden_size", 128))
    allowed = {row["relative"] for row in plan["clips"] if row["split"] == "train"}
    normalization = ForecastNormalization.from_dict(payload["normalization"], allowed)
    model = make_model(spec).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return torch, model, normalization


def _evaluate_method(corpus, plan, split, method, items, *, torch=None, model=None,
                     normalization=None, device=None):
    accumulator = _new_accumulator(plan["future_frames"])
    for relative, cutoff in items:
        window = corpus.window(relative, cutoff, split=split)
        history = encode_history(window.input_positions, window.input_quaternions)
        if method in METHODS:
            labels = baseline_targets(window.input_positions, window.input_quaternions, method,
                                      horizon=plan["future_frames"])
        else:
            if torch is None or model is None or normalization is None:
                raise ValueError("GRU evaluation requires a loaded model and normalization")
            inputs = torch.from_numpy(normalization.inputs(history.features)[None]).to(device)
            with torch.inference_mode():
                normalized = model(inputs).detach().cpu().numpy()[0]
            labels = normalization.physical(normalized)
        prediction = decode_targets(labels, history.anchor)
        _add(accumulator, prediction, window.target_positions, window.target_quaternions,
             relative)
    return _finish(accumulator, method, split)


def evaluate(plan, corpus, *, checkpoint=None, device="cpu", max_windows=None,
             selection_metric="position_rmse_m"):
    methods = list(METHODS)
    torch = model = normalization = device_obj = None
    if checkpoint is not None:
        if device == "cuda":
            import torch as torch_module
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA evaluation requested but unavailable")
            device_obj = torch_module.device("cuda")
        else:
            import torch as torch_module
            device_obj = torch_module.device("cpu")
        torch, model, normalization = _load_gru(checkpoint, plan, device_obj)
        methods.append("gru")
    reports = {split: [] for split in ("validation", "test")}
    for split in reports:
        for method in methods:
            reports[split].append(_evaluate_method(
                corpus, plan, split, method, _items(plan, split, max_windows),
                torch=torch, model=model, normalization=normalization, device=device_obj))
    if selection_metric not in ("position_rmse_m", "root_position_rmse_m",
                                "orientation_geodesic_rmse_rad"):
        raise ValueError("Unknown validation selection metric")
    selected = min(reports["validation"],
                   key=lambda row: float(np.mean(row[selection_metric])))
    return dict(schema="q1_causal_forecast_evaluation_v1", methods=methods,
                selection_metric=selection_metric, selected_method=selected["method"],
                test_used_for_selection=False, validation=reports["validation"],
                test=reports["test"], max_windows_per_split=max_windows,
                checkpoint=None if checkpoint is None else identity(checkpoint),
                plan_identity=identity(plan["_path"]) if "_path" in plan else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("plan", "admission", "sources", "human-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--selection-metric", choices=("position_rmse_m", "root_position_rmse_m",
                                                         "orientation_geodesic_rmse_rad"),
                        default="position_rmse_m")
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence")
    plan = load_plan(args.admission, args.sources, args.plan)
    plan["_path"] = args.plan
    corpus = __import__("forecast_data", fromlist=["ForecastCorpus"]).ForecastCorpus(
        plan, args.human_root)
    result = evaluate(plan, corpus, checkpoint=args.checkpoint, device=args.device,
                      max_windows=args.max_windows, selection_metric=args.selection_metric)
    result.pop("plan_identity", None)
    result["plan_identity"] = identity(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), identity=identity(args.output),
                          selected_method=result["selected_method"]), allow_nan=False))


if __name__ == "__main__":
    main()
