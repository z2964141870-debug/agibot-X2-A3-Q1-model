"""CPU causal joint-reference prediction with a fixed whole-motion split."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import DATA, ROOT
from script.a3.reference_contract import load_sim


FPS = 30
HISTORY = 6
HORIZONS = np.arange(1, 11) / 50
RIDGE_ALPHA = 1.0


def windows(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(q).all() or len(q) < 20:
        raise ValueError("Need a finite joint sequence with enough context")
    index = np.arange(HISTORY-1, len(q)-7)
    history_index = index[:, None] + np.arange(-HISTORY+1, 1)
    if np.any(history_index > index[:, None]):
        raise ValueError("Input requested an unavailable source frame")
    history = q[history_index]
    current = q[index]
    feature = history - current[:, None]
    feature[:, -1] = current
    future_offset = HORIZONS * FPS
    lower = index[:, None] + np.floor(future_offset).astype(int)
    fraction = future_offset - np.floor(future_offset)
    target = q[lower] * (1-fraction[None, :, None]) + q[lower+1] * fraction[None, :, None]
    target -= current[:, None]
    velocity = (current-q[index-1]) * FPS
    cv = velocity[:, None] * HORIZONS[None, :, None]
    return feature.reshape(len(index), -1), target, cv, index


def fit(x, target):
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std < 1e-8, 1, std)
    z = (x-mean) / std
    y = target.reshape(len(target), -1)
    bias = y.mean(0)
    weights = np.linalg.solve(z.T @ z + np.eye(z.shape[1]) * RIDGE_ALPHA, z.T @ (y-bias))
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite fitted predictor")
    return dict(mean=mean, std=std, weights=weights, bias=bias)


def predict(model, x, dimensions):
    return (((x-model["mean"])/model["std"]) @ model["weights"] + model["bias"]).reshape(
        len(x), len(HORIZONS), dimensions)


def metrics(target, prediction):
    error = prediction - target
    return dict(rmse_rad=float(np.sqrt(np.mean(error**2))),
                horizon_rmse_rad=np.sqrt(np.mean(error**2, axis=(0, 2))).tolist(),
                absolute_error_p95_rad=float(np.quantile(np.abs(error), .95)))


def main(experiment="E11"):
    use_absolute_pose = experiment == "E11"
    output = ROOT / "data/experiments" / f"a3_future_reference_20261010_{experiment}"
    output.mkdir(parents=True, exist_ok=False)
    split_path = DATA / "split.json"
    split = json.loads(split_path.read_text())
    sim = load_sim()
    blocks = {}
    for part in ("train", "heldout"):
        blocks[part] = []
        for row in split[part]:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError("Registered source changed")
            _, _, q, _ = sim.load_a3_flat_csv(path, source_fps=FPS, frame_stride=4)
            x, y, cv, index = windows(q)
            if not use_absolute_pose:
                x = x[:, :-q.shape[-1]]
            blocks[part].append(dict(name=row["name"], sha256=row["sha256"], x=x, y=y, cv=cv, index=index))
    train_x = np.concatenate([b["x"] for b in blocks["train"]])
    train_y = np.concatenate([b["y"] for b in blocks["train"]])
    dimensions = train_y.shape[-1]
    model = fit(train_x, train_y)
    model_path = output / "ridge_joint_future.npz"
    np.savez_compressed(model_path, **model, horizons_s=HORIZONS, history_frames=HISTORY,
                        source_fps=FPS, ridge_alpha=RIDGE_ALPHA,
                        usage=np.asarray("DIAGNOSTIC_OFFICIAL_SAMPLES_NOT_DEPLOYMENT"))
    with np.load(model_path, allow_pickle=False) as stored:
        reloaded = {key: stored[key].copy() for key in model}
    if any(not np.array_equal(model[key], reloaded[key]) for key in model):
        raise ValueError("Saved model did not reload exactly")
    aggregate, per_motion = {}, []
    for part, items in blocks.items():
        targets, constant, learned = [], [], []
        for item in items:
            yhat = predict(reloaded, item["x"], dimensions)
            if not np.isfinite(yhat).all():
                raise ValueError("Nonfinite prediction")
            baselines = dict(hold=metrics(item["y"], np.zeros_like(item["y"])),
                             constant_velocity=metrics(item["y"], item["cv"]), ridge=metrics(item["y"], yhat))
            per_motion.append(dict(name=item["name"], split=part, source_sha256=item["sha256"],
                                   input_frames=len(item["index"]), metrics=baselines))
            targets.append(item["y"])
            constant.append(item["cv"])
            learned.append(yhat)
        target = np.concatenate(targets)
        aggregate[part] = dict(samples=len(target), hold=metrics(target, np.zeros_like(target)),
                               constant_velocity=metrics(target, np.concatenate(constant)),
                               ridge=metrics(target, np.concatenate(learned)))
    timings = []
    for x in blocks["heldout"][0]["x"][:100]:
        started = time.perf_counter_ns()
        predict(reloaded, x[None], dimensions)
        timings.append((time.perf_counter_ns()-started)/1e6)
    validation = aggregate["heldout"]
    selected = (validation["ridge"]["rmse_rad"] < min(validation["hold"]["rmse_rad"],
                validation["constant_velocity"]["rmse_rad"]) and all(
                r["metrics"]["ridge"]["rmse_rad"] < r["metrics"]["constant_velocity"]["rmse_rad"]
                for r in per_motion if r["split"] == "heldout"))
    receipt = dict(status="complete", experiment=experiment, source_fps=FPS, raw_fps=120, stride=4,
        use_absolute_pose=use_absolute_pose, feature_dimension=train_x.shape[1],
        history_frames=HISTORY, horizons_s=HORIZONS.tolist(), ridge_alpha=RIDGE_ALPHA,
        fit_motion_count=len(blocks["train"]), validation_motion_count=len(blocks["heldout"]),
        split_sha256=sha256(split_path), code_sha256=sha256(Path(__file__)), model_sha256=sha256(model_path),
        model_bytes=model_path.stat().st_size, independent_reload_exact=True, model_finite=True,
        input_never_reads_future=True, device="CPU", optimizer_or_policy_updates=0,
        results=aggregate, per_motion=per_motion, selected_joint_predictor=selected,
        timing_ms=dict(single_sample_median=float(np.median(timings)), p95=float(np.quantile(timings, .95)), samples=len(timings)),
        scope="JOINT_FUTURE_PREDICTION_ONLY_NOT_ROOT_OR_ORIENTATION_OR_POLICY_OR_MOCAP_ACCEPTANCE",
        split_note="FOUR_REPEATED_DIAGNOSTIC_VALIDATION_MOTIONS_NOT_FRESH_FINAL_TEST",
        backup_status="LOCAL_ONLY")
    atomic_json(output / "result.json", receipt)
    manifest = ("a3_future_reference_20261010.json" if experiment == "E11" else
                f"a3_future_reference_20261010_{experiment}.json")
    atomic_json(ROOT / "data/manifests" / manifest, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E11", "E12"), default="E11")
    main(parser.parse_args().experiment)
