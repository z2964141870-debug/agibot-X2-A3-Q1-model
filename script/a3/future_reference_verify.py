"""Rebuild labels with interpolation and independently verify fitted artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import DATA, ROOT
from script.a3.reference_contract import load_sim
from script.a3.forecast_noise import noise_covariances


def main(experiment):
    output = ROOT / "data/experiments" / f"a3_future_reference_20261010_{experiment}"
    record = json.loads((output / "result.json").read_text())
    path = output / "ridge_joint_future.npz"
    if sha256(path) != record["model_sha256"] or path.stat().st_size != record["model_bytes"]:
        raise ValueError("Predictor checksum changed")
    with np.load(path, allow_pickle=False) as saved:
        model = {key: saved[key].copy() for key in ("mean", "std", "weights", "bias")}
    if not all(np.isfinite(v).all() for v in model.values()) or not (model["std"] > 0).all():
        raise ValueError("Nonfinite predictor")
    split_path = DATA / "split.json"
    if sha256(split_path) != record["split_sha256"]:
        raise ValueError("Dataset split changed")
    split = json.loads(split_path.read_text())
    sim = load_sim()
    training = []
    training_targets = []
    comparisons = []
    recorded = {r["name"]: r for r in record["per_motion"]}
    for part in ("train", "heldout"):
        for row in split[part]:
            source = Path(row["path"])
            if sha256(source) != row["sha256"]:
                raise ValueError("Source changed")
            _, _, q, _ = sim.load_a3_flat_csv(source, source_fps=30, frame_stride=4)
            indices = np.arange(5, len(q)-7)
            history = np.stack([q[indices-5+i] for i in range(6)], axis=1)
            features = history - q[indices, None]
            if record.get("use_absolute_pose", True):
                features[:, -1] = q[indices]
                x = features.reshape(len(indices), -1)
            else:
                x = features[:, :-1].reshape(len(indices), -1)
            if part == "train":
                training.append(x)
            times = np.arange(len(q)) / 30
            target = np.stack([np.stack([np.interp(indices/30+horizon, times, q[:, channel])
                    for channel in range(q.shape[1])], axis=-1) for horizon in record["horizons_s"]], axis=1)
            target -= q[indices, None]
            if part == "train":
                training_targets.append(target.reshape(len(indices), -1))
            prediction = (((x-model["mean"])/model["std"])@model["weights"]+model["bias"]).reshape(target.shape)
            rmse = float(np.sqrt(np.mean((prediction-target)**2)))
            expected = recorded[row["name"]]["metrics"]["ridge"]["rmse_rad"]
            if not np.isclose(rmse, expected, atol=1e-12, rtol=1e-10):
                raise ValueError("Independent interpolation did not reproduce prediction error")
            comparisons.append(dict(name=row["name"], rmse_rad=rmse, matched=True))
    x = np.concatenate(training)
    expected_std = x.std(0)
    expected_std[expected_std < 1e-8] = 1
    if not np.allclose(x.mean(0), model["mean"], atol=1e-12, rtol=1e-10) or not np.allclose(
            expected_std, model["std"], atol=1e-12, rtol=1e-10):
        raise ValueError("Normalization was not derived solely from registered training motions")
    z = (x-model["mean"])/model["std"]
    y = np.concatenate(training_targets)
    matrix = z.T@z + np.eye(z.shape[1])*record["ridge_alpha"]
    right = z.T@(y-model["bias"])
    sigma = record.get("assumed_training_noise_std_rad", 0)
    if sigma:
        covariance, cross = noise_covariances(model["std"], 29, len(record["horizons_s"]), sigma)
        matrix += len(z)*covariance
        right += len(z)*cross
    residual = np.linalg.norm(matrix@model["weights"]-right) / max(np.linalg.norm(right), 1e-12)
    if residual > 1e-9 or not np.allclose(y.mean(0), model["bias"], atol=1e-12):
        raise ValueError("Predictor does not solve its registered training-only objective")
    receipt = dict(status="pass", experiment=experiment, model_sha256=sha256(path), cpu_reload_finite=True,
                   training_only_normalization_verified=True, independent_interpolation_cases=len(comparisons),
                   comparisons=comparisons, code_sha256=sha256(Path(__file__)),
                   normal_equation_relative_residual=float(residual), registered_objective_verified=True,
                   policy_improvement_verified=False, backup_status="LOCAL_ONLY")
    atomic_json(output / "independent_verification.json", receipt)
    print(json.dumps(dict(status="PASS", experiment=experiment, independently_recomputed_motions=len(comparisons))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E11", "E12", "E14"), required=True)
    main(parser.parse_args().experiment)
