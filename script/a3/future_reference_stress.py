"""Frozen E12 predictor versus equally noisy causal baselines."""

import argparse
import json
from pathlib import Path

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import DATA, ROOT
from script.a3.future_reference_probe import windows, predict, metrics
from script.a3.reference_contract import load_sim


def main(experiment="E13"):
    output = ROOT / "data/experiments" / f"a3_future_reference_20261010_{experiment}"
    output.mkdir(parents=True, exist_ok=False)
    source_experiment = "E14" if experiment == "E15" else "E12"
    source = ROOT / "data/experiments" / f"a3_future_reference_20261010_{source_experiment}"
    record = json.loads((source / "result.json").read_text())
    verification = json.loads((source / "independent_verification.json").read_text())
    model_path = source / "ridge_joint_future.npz"
    model_sha = sha256(model_path)
    if verification["status"] != "pass" or model_sha != record["model_sha256"] or record["use_absolute_pose"]:
        raise ValueError("Need the independently verified relative-only predictor")
    with np.load(model_path, allow_pickle=False) as stored:
        model = {key: stored[key].copy() for key in ("mean", "std", "weights", "bias")}
    split_path = DATA / "split.json"
    if sha256(split_path) != record["split_sha256"]:
        raise ValueError("Motion split changed")
    split = json.loads(split_path.read_text())
    sim = load_sim()
    sources = []
    for row in split["heldout"]:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Motion changed")
        _, _, q, _ = sim.load_a3_flat_csv(path, source_fps=30, frame_stride=4)
        sources.append((row, q))
    cases = []
    for noise in (0., .005, .01):
        for seed in (0, 1, 2):
            generator = np.random.default_rng(seed)
            groups = {key: [] for key in ("target", "hold", "constant_velocity", "ridge")}
            motions = []
            for row, q in sources:
                _, offsets, _, index = windows(q)
                target = q[index, None] + offsets
                observed = q + generator.normal(size=q.shape) * noise
                x, _, cv, noisy_index = windows(observed)
                if not np.array_equal(index, noisy_index):
                    raise ValueError("Noisy input changed source timestamps")
                x = x[:, :-q.shape[1]]
                origin = observed[index, None]
                predictions = dict(hold=np.broadcast_to(origin, target.shape),
                                   constant_velocity=origin+cv,
                                   ridge=origin+predict(model, x, q.shape[1]))
                values = {key: metrics(target, prediction) for key, prediction in predictions.items()}
                if noise == 0 and not np.isclose(values["ridge"]["rmse_rad"], next(
                        r["metrics"]["ridge"]["rmse_rad"] for r in record["per_motion"] if r["name"] == row["name"]),
                        atol=1e-12, rtol=1e-10):
                    raise ValueError("Zero-noise replay does not match the frozen result")
                motions.append(dict(name=row["name"], source_sha256=row["sha256"], metrics=values))
                groups["target"].append(target)
                for key, prediction in predictions.items():
                    if not np.isfinite(prediction).all():
                        raise ValueError("Nonfinite noise response")
                    groups[key].append(prediction)
            target = np.concatenate(groups["target"])
            aggregate = {key: metrics(target, np.concatenate(groups[key])) for key in predictions}
            cases.append(dict(noise_std_rad=noise, seed=seed, input_windows=len(target), aggregate=aggregate,
                per_motion=motions, all_motions_improve_vs_cv=all(
                    r["metrics"]["ridge"]["rmse_rad"] < r["metrics"]["constant_velocity"]["rmse_rad"] for r in motions)))
    if sha256(model_path) != model_sha:
        raise ValueError("Stress evaluation changed the predictor")
    result = dict(status="complete", experiment=experiment, source_experiment=source_experiment, frozen_model_sha256=model_sha,
        split_sha256=sha256(split_path), code_sha256=sha256(Path(__file__)), noise_std_rad=[0., .005, .01],
        seeds=[0, 1, 2], cases=cases, frozen_model_unchanged=True, clean_labels=True,
        training_or_policy_updates=0, zero_noise_matches_previous=True,
        robust_joint_prediction_gate=all(case["all_motions_improve_vs_cv"] for case in cases),
        scope="ARTIFICIAL_JOINT_NOISE_NOT_MEASURED_MOCAP_NOISE_OR_POLICY_ACCEPTANCE", backup_status="LOCAL_ONLY")
    atomic_json(output / "result.json", result)
    atomic_json(ROOT / "data/manifests" / f"a3_future_reference_20261010_{experiment}.json", result)
    print(json.dumps(dict(status=result["status"], robust_gate=result["robust_joint_prediction_gate"],
        cases=[dict(noise=c["noise_std_rad"], seed=c["seed"], rmse={k:v["rmse_rad"] for k,v in c["aggregate"].items()},
                    all_motions_improve=c["all_motions_improve_vs_cv"]) for c in cases]), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E13", "E15"), default="E13")
    main(parser.parse_args().experiment)
