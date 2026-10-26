#!/usr/bin/env python3
"""Compare completed Q1 trials using their preserved physical evaluation traces."""

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]
for directory in (SCRIPT / "supermimic_q1_analysis",
                  SCRIPT / "supermimic_q1_reproduction/leg_control"):
    sys.path.insert(0, str(directory))


def movement(state, initial, target, dt):
    actual = np.asarray(state, dtype=float)[:, :2]
    origin = np.asarray(initial, dtype=float)[:2]
    direction = target[-1, :2] - target[0, :2]
    distance = float(np.linalg.norm(direction))
    vector = actual[-1] - origin
    path = np.concatenate([origin[None], actual])
    seconds = len(actual) * dt
    projection = float(vector @ (direction / distance)) if distance > 1e-10 else None
    return {"executed_seconds": seconds,
            "net_displacement_m": float(np.linalg.norm(vector)),
            "path_length_m": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
            "displacement_vector_xy_m": vector.tolist(),
            "net_displacement_along_full_reference_direction_m": projection,
            "mean_velocity_along_full_reference_direction_mps":
                projection / seconds if projection is not None else None,
            "mean_velocity_xy_mps": (vector / seconds).tolist(),
            "root_xy_error_mean_m": float(np.linalg.norm(
                actual - target[:len(actual), :2], axis=1).mean()),
            "full_reference_displacement_m": distance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trial", nargs=2, action="append", required=True,
                        metavar=("LABEL", "COMPLETED_TRIAL_DIRECTORY"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from control import ContactCorrectedAssets, Leg50Assets, VARIANT
    from diagnose_fullbody import checked, kinematic_metrics, write_json
    from supermimic.q1_assets import identity
    torch.set_num_threads(2)
    if len(args.trial) < 2 or len({label for label, _ in args.trial}) != len(args.trial):
        parser.error("Compare at least two distinct trial labels")
    args.root = args.root.resolve(strict=True)
    args.output = args.output.resolve()
    if (not args.output.is_relative_to(args.root / "data/supermimic_for_q1")
            or args.output.exists()):
        parser.error("Use a new comparison file in the unified Q1 data directory")
    records, trials, arrays, reference, protocol, initial = [], {}, {}, None, None, None

    def read(path):
        records.append(identity(path))
        return json.loads(Path(path).read_text())

    with torch.no_grad():
        for label, directory in args.trial:
            directory = Path(directory)
            status = read(directory / "status.json")
            guard = read(directory / "train_guard.json")
            train = read(directory / "run/receipt.json")
            evaluation = read(directory / "evaluation_final/receipt.json")
            if (status["status"] != "COMPLETED" or guard["status"] != "WORKER_COMPLETED"
                    or not guard.get("cleanup", {}).get("worker_reaped")
                    or train["error"] is not None
                    or evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"):
                raise ValueError("Compare completed pipelines with preserved physical evidence")
            row, = evaluation["rows"]
            variant = evaluation["assets"].get("control_variant", {}).get("version")
            if variant not in (None, VARIANT):
                raise ValueError("Unknown Q1 control variant")
            assets = (Leg50Assets if variant else ContactCorrectedAssets)(args.root)
            if assets.manifest() != evaluation["assets"]:
                raise ValueError("Saved control contract differs from the analysis assets")
            if reference is None:
                reference, protocol = row["reference"], evaluation["protocol"]
                with np.load(checked(reference), allow_pickle=False) as data:
                    target = data["qpos"].copy()
                    ref_body = {"pos": data["body_pos_w"].copy(),
                                "quat": data["body_quat_w"].copy()}
                records.append(reference)
            if row["reference"] != reference or evaluation["protocol"] != protocol:
                raise ValueError("Trial reference or deterministic evaluation protocol differs")
            records.append(row["trace"])
            with np.load(checked(row["trace"]), allow_pickle=False) as data:
                state, start = data["state"].copy(), data["initial_state"].copy()
            if initial is None:
                initial = start
            if (not np.array_equal(start, initial) or len(state) != row["executed_steps"]
                    or len(target) != row["expected_steps"] or not np.isfinite(state).all()):
                raise ValueError("Initial condition or trace frame accounting differs")
            body = {key: value.numpy() for key, value in
                    assets.heads().robot_bodies(torch.from_numpy(state)).items()}
            motion = movement(state, start, target, evaluation["assets"]["dt"])
            for key, value in (("root_xy_displacement_m", motion["net_displacement_m"]),
                               ("root_xy_path_length_m", motion["path_length_m"]),
                               ("root_xy_error_mean_m", motion["root_xy_error_mean_m"])):
                if abs(row[key] - value) > 2e-6:
                    raise ValueError("Independent movement reconstruction disagrees: " + key)
            leg_path = directory / "leg_analysis/receipt.json"
            legs = None
            if leg_path.exists():
                leg_receipt = read(leg_path)
                if (leg_receipt["status"] != "Q1_SAVED_LEG_SUPPORT_ANALYSIS_COMPLETED"
                        or leg_receipt["evaluation"] != identity(directory / "evaluation_final/receipt.json")):
                    raise ValueError("Saved leg analysis belongs to another evaluation")
                legs, = leg_receipt["rows"]
            arrays[label] = (state, body, assets.names, evaluation["assets"]["dt"])
            trials[label] = {"training": {key: train[key] for key in
                ("training_seconds", "step", "env_steps", "wm_updates", "policy_updates")},
                "control_variant": variant, "evaluation": row, "movement": motion,
                "executed_kinematics": kinematic_metrics(state, target, body, ref_body, assets.names),
                "legs": legs}
        pairs = []
        for first, second in itertools.combinations(trials, 2):
            count = min(len(arrays[first][0]), len(arrays[second][0]))
            paired = {}
            for label in (first, second):
                state, body, names, dt = arrays[label]
                paired[label] = {
                    "kinematics": kinematic_metrics(state[:count], target,
                        {key: value[:count] for key, value in body.items()}, ref_body, names),
                    "movement": movement(state[:count], initial, target, dt)}
            pairs.append({"labels": [first, second], "common_prefix_steps": count,
                          "evaluations": paired})
    if records != [identity(item["path"]) for item in records]:
        raise ValueError("Input evidence changed during comparison")
    report = {"status": "Q1_COMPLETED_PHYSICAL_TRIAL_COMPARISON_COMPLETED",
        "source": identity(__file__), "reference": reference, "inputs": records,
        "trials": trials, "pairs": pairs, "new_physics_integration_steps": 0,
        "new_policy_inference": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Unequal training budgets and batch counts are reported explicitly; not a matched-duration ablation. Movement may include falling. Boundary support is not a walking-success classifier. Executed gesture peaks are censored by early termination; pair metrics use the common prefix. Missing historical leg analysis remains null. One training reference, no held-out or hardware claim."}
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "trials": {
        label: {"training": value["training"], "steps": value["evaluation"]["executed_steps"],
                "failed": value["evaluation"]["failed"], "movement": value["movement"]}
        for label, value in trials.items()}}), flush=True)


if __name__ == "__main__":
    main()
