#!/usr/bin/env python3
"""Plot complete PHUMA paired evaluations without loading a policy or simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked(record):
    if identity(record["path"]) != record:
        raise ValueError("Recorded input changed: " + record["path"])
    return Path(record["path"])


def trajectory_errors(state, reference):
    state, reference = np.asarray(state), np.asarray(reference)
    if (state.ndim != 2 or state.shape[1] != 57 or reference.ndim != 2
            or reference.shape[1] != 29 or not 0 < len(state) <= len(reference)
            or not np.isfinite(state).all() or not np.isfinite(reference).all()):
        raise ValueError("Expected finite Q1 states and a complete reference denominator")
    difference = state[:, :29] - reference[:len(state)]
    root_error = np.linalg.norm(difference[:, :3], axis=1)
    joint_error = np.linalg.norm(difference[:, 7:29], axis=1) / np.sqrt(22.)
    return root_error, joint_error


def style(axis):
    axis.grid(axis="y", color="#e1e5e8", linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    experiment = args.experiment.resolve()
    controller = json.loads((experiment / "controller_status.json").read_text())
    comparison_path = experiment / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    spec_path = checked(comparison["spec"])
    spec = json.loads(spec_path.read_text())
    if (controller["status"] != "PHUMA_TRAINING_AND_MATCHED_EVALUATION_COMPLETED"
            or comparison["status"] != "PHUMA_MATCHED_PHYSICS_COMPARISON_COMPLETED"
            or spec_path != experiment / "spec.json"):
        raise ValueError("Require the original completed PHUMA paired-evaluation pipeline")
    pairs = comparison["pairs"]
    names = [row["clip"] for row in pairs]
    expected = [name for split in ("train", "development", "test") for name in spec["split_clips"][split]]
    if len(set(names)) != len(names) or sorted(names) != sorted(expected):
        raise ValueError("Plot coverage differs from the registered three splits")
    source = json.loads((Path(spec["motions"]) / "manifest.json").read_text())
    source_rows = {row["clip"]: row for row in source["motions"]}
    traces, inputs, reconciliation = {}, [], []
    for record in comparison["evidence"]:
        checked(record)
        inputs.append(record)
    for label in ("initial", "final"):
        receipt_path = experiment / ("evaluation_" + label) / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        rows = {row["clip"]: row for row in receipt["rows"]}
        if len(rows) != len(names) or rows.keys() != set(names):
            raise ValueError("Evaluation coverage is incomplete")
        traces[label] = {}
        for pair in pairs:
            name, metrics, row = pair["clip"], pair[label], rows[pair["clip"]]
            if (pair["split"] not in spec["split_clips"]
                    or name not in spec["split_clips"][pair["split"]]
                    or pair["bucket"] != source_rows[name]["bucket"]):
                raise ValueError("Plot split or motion category differs from its source")
            reference_path, trace_path = checked(row["reference"]), checked(row["trace"])
            with np.load(reference_path, allow_pickle=False) as reference, np.load(trace_path, allow_pickle=False) as trace:
                target, state = reference["qpos"].astype(float), trace["state"].astype(float)
                if float(reference["fps"]) != 50.:
                    raise ValueError("Expected the registered native-time 50 Hz reference")
                root_error, joint_error = trajectory_errors(state, target)
                root_rmse, joint_rmse = (float(np.linalg.norm(values) / np.sqrt(len(values)))
                                         for values in (root_error, joint_error))
                if (len(state) != metrics["executed_steps"] or len(target) != metrics["expected_steps"]
                        or row["failed"] != metrics["failed"]
                        or abs(root_rmse - metrics["root_position_rmse_m"]) > 2e-5
                        or abs(joint_rmse - metrics["executed_joint_rmse_rad"]) > 2e-5):
                    raise ValueError("Recorded paired metrics differ from raw saved states")
                reconciliation.append({"clip": name, "label": label, "executed_steps": len(state),
                                       "expected_steps": len(target), "root_rmse_m": root_rmse,
                                       "joint_rmse_rad": joint_rmse})
                if pair["bucket"] == "locomotion":
                    origin = target[0, :2]
                    traces[label][name] = {"reference_xy": target[:, :2] - origin,
                                          "policy_xy": state[:, :2] - origin,
                                          "root_error_m": root_error}
            inputs.extend([row["reference"], row["trace"]])
    args.output.mkdir(parents=True, exist_ok=False)
    colors = {"initial": "#537e96", "final": "#b33850"}
    tags = {"train": "T", "development": "D", "test": "E"}
    labels = [tags[row["split"]] + ":" + row["bucket"][:3] + ":" + row["clip"].split("_")[-1][:6] for row in pairs]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    panels = [("executed_joint_rmse_rad", "All joints (rad)", .15),
              ("root_position_rmse_m", "Root position (m)", .10),
              ("normalized_return", "Full-denominator return", None),
              ("legs", "Leg joints (rad)", None), ("waist", "Waist joints (rad)", None),
              ("arms", "Arm joints (rad)", None)]
    x = np.arange(len(pairs))
    for axis, (key, title, limit) in zip(axes.flat, panels):
        for offset, label in ((-.19, "initial"), (.19, "final")):
            values = [row[label]["joint_group_rmse_rad"][key] if key in ("legs", "waist", "arms") else row[label][key] for row in pairs]
            axis.bar(x + offset, values, .36, color=colors[label], label=label)
        if limit is not None:
            axis.axhline(limit, color="#28845a", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(x, labels, rotation=90)
        style(axis)
    axes[0, 0].legend(frameon=False, fontsize=9)
    failures = {label: sum(row[label]["failed"] for row in pairs) for label in colors}
    figure.suptitle("PHUMA / Q1 / initial vs final\n"
                     f"Early failures: initial {failures['initial']}, final {failures['final']}; error metrics use the executed prefix. "
                     "T=train, D=development, E=test.", fontsize=11)
    artifacts = []
    path = args.output / "paired_errors.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    artifacts.append(identity(path))
    movement = [row for row in pairs if row["bucket"] == "locomotion"]
    if movement:
        figure, axes = plt.subplots(len(movement), 2, figsize=(12, 3.7 * len(movement)), squeeze=False, layout="constrained")
        for row_axes, pair in zip(axes, movement):
            name = pair["clip"]
            reference = traces["initial"][name]["reference_xy"]
            if not np.array_equal(reference, traces["final"][name]["reference_xy"]):
                raise ValueError("Initial/final movement reference differs")
            row_axes[0].plot(reference[:, 0], reference[:, 1], color="#28845a", linewidth=2, label="reference")
            for label in colors:
                saved = traces[label][name]
                xy, error = saved["policy_xy"], saved["root_error_m"]
                row_axes[0].plot(xy[:, 0], xy[:, 1], color=colors[label], label=label)
                row_axes[0].scatter(*xy[-1], color=colors[label], s=16)
                row_axes[1].plot(np.arange(len(error)) / 50., error, color=colors[label], label=label)
            row_axes[0].set_aspect("equal", adjustable="datalim")
            row_axes[0].set(xlabel="Root X from reference start (m)", ylabel="Root Y (m)", title=name)
            row_axes[1].axhline(.10, color="#28845a", linestyle="--", linewidth=1)
            row_axes[1].set(xlabel="Reference sample time (s)", ylabel="Root position error (m)", title=pair["split"] + " / complete reference span")
            row_axes[1].set_xlim(0, (len(reference) - 1) / 50.)
            for axis in row_axes:
                style(axis)
                axis.legend(frameon=False, fontsize=8)
        counts = {split: sum(row["split"] == split for row in movement) for split in ("train", "development", "test")}
        figure.suptitle("All qualified locomotion clips / saved physical trajectories\n"
                       f"Endpoints mark executed traces; train {counts['train']}, development {counts['development']}, test {counts['test']}.", fontsize=11)
        path = args.output / "locomotion_paths.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        artifacts.append(identity(path))
    receipt = {"status": "PHUMA_MATCHED_EVALUATION_PLOTS_COMPLETED", "comparison": identity(comparison_path),
               "spec": comparison["spec"], "clips": names, "locomotion_clips": [row["clip"] for row in movement],
               "metric_reconciliation": reconciliation, "artifacts": artifacts, "inputs": inputs,
               "source": identity(__file__), "new_physics_steps": 0, "new_policy_calls": 0,
               "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
               "limits": "Saved-trace plots, not new simulation or a hardware result. Early-failure errors cover executed prefixes."}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "clips": len(names), "artifacts": artifacts}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
