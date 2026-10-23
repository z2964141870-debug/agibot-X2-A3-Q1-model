#!/usr/bin/env python3
"""Compare audited Q1 traces over the same executed reference prefix."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_file(item):
    if identity(item["path"]) != item:
        raise ValueError("Recorded input identity differs from the saved file")
    return Path(item["path"])


def load_evaluation(receipt_path, audit_path):
    receipt = json.loads(Path(receipt_path).read_text())
    audit = json.loads(Path(audit_path).read_text())
    if (receipt["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"
            or audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
            or audit["verified"] is not True or audit["receipt"] != identity(receipt_path)):
        raise ValueError("Evaluation requires its matching successful independent audit")
    rows = {row["clip"]: row for row in receipt["rows"]}
    if not rows or len(rows) != len(receipt["rows"]):
        raise ValueError("Evaluation clip set is empty or duplicated")
    return {"receipt": identity(receipt_path), "audit": identity(audit_path), "rows": rows}


def errors(state, target):
    joint_error = state[:, 7:29] - target[:, 7:29]
    return {"joint_rmse_rad": float(np.sqrt(np.mean(joint_error ** 2))),
            "leg_joint_rmse_rad": float(np.sqrt(np.mean(joint_error[:, :12] ** 2))),
            "arm_joint_rmse_rad": float(np.sqrt(np.mean(joint_error[:, 14:22] ** 2))),
            "root_horizontal_rmse_m": float(np.sqrt(np.mean(np.sum((state[:, :2] - target[:, :2]) ** 2, axis=1))))}


def compare(evaluations):
    clip_sets = [set(evaluation["rows"]) for evaluation in evaluations.values()]
    if len(evaluations) < 2 or any(clips != clip_sets[0] for clips in clip_sets):
        raise ValueError("Compare at least two evaluations with identical clip coverage")
    output = []
    for clip in sorted(clip_sets[0]):
        rows = {label: evaluation["rows"][clip] for label, evaluation in evaluations.items()}
        first = next(iter(rows.values()))
        if any(row["reference"] != first["reference"] for row in rows.values()):
            raise ValueError("Compared reference identities differ")
        with np.load(checked_file(first["reference"]), allow_pickle=False) as saved:
            target = saved["qpos"].astype(np.float64)
            fps = float(saved["fps"])
        if target.shape != (first["expected_steps"], 29) or fps != 50. or not np.isfinite(target).all():
            raise ValueError("Expected a finite 50Hz Q1 reference with 29 qpos values")
        states = {}
        for label, row in rows.items():
            with np.load(checked_file(row["trace"]), allow_pickle=False) as saved:
                state = saved["state"].astype(np.float64)
            count = row["executed_steps"]
            if (state.shape != (count, 57) or not 0 < count <= len(target)
                    or row["expected_steps"] != len(target) or not np.isfinite(state).all()
                    or abs(row["full_clip_completion"] - count / len(target)) > 1e-12):
                raise ValueError("Saved trace dimensions or completion denominator differ")
            if abs(errors(state, target[:count])["joint_rmse_rad"] - row["executed_joint_rmse_rad"]) > 2e-5:
                raise ValueError("Saved trace joint error differs from the audited receipt")
            states[label] = state
        count = min(map(len, states.values()))
        output.append({"clip": clip, "reference": first["reference"],
            "expected_steps": len(target), "common_prefix_steps": count,
            "common_prefix_seconds": count / fps, "common_prefix_fraction": count / len(target),
            "evaluations": {label: {"trace": rows[label]["trace"],
                "executed_steps": len(state), "full_clip_completion": rows[label]["full_clip_completion"],
                "failed": rows[label]["failed"],
                "executed_errors": errors(state, target[:len(state)]),
                "common_prefix_errors": errors(state[:count], target[:count])}
                for label, state in states.items()}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", nargs=3, action="append", required=True,
                        metavar=("LABEL", "RECEIPT", "AUDIT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len({entry[0] for entry in args.evaluation}) != len(args.evaluation):
        parser.error("Evaluation labels must be unique")
    evaluations = {label: load_evaluation(receipt, audit) for label, receipt, audit in args.evaluation}
    report = {"status": "AUDITED_COMMON_PREFIX_COMPARISON_COMPLETED", "source": identity(__file__),
        "evaluations": {label: {key: value for key, value in evaluation.items() if key != "rows"}
                        for label, evaluation in evaluations.items()},
        "rows": compare(evaluations), "new_physics_steps": 0, "new_policy_inference": 0,
        "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Descriptive comparison on a shared executed prefix, not a full-clip success result or a new usability gate. Original completion and failure remain authoritative. Post-control states use the original evaluation reference alignment; reused development clips are not an untouched test set."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
