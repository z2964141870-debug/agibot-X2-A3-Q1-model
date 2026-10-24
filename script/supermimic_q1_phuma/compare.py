#!/usr/bin/env python3
"""Compare matched PHUMA physics traces, retaining root movement and true splits."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_training import identity


def compare(experiment):
    output = experiment / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads((experiment / "spec.json").read_text())
    manifest = json.loads((Path(spec["motions"]) / "manifest.json").read_text())
    source = {row["clip"]: row for row in manifest["motions"]}
    results = {}
    evidence = []
    for label in ("initial", "final"):
        receipt_path = experiment / ("evaluation_" + label) / "receipt.json"
        audit_path = experiment / ("evaluation_audit_" + label + ".json")
        receipt, audit = json.loads(receipt_path.read_text()), json.loads(audit_path.read_text())
        if audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED" or audit["receipt"] != identity(receipt_path):
            raise ValueError("Complete-clip evaluation audit missing or changed")
        if sorted(row["clip"] for row in receipt["rows"]) != sorted(spec["evaluation_clips"]):
            raise ValueError("Evaluation did not execute the registered clip set")
        evidence.extend([identity(receipt_path), identity(audit_path)])
        results[label] = {}
        for row in receipt["rows"]:
            if any(identity(row[key]["path"]) != row[key] for key in ("reference", "trace")):
                raise ValueError("A physics trace or reference changed")
            with np.load(row["trace"]["path"], allow_pickle=False) as trace, np.load(row["reference"]["path"], allow_pickle=False) as ref:
                state, target = trace["state"].astype(float), ref["qpos"].astype(float)
                prefix = target[:len(state)]
                root_rmse = float(np.sqrt(np.mean(np.sum((state[:, :3] - prefix[:, :3]) ** 2, axis=1))))
                errors = state[:, 7:29] - prefix[:, 7:29]
                groups = {key: float(np.sqrt(np.mean(errors[:, columns] ** 2)))
                          for key, columns in (("legs", slice(0, 12)), ("waist", slice(12, 14)), ("arms", slice(14, 22)))}
                metrics = {key: row[key] for key in ("executed_steps", "expected_steps", "failed", "normalized_return",
                           "full_clip_completion", "executed_joint_rmse_rad")}
                metrics.update({"root_position_rmse_m": root_rmse, "joint_group_rmse_rad": groups,
                    "reference_max_horizontal_displacement_m": float(np.linalg.norm(target[:, :2] - target[:1, :2], axis=1).max()),
                    "policy_max_horizontal_displacement_m": float(np.linalg.norm(state[:, :2] - target[:1, :2], axis=1).max()),
                    "completed_root_endpoint_error_m": float(np.linalg.norm(state[-1, :3] - target[-1, :3])) if len(state) == len(target) else None})
                metrics["meets_registered_tracking_gates"] = (not row["failed"] and len(state) == len(target)
                    and row["executed_joint_rmse_rad"] <= .15 and root_rmse <= .10)
                results[label][row["clip"]] = metrics
    pairs = [{"clip": name, "split": source[name]["split"], "bucket": source[name]["bucket"],
              "source_group": source[name]["phuma_source"]["source_group"],
              "initial": results["initial"][name], "final": results["final"][name]}
             for name in spec["evaluation_clips"]]
    summary = {}
    for split in ("train", "development", "test"):
        rows = [row for row in pairs if row["split"] == split]
        summary[split] = {"clips": len(rows)}
        for label in ("initial", "final"):
            summary[split][label] = {"failures": sum(row[label]["failed"] for row in rows),
                "tracking_gates_passed": sum(row[label]["meets_registered_tracking_gates"] for row in rows),
                "mean_completion": float(np.mean([row[label]["full_clip_completion"] for row in rows])),
                "mean_normalized_return": float(np.mean([row[label]["normalized_return"] for row in rows])),
                "mean_executed_joint_rmse_rad": float(np.mean([row[label]["executed_joint_rmse_rad"] for row in rows])),
                "mean_executed_root_rmse_m": float(np.mean([row[label]["root_position_rmse_m"] for row in rows]))}
    report = {"status": "PHUMA_MATCHED_PHYSICS_COMPARISON_COMPLETED", "spec": identity(experiment / "spec.json"),
              "summary": summary, "pairs": pairs, "evidence": evidence, "source": identity(__file__),
              "tracking_gates": {"full_clip_without_early_failure": True,
                 "executed_joint_rmse_cap_rad": .15, "executed_root_position_rmse_cap_m": .10},
              "physics_steps": 0, "optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
              "limits": "Early-failure errors cover only the executed prefix; completion is reported separately. No hardware result."}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "summary": summary, "output": identity(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    compare(parser.parse_args().experiment.resolve())
