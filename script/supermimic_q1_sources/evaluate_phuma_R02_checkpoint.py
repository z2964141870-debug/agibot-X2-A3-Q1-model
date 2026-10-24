#!/usr/bin/env python3
"""Compare R02 step zero and the audited step 5000 on development only."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_phuma_R02_checkpoint import identity
from runtime_guard import save


def metrics(row):
    with np.load(row["trace"]["path"], allow_pickle=False) as trace:
        state = trace["state"].astype(float)
    with np.load(row["reference"]["path"], allow_pickle=False) as reference:
        target = reference["qpos"].astype(float)
    if not len(state) or len(state) > len(target):
        raise ValueError("Invalid executed trace length")
    prefix = target[:len(state)]
    joints = state[:, 7:29] - prefix[:, 7:29]
    root_error = float(np.sqrt(np.mean(np.sum((state[:, :3] - prefix[:, :3]) ** 2, axis=1))))
    joint_error = float(np.sqrt(np.mean(joints ** 2)))
    full = not row["failed"] and len(state) == len(target)
    return {
        "executed_steps": len(state), "expected_steps": len(target), "failed": row["failed"],
        "completion": len(state) / len(target), "joint_rmse_rad": joint_error,
        "root_position_rmse_m": root_error,
        "leg_joint_rmse_rad": float(np.sqrt(np.mean(joints[:, :12] ** 2))),
        "reference_max_horizontal_displacement_m": float(np.linalg.norm(target[:, :2] - target[:1, :2], axis=1).max()),
        "policy_max_horizontal_displacement_m": float(np.linalg.norm(state[:, :2] - target[:1, :2], axis=1).max()),
        "full_clip_root_endpoint_error_m": float(np.linalg.norm(state[-1, :3] - target[-1, :3])) if full else None,
        "normalized_return": row["normalized_return"],
        "meets_tracking_gates": full and joint_error <= .15 and root_error <= .10,
    }


def health():
    cg = Path("/sys/fs/cgroup/memory")
    memory = int((cg / "memory.usage_in_bytes").read_text())
    counters = dict(line.split() for line in (cg / "memory.oom_control").read_text().splitlines())
    if memory >= int(28.75 * 1024 ** 3) or int(counters["oom_kill"]) != 14:
        raise RuntimeError("Development evaluation memory/OOM reserve failed")
    return {"cgroup_memory_bytes": memory, "oom_kill": int(counters["oom_kill"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.output.exists() or not args.output.resolve().is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent development output directory")
    experiment = root / "data/supermimic_for_q1/PHUMA_R02_expanded64_H32_12000_20261024"
    spec_path = experiment / "spec.json"
    spec = json.loads(spec_path.read_text())
    audit = json.loads(args.checkpoint_audit.read_text())
    if not (audit["status"] == "PARTIAL_TRAINING_CHECKPOINT_AUDIT_PASSED"
            and audit["checkpoint_step"] == 5000 and all(audit["gates"].values())
            and not audit["final_budget_completed"]):
        raise ValueError("An independently audited partial step5000 checkpoint is required")
    checkpoints = {label: Path(row["path"]) for label, row in zip(("initial", "step5000"), audit["files"])}
    for row in audit["files"]:
        if identity(Path(row["path"])) != row:
            raise ValueError("Audited checkpoint bytes changed")
    names = spec["split_clips"]["development"]
    if (len(names) != 6 or len(set(names)) != 6
            or set(names) & set(spec["split_clips"]["train"] + spec["split_clips"]["test"])):
        raise ValueError("Expected the original six isolated development clips")
    motions = Path(spec["motions"])
    manifest = json.loads((motions / "manifest.json").read_text())
    source = {row["clip"]: row for row in manifest["motions"]}
    if any(source[name]["split"] != "development" for name in names):
        raise ValueError("Development split disagrees with source manifest")
    before = health()
    args.output.mkdir()
    plan = {"status": "REGISTERED_DEVELOPMENT_ONLY_PARTIAL_CHECKPOINT_COMPARISON",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source": identity(Path(__file__).resolve()),
            "spec": identity(spec_path), "checkpoint_audit": identity(args.checkpoint_audit),
            "checkpoints": audit["files"], "clips": names, "split": "development",
            "test_clips": [], "optimizer_updates": 0, "health_before": before,
            "protocol": "Preserved corrected-contact MuJoCo evaluator; free root, frame0 reset, whole clip, no noise or DR",
            "tracking_gates": {"complete_without_early_failure": True, "joint_rmse_rad": .15, "root_position_rmse_m": .10},
            "storage_status": "LOCAL_ONLY"}
    save(args.output / "plan.json", plan)
    events, results, evidence = [], {}, []
    status = {"status": "RUNNING", "events": events, "plan": identity(args.output / "plan.json")}
    save(args.output / "status.json", status)
    core = root / "script/supermimic_for_q1"
    try:
        for label, checkpoint in checkpoints.items():
            destination = args.output / ("evaluation_" + label)
            audit_path = args.output / ("evaluation_audit_" + label + ".json")
            tasks = [
                ("evaluate_" + label, [str(core / "run_contact_corrected.py"), "evaluate", "--root", str(root),
                 "--motions", str(motions), "--checkpoint", str(checkpoint), "--output", str(destination), "--clips", *names]),
                ("audit_" + label, [str(core / "audit_evaluation.py"), "--root", str(root),
                 "--evaluation", str(destination), "--output", str(audit_path)]),
            ]
            for phase, command in tasks:
                health()
                log = args.output / (phase + ".log")
                with log.open("x") as stream:
                    result = subprocess.run([sys.executable, *command], cwd=root, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=600, check=False)
                events.append({"phase": phase, "exit_code": result.returncode, "log": identity(log), "health": health()})
                save(args.output / "status.json", {**status, "phase": phase})
                if result.returncode:
                    raise RuntimeError(f"{phase} exited {result.returncode}")
            receipt_path = destination / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            trace_audit = json.loads(audit_path.read_text())
            if (trace_audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
                    or trace_audit["receipt"] != identity(receipt_path)
                    or receipt["checkpoint"] != identity(checkpoint)
                    or sorted(row["clip"] for row in receipt["rows"]) != sorted(names)):
                raise ValueError("Development trajectory audit or checkpoint binding failed")
            evidence.extend([identity(receipt_path), identity(audit_path)])
            results[label] = {}
            for row in receipt["rows"]:
                for key in ("reference", "trace"):
                    if identity(Path(row[key]["path"])) != row[key]:
                        raise ValueError("Evaluation artifact changed")
                results[label][row["clip"]] = metrics(row)
        pairs = [{"clip": name, "split": "development", "bucket": source[name]["bucket"],
                  "source_group": source[name]["phuma_source"]["source_group"],
                  **{label: results[label][name] for label in checkpoints}} for name in names]
        for row in audit["files"]:
            if identity(Path(row["path"])) != row:
                raise ValueError("Checkpoint changed during development evaluation")
        if identity(spec_path) != plan["spec"]:
            raise ValueError("Original training spec changed")
        comparison = {"status": "DEVELOPMENT_PARTIAL_CHECKPOINT_MATCHED_PHYSICS_COMPLETED",
                      "plan": identity(args.output / "plan.json"), "pairs": pairs, "evidence": evidence,
                      "training_complete": False, "test_evaluated": False, "optimizer_updates": 0,
                      "health_after": health(), "storage_status": "LOCAL_ONLY",
                      "limits": "Development direction finding only. Prefix errors for early failures; no test success or hardware claim."}
        save(args.output / "comparison.json", comparison)
        save(args.output / "status.json", {**status, "status": "COMPLETED", "comparison": identity(args.output / "comparison.json")})
        print(json.dumps(comparison, allow_nan=False), flush=True)
    except BaseException as exc:
        save(args.output / "status.json", {**status, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
