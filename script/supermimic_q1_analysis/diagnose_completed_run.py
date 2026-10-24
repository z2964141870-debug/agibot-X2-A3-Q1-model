#!/usr/bin/env python3
"""Inspect paired saved rewards and learned dynamics after a completed Q1 run."""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

from render_audited_trace import checked_file, identity


def verified_evaluation(receipt_record, audit_record):
    receipt_path = checked_file(receipt_record)
    audit_path = checked_file(audit_record)
    receipt = json.loads(receipt_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (receipt["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"
            or audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
            or audit["verified"] is not True or audit["receipt"] != receipt_record):
        raise ValueError("Expected the matching successful saved evaluation audit")
    audited = {row["clip"]: row for row in audit["clips"]}
    rows = {row["clip"]: row for row in receipt["rows"]}
    if (len(rows) != len(receipt["rows"]) or len(audited) != len(audit["clips"])
            or rows.keys() != audited.keys()
            or not all(row["passed"] is True for row in audited.values())):
        raise ValueError("Evaluation and audited clip coverage differ")
    checked_file(receipt["checkpoint"])
    for row in rows.values():
        for key in ("reference", "trace"):
            checked_file(row[key])
        if any(row[key] != audited[row["clip"]]["recomputed"][key]
               for key in ("expected_steps", "executed_steps", "failed")):
            raise ValueError("Saved evaluation counts differ from its audit")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    experiment = args.experiment.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not all(p.is_relative_to(root / "data/supermimic_for_q1")
               for p in (experiment, output)):
        raise ValueError("Analysis must stay under the Q1 artifact root")
    core = root / "script/supermimic_for_q1"
    extension = root / "script/supermimic_q1_extension"
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(extension))
    import pipeline
    plan = json.loads((experiment / "plan.json").read_text())
    pipeline.verify_plan(plan)
    summary = json.loads((experiment / "summary.json").read_text())
    if (summary["status"] != "CONTINUATION_PIPELINE_COMPLETED"
            or json.loads((experiment / "receipt.json").read_text())["status"]
            != "CONTINUATION_PIPELINE_COMPLETED"):
        raise ValueError("The original continuation must already be completed")
    evaluations = {
        phase: verified_evaluation(summary["evaluations"][phase]["receipt"],
                                   summary["evaluations"][phase]["audit"])
        for phase in ("initial", "final")
    }
    initial_rows = evaluations["initial"]["rows"]
    final_rows = evaluations["final"]["rows"]
    if ([r["clip"] for r in initial_rows] != [r["clip"] for r in final_rows]
            or any(a["reference"] != b["reference"]
                   for a, b in zip(initial_rows, final_rows))):
        raise ValueError("Paired reference coverage differs")
    baseline_oom = plan["baseline_health"]["oom_kill"]

    def health():
        current = pipeline.runner.health(root, baseline_oom)
        if (current["oom_kill"] != baseline_oom
                or current["cgroup_memory_bytes"] >= 30 * 1024**3 - 768 * 1024**2
                or current["gpu_temperature_c"] >= 86
                or current["gpu_free_mib"] < 2048):
            raise RuntimeError("Independent analysis resource guard refused execution")
        return current

    initial_health = health()
    import torch
    from contact_protocol import ContactCorrectedAssets
    from supermimic.q1_assets import TRACKED
    import analyze_learning
    measurement = importlib.import_module("measure_world_model")

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assets = ContactCorrectedAssets(root)
    for evaluation in evaluations.values():
        admission_path = Path(evaluation["checkpoint"]["path"]).parent / "admission.json"
        admission = json.loads(admission_path.read_text())
        if assets.manifest() != admission["assets"]:
            raise ValueError("Corrected contact assets differ from the admission")
    heads = assets.heads("cpu")
    output.mkdir(parents=True, exist_ok=False)
    rewards = {}
    with torch.no_grad():
        for phase in ("initial", "final"):
            health()
            rewards[phase] = analyze_learning.describe_evaluation(
                Path(summary["evaluations"][phase]["receipt"]["path"]), assets, heads)
    shares = []
    for first, last in zip(rewards["initial"]["rows"], rewards["final"]["rows"]):
        delta = last["normalized_return"] - first["normalized_return"]
        shares.append({
            "clip": last["clip"], "normalized_return_delta": delta,
            "normalized_term_deltas": {
                key: last["normalized_score_contributions"][key]
                - first["normalized_score_contributions"][key]
                for key in last["normalized_score_contributions"] if key != "total"},
        })
    model_results = {}
    original_assets, original_argv = measurement.Q1Assets, sys.argv
    try:
        measurement.Q1Assets = ContactCorrectedAssets
        for phase in ("initial", "final"):
            health()
            receipt_path = Path(summary["evaluations"][phase]["receipt"]["path"])
            model_output = output / ("world_model_" + phase + ".json")
            sys.argv = [str(core / "measure_world_model.py"), "--root", str(root),
                        "--checkpoint", evaluations[phase]["checkpoint"]["path"],
                        "--evaluation", str(receipt_path), "--output", str(model_output),
                        "--device", args.device, "--stride", "8"]
            measurement.main()
            model_results[phase] = {
                "identity": identity(model_output),
                "document": json.loads(model_output.read_text()),
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        measurement.Q1Assets, sys.argv = original_assets, original_argv
    final_health = health()
    pipeline.verify_plan(plan)
    report = {
        "status": "Q1_PAIRED_REWARD_AND_MODEL_DIAGNOSIS_COMPLETED",
        "source": identity(__file__), "plan": identity(experiment / "plan.json"),
        "summary": identity(experiment / "summary.json"),
        "reward_config": asdict(heads.reward_cfg),
        "max_positive_step_reward": heads.reward_cfg.max_step_reward,
        "rewards": rewards, "reward_changes": shares, "world_models": model_results,
        "assets": assets.manifest(), "initial_health": initial_health,
        "final_health": final_health, "frozen_verified_before_and_after": True,
        "sources": [identity(core / name) for name in
                    ("analyze_learning.py", "measure_world_model.py", "contact_protocol.py")],
        "new_physics_steps": 0, "new_policy_inference": 0,
        "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Two models on their own visited trajectories; descriptive, not a common-state causal model comparison. Reused development clips. Model prediction accuracy does not establish action-gradient correctness. Reward deltas are arithmetic attribution, not ablation.",
    }
    with (output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "receipt": identity(output / "receipt.json")}),
          flush=True)


if __name__ == "__main__":
    main()
