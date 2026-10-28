"""Read saved residual evidence; never construct a simulator or optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save, stamp


def prep(a):
    root = a.trial
    statuses = {}
    for name in ("preflight16", "preflight4096", "preflight4096_v02", "preflight4096_final", "diagnose33"):
        directory = root / name
        status = json.loads((directory / "status.json").read_text())
        guard = json.loads((directory / "train_guard.json").read_text())
        worker = json.loads((directory / "run/status.json").read_text())
        statuses[name] = {"pipeline": status, "worker": worker, "guard_status": guard["status"],
                          "cleanup": guard["cleanup"], "last_resources": guard.get("last_storage_check")}
    if statuses["preflight4096_final"]["pipeline"]["status"] != "COMPLETED" or statuses["diagnose33"]["pipeline"]["status"] != "COMPLETED":
        raise AssertionError("Final physical preflight and fallback diagnostic must complete")
    measured = json.loads((root / "diagnose33/run/comparison.json").read_text())
    aggregate = {}
    for mode in ("teacher", "static075", "gated"):
        values = [r[mode] for r in measured["repeats"]]
        keys = ("root_xy_rmse", "upper_rmse", "leg_rmse", "wrist_relative_rmse", "completion")
        aggregate[mode] = {key: sum(v["summary"]["validation"][key] for v in values) / 3 for key in keys}
        aggregate[mode]["failures_each_repeat"] = [v["summary"]["validation"]["failures"] for v in values]
        aggregate[mode]["fixed_demos_each_repeat"] = [v["fixed_demos"] for v in values]
        aggregate[mode]["intervention_each_repeat"] = [v["intervention"] for v in values]
    sources = json.loads(a.frozen_plan.read_text())["source"]
    for item in sources:
        if identity(item["path"]) != item:
            raise AssertionError("Historical frozen source changed")
    save(a.output, {"status": "UR04_PREPARATION_COMPLETED", "created_utc": stamp(), "optimizer_steps": 0,
        "formal_training_started": False, "statuses": statuses, "zero_head_fallback_diagnostic": aggregate,
        "calibration": json.loads((root / "calibration_v02.json").read_text()),
        "historical_frozen_sources_count": len(sources), "historical_frozen_sources_unchanged": True,
        "decision": "Disable heuristic fallback for RF13: about3% root improvement costs much upper precision and no fewer failures. Train only lower14 residual under intervention curriculum; retain fallback as an independently reported final diagnostic.",
        "recipe": {"envs": 4096, "actor_stochastic_coordinates": 14, "physical_joints": 22,
            "latent_std": .1, "residual_radius_rad": .15, "curriculum_cap": .75,
            "curriculum_actual_seconds": 1200, "baseline_episode_fraction": .25,
            "maximum_training_seconds": 3600, "soft_seconds": 3500, "external_stop_seconds": 3540,
            "fault_retry": False, "automatic_extension": False, "fallback_enabled": False},
        "limits": "Repeated fixed diagnostic clips are not an independent test. Simple fallback was not established as an improvement or safety mechanism.",
        "large_artifacts": "SERVER_LOCAL_ONLY", "extra_old_training": False})


def training(a):
    import torch
    run = a.trial / "run"
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=True)
    final = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    receipt = json.loads((run / "receipt.json").read_text())
    clock = json.loads((run / "training_clock.json").read_text())
    plan = json.loads((a.trial / "plan.json").read_text())
    checks = {}
    checks["format"] = initial["format"] == final["format"] == "Q1_RF13_BALANCE_HEAD_V1"
    checks["teacher_identity"] = initial["source_RF11"] == final["source_RF11"] == identity(final["source_RF11"]["path"])
    checks["teacher_fingerprint"] = initial["source_teacher_digest"] == final["source_teacher_digest"] and receipt["teacher_unchanged"]
    checks["budget"] = clock["active_started_monotonic"] is None and 0 < final["training_seconds"] <= 3600 and final["training_seconds"] == receipt["training_seconds"] == clock["accumulated_training_seconds"]
    checks["counters"] = final["sim_steps"] == final["iteration"] * 98304 and final["optimizer_steps"] == final["iteration"] * 20 and final["actor_optimizer_steps"] == max(final["iteration"] - 10, 0) * 20
    checks["zero_new_WM"] = final["new_world_model_updates"] == 0
    checks["head_changed"] = any(not torch.equal(value, final["actor"][key]) for key, value in initial["actor"].items() if key.startswith("head."))
    for group in ("actor", "critic"):
        checks[group + "_finite"] = all(bool(torch.isfinite(t).all()) for t in final[group].values())
    for key, value in initial["actor"].items():
        if not key.startswith("head."):
            checks["fixed_distribution_" + key] = torch.equal(value, final["actor"][key])
    checks["sources_frozen"] = all(identity(item["path"]) == item for item in plan["source"])
    checks["no_extension"] = final["maximum_training_seconds"] == 3600 and plan["maximum_training_seconds"] == 3600
    output = {"status": "SAVED_EVIDENCE_AUDIT_PASSED" if all(checks.values()) else "FAILED", "created_utc": stamp(),
              "checks": checks, "initial": identity(run / "initial.pt"), "final": identity(run / "final.pt"),
              "training_receipt": receipt, "new_physical_steps": 0, "new_optimizer_steps": 0,
              "source_files": len(plan["source"])}
    save(a.output, output)
    if not all(checks.values()):
        raise AssertionError("Saved residual audit failed: " + str([k for k, v in checks.items() if not v]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prep", "training"))
    p.add_argument("--trial", type=Path, required=True)
    p.add_argument("--frozen-plan", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.mode == "prep":
        if a.frozen_plan is None:
            p.error("Prep audit requires historical frozen plan")
        prep(a)
    else:
        training(a)
    result = json.loads(a.output.read_text())
    print(json.dumps({"output": str(a.output), "status": result["status"], "bytes": a.output.stat().st_size}))


if __name__ == "__main__":
    main()
