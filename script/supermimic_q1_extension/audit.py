#!/usr/bin/env python3
"""Independently verify retained Q1 state and additional training in saved files."""

import argparse
import json
from pathlib import Path

from initialization import identity, check_identity, validate_contract, validated_parent, MODEL_KEYS, OPTIMIZER_KEYS


def equal_tree(left, right):
    import torch

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def finite_tree(value):
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    if isinstance(value, float):
        import math
        return math.isfinite(value)
    return True


def audit_run(run):
    import torch

    torch.set_num_threads(2)
    receipt = json.loads((run / "receipt.json").read_text())
    initial = torch.load(run / "step_000000.pt", map_location="cpu", mmap=True, weights_only=False)
    final = torch.load(run / "final.pt", map_location="cpu", mmap=True, weights_only=False)
    manifest = final["manifest"]
    evidence = manifest["initialization"]["evidence"]
    observed_parent = validated_parent(Path(evidence["comparison"]["path"]).parent,
                                       Path(evidence["checkpoint"]["path"]).parent)
    if observed_parent != evidence:
        raise ValueError("Continuation parent lineage differs from its completed pair")
    for item in evidence.values():
        if isinstance(item, dict) and {"path", "bytes", "sha256"} <= item.keys():
            check_identity(item)
    parent = torch.load(evidence["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    validate_contract(parent["manifest"], manifest)
    steps = manifest["args"]["steps"]
    # Independent integer arithmetic for the 2/5 update cadence.
    wm_updates = max(0, steps - manifest["warmup_steps"] + 1) * 2 // 5
    actor_updates = max(0, wm_updates - manifest["model_warmup_updates"])
    increments = {"wm_opt": wm_updates * manifest["wm_trainer"]["batches_per_update"],
                  "policy_opt": actor_updates,
                  "critic_opt": actor_updates * manifest["policy_trainer"]["critic_epochs"]
                                * manifest["policy_trainer"]["critic_minibatches"]}
    retained = {key: equal_tree(parent[key], initial[key])
                for key in MODEL_KEYS + OPTIMIZER_KEYS + ("bin_return", "bin_memory")}
    counters = {}
    for key, increment in increments.items():
        before, after = parent[key]["state"], final[key]["state"]
        passed = before.keys() == after.keys() and bool(before)
        passed = passed and all(int(after[index]["step"]) == int(before[index]["step"]) + increment for index in before)
        counters[key] = {"increment": increment, "passed": passed,
                         "initial_distinct": sorted({int(state["step"]) for state in before.values()}),
                         "final_distinct": sorted({int(state["step"]) for state in after.values()})}
    changed = {key: any(not torch.equal(initial[key][name], value) for name, value in final[key].items()
                        if value.is_floating_point()) for key in MODEL_KEYS}
    for item in receipt["artifacts"] + manifest["source_files"] + manifest["train_clips"]:
        check_identity(item)
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    offset = parent["manifest"]["learning_rate_step_offset"] + parent["step"]
    gates = {"parent_state_retained": all(retained.values()),
             "bounded_budget": steps >= 3000 and steps % 500 == 0,
             "initial_local_counters_zero": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0,
             "native_finished": receipt["status"] == "TRAINING_FINISHED_PENDING_EVALUATION",
             "fixed_additional_budget": receipt["step"] == final["step"] == steps
                  and receipt["env_steps"] == steps * manifest["sim"]["num_envs"],
             "optimizer_counters": all(row["passed"] for row in counters.values()),
             "local_update_counts": receipt["wm_updates"] == final["wm_updates"] == wm_updates
                  and receipt["policy_updates"] == final["policy_updates"] == actor_updates,
             "cumulative_schedule": manifest["learning_rate_step_offset"] == offset
                  and final["total_collection_rounds"] == receipt["total_collection_rounds"] == offset + steps,
             "manifest_unchanged": equal_tree(initial["manifest"], manifest)
                  and json.loads((run / "admission.json").read_text()) == json.loads(json.dumps(manifest)),
             "all_networks_changed": all(changed.values()),
             "finite": finite_tree({key: final[key] for key in MODEL_KEYS + OPTIMIZER_KEYS + ("bin_return", "bin_memory")})
                  and finite_tree(metrics),
             "complete_logging": bool(metrics) and [row["step"] for row in metrics] == list(range(25, steps + 1, 25)),
             "no_new_oom": all(row["health"]["oom_kill"] == manifest["baseline_health"]["oom_kill"] for row in metrics)}
    return {"status": "CONTINUATION_SAVE_AUDIT_PASSED" if all(gates.values()) else "CONTINUATION_SAVE_AUDIT_FAILED",
            "gates": gates, "parent_state_retention": retained, "optimizer_counters": counters,
            "networks_changed": changed, "source_checkpoint": evidence["checkpoint"],
            "initial": identity(run / "step_000000.pt"), "final": identity(run / "final.pt"),
            "training_receipt": identity(run / "receipt.json"), "code_commit": manifest["code_commit"],
            "physics_steps": 0, "optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_run(args.run)
    report["audit_source"] = identity(__file__)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    if not all(report["gates"].values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
