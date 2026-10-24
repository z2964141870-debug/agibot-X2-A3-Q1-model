#!/usr/bin/env python3
"""Inspect the last saved R02C state without reinterpreting its failed terminal."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from prepare_amass_body import identity


def audit(root, output, expected_oom):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget

    experiment = root / "data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025"
    budget = MemoryBoundBudget(StorageBudget(experiment, experiment, 100 * 1024 ** 3,
                                            12 * 1024 ** 3, 12 * 1024 ** 3),
                               int(28.75 * 1024 ** 3), expected_oom)
    before = budget.check()
    receipt_path = experiment / "run/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    spec_path = experiment / "spec.json"
    spec = json.loads(spec_path.read_text())
    run = experiment / "run"
    if (receipt["status"] != "FAILED" or receipt["step"] != 1810
            or receipt["error"] != "RuntimeError: Nonfinite continuation training metrics"):
        raise ValueError("Require the original unchanged R02C numerical failure")
    paths = [run / "step_000000.pt", run / "step_001500.pt", receipt_path, spec_path,
             run / "admission.json", run / "metrics.jsonl", experiment / "controller_status.json",
             experiment / "storage_guard.json"]
    inputs = [identity(path) for path in paths]
    import torch

    torch.set_num_threads(1)
    initial = torch.load(paths[0], map_location="cpu", mmap=True, weights_only=False)
    saved = torch.load(paths[1], map_location="cpu", mmap=True, weights_only=False)
    manifest = saved["manifest"]
    step = saved["step"]
    wm_updates = max(0, step - manifest["warmup_steps"] + 1) * 2 // 5
    actor_updates = max(0, wm_updates - manifest["model_warmup_updates"])
    increments = {"wm_opt": wm_updates * manifest["wm_trainer"]["batches_per_update"],
                  "policy_opt": actor_updates,
                  "critic_opt": actor_updates * manifest["policy_trainer"]["critic_epochs"]
                                * manifest["policy_trainer"]["critic_minibatches"]}
    finite, counters, networks = True, {}, {}
    for key in ("wm", "policy", "critic"):
        current, original = saved[key], initial[key]
        same = current.keys() == original.keys()
        changed = 0
        delta = 0.
        for name, value in current.items():
            finite = bool(torch.isfinite(value).all()) and finite
            same = same and name in original and value.shape == original[name].shape
            if same and value.is_floating_point():
                difference = float((value - original[name]).abs().max())
                delta = max(delta, difference)
                changed += difference > 0
            budget.memory()
        networks[key] = {"same_keys_shapes": same, "changed_tensors": changed, "max_abs_change": delta}
    for key, increment in increments.items():
        first, last = initial[key]["state"], saved[key]["state"]
        matched = bool(first) and first.keys() == last.keys()
        for index, state in last.items():
            for value in state.values():
                if torch.is_tensor(value):
                    finite = bool(torch.isfinite(value).all()) and finite
            matched = matched and index in first and int(state["step"]) == int(first[index]["step"]) + increment
            budget.memory()
        counters[key] = {"increment": increment, "passed": matched,
                         "initial_distinct": sorted({int(state["step"]) for state in first.values()}),
                         "saved_distinct": sorted({int(state["step"]) for state in last.values()})}
    for key in ("bin_return", "bin_memory"):
        finite = bool(torch.isfinite(saved[key]).all()) and finite
    metrics = [json.loads(line) for line in paths[5].read_text().splitlines()]
    prefix = [row for row in metrics if row["step"] <= step]
    frozen_changes = [row["path"] for row in spec["frozen_files"] if identity(row["path"]) != row]
    gates = {"original_failure_retained": json.loads(paths[6].read_text())["status"] == "FAILED"
             and json.loads(paths[7].read_text())["status"] == "WORKER_FAILED",
             "saved1500_not_unsaved1810": step == 1500 < receipt["step"] < spec["steps"] == 7000
             and saved["total_collection_rounds"] == 6500 and not (run / "final.pt").exists(),
             "initial_zero_retained_parent_counters": initial["step"] == initial["wm_updates"]
             == initial["policy_updates"] == 0 and initial["total_collection_rounds"] == 5000,
             "saved_local_updates": saved["wm_updates"] == wm_updates == 549
             and saved["policy_updates"] == actor_updates == 499,
             "three_adam_counters_match": all(row["passed"] for row in counters.values()),
             "three_networks_changed": all(row["same_keys_shapes"] and row["changed_tensors"] > 0
                                           for row in networks.values()),
             "saved_tensors_finite": finite,
             "manifest_matches_initial_and_admission": initial["manifest"] == manifest
             and json.loads(json.dumps(manifest)) == json.loads(paths[4].read_text()),
             "code_commit_matches_spec": manifest["code_commit"] == spec["code_commit"],
             "registered_train_and_heldout_lists": sorted(Path(row["path"]).stem for row in manifest["train_clips"])
             == spec["split_clips"]["train"] and sorted(manifest["heldout_clips"])
             == sorted(spec["split_clips"]["development"] + spec["split_clips"]["test"]),
             "checkpoint_bound_to_failure": inputs[1] in receipt["artifacts"],
             "all496_frozen_unchanged": not frozen_changes and len(spec["frozen_files"]) == 496,
             "complete_finite_prefix": [row["step"] for row in prefix] == list(range(25, 1501, 25))
             and all(math.isfinite(value) for row in prefix for value in row.values()
                     if isinstance(value, (int, float))),
             "input_identities_unchanged": inputs == [identity(path) for path in paths],
             "no_cuda_context": not torch.cuda.is_initialized()}
    report = {"status": "FAILED_R02C_SAVED1500_AUDIT_PASSED" if all(gates.values()) else "FAILED_R02C_SAVED1500_AUDIT_FAILED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "inputs": inputs,
              "checkpoint": inputs[1], "source": identity(__file__), "optimizer_counters": counters,
              "networks": networks, "frozen_changes": frozen_changes,
              "health_before": before, "health_after": budget.check(),
              "training_terminal": "FAILED_UNCHANGED", "new_physics_steps": 0, "new_optimizer_updates": 0,
              "tracking_effectiveness_proven": False, "storage_status": "LOCAL_ONLY"}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raise SystemExit(audit(args.root.resolve(strict=True), args.output, args.expected_oom_kill))
