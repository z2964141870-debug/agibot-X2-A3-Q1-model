#!/usr/bin/env python3
"""Inspect an immutable R02D intermediate save while its fixed-budget run continues."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget


def main(args):
    experiment = args.root.resolve(strict=True) / "data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025"
    if args.output.exists() or args.step < 500 or args.step % 500:
        raise ValueError("Use a new audit output and an immutable 500-round save")
    budget = MemoryBoundBudget(StorageBudget(experiment, experiment, 100 * 1024 ** 3,
        12 * 1024 ** 3, 12 * 1024 ** 3), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.check()
    run = experiment / "run"
    paths = [run / "step_000000.pt", run / f"step_{args.step:06d}.pt", run / "admission.json", experiment / "spec.json"]
    inputs = [identity(path) for path in paths]
    spec, admission = json.loads(paths[3].read_text()), json.loads(paths[2].read_text())
    if args.step > spec["steps"]:
        raise ValueError("Checkpoint step exceeds the registered budget")
    import torch

    torch.set_num_threads(1)
    initial, saved = [torch.load(path, map_location="cpu", mmap=True, weights_only=False) for path in paths[:2]]
    manifest, step = saved["manifest"], saved["step"]
    groups = max(0, step - manifest["warmup_steps"] + 1) * 2 // 5
    actor = max(0, groups - manifest["model_warmup_updates"])
    increments = {"wm_opt": groups * manifest["wm_trainer"]["batches_per_update"],
                  "policy_opt": actor, "critic_opt": actor * manifest["policy_trainer"]["critic_epochs"]
                  * manifest["policy_trainer"]["critic_minibatches"]}
    networks, counters, finite = {}, {}, True
    for key in ("wm", "policy", "critic"):
        current, original = saved[key], initial[key]
        same = current.keys() == original.keys()
        changed, maximum = 0, 0.
        for name, value in current.items():
            finite = bool(torch.isfinite(value).all()) and finite
            same = same and name in original and value.shape == original[name].shape
            if name in original and value.shape == original[name].shape and value.is_floating_point():
                delta = float((value - original[name]).abs().max())
                maximum = max(maximum, delta)
                changed += delta > 0
            budget.memory()
        networks[key] = {"same_keys_shapes": same, "changed_tensors": changed, "max_abs_change": maximum}
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
                         "initial_distinct": sorted({int(value["step"]) for value in first.values()}),
                         "saved_distinct": sorted({int(value["step"]) for value in last.values()})}
    for key in ("bin_return", "bin_memory"):
        finite = bool(torch.isfinite(saved[key]).all()) and finite
    lines = (run / "metrics.jsonl").read_bytes().split(b"\n")[:-1]
    prefix = [row for line in lines if line and (row := json.loads(line))["step"] <= step]
    frozen_changes = [record["path"] for record in spec["frozen_files"] if identity(record["path"]) != record]
    gates = {
        "registered_bounded_R02D": manifest["code_commit"] == spec["code_commit"]
            == "39397087493088605e080494d7728ff458d69939" and spec["steps"] == 7000
            and manifest["action_domain_correction"] == spec["action_domain_correction"],
        "requested_save_and_cumulative_counts": step == args.step
            and saved["total_collection_rounds"] == 5000 + step and manifest["sim"]["num_envs"] == 64,
        "initial_parent_offset_retained": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0
            and initial["total_collection_rounds"] == 5000 and manifest["learning_rate_step_offset"] == 5000,
        "local_update_counts": saved["wm_updates"] == groups and saved["policy_updates"] == actor,
        "three_adam_counters_match": all(row["passed"] for row in counters.values()),
        "three_networks_updated": all(row["same_keys_shapes"] and row["changed_tensors"] > 0 for row in networks.values()),
        "saved_network_optimizer_sampler_tensors_finite": finite,
        "manifest_matches_initial_and_admission": initial["manifest"] == manifest and json.loads(json.dumps(manifest)) == admission,
        "train_and_heldout_inputs_retained": sorted(Path(row["path"]).stem for row in manifest["train_clips"])
            == spec["split_clips"]["train"] and sorted(manifest["heldout_clips"])
            == sorted(spec["split_clips"]["development"] + spec["split_clips"]["test"]),
        "all516_frozen_files_unchanged": not frozen_changes and len(spec["frozen_files"]) == 516,
        "complete_finite_logged_prefix": [row["step"] for row in prefix] == list(range(25, step + 1, 25))
            and prefix[-1]["env_steps"] == 64 * step
            and all(math.isfinite(value) for row in prefix for value in row.values() if isinstance(value, (int, float))),
        "immutable_input_identities_unchanged": inputs == [identity(path) for path in paths],
        "no_cuda_context": not torch.cuda.is_initialized(),
    }
    report = {"status": "BOUNDED_R02D_INTERMEDIATE_SAVE_AUDIT_PASSED" if all(gates.values()) else "BOUNDED_R02D_INTERMEDIATE_SAVE_AUDIT_FAILED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_step": step,
              "gates": gates, "inputs": inputs, "source": identity(__file__), "networks": networks,
              "optimizer_counters": counters, "frozen_changes": frozen_changes,
              "health_before": before, "health_after": budget.check(),
              "new_physics_steps": 0, "new_policy_calls": 0, "new_optimizer_updates": 0,
              "final_budget_completed": False, "tracking_effectiveness_proven": False, "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report), flush=True)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    sys.exit(main(parser.parse_args()))
