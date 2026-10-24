#!/usr/bin/env python3
"""Read and verify a saved horizon-arm initial state without changing the run."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--arm", choices=("A_H32", "B_H8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "script/supermimic_q1_horizon_v2"))
    import run_horizon_pair as entry
    import torch

    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    if Path(plan["root"]).resolve() != root:
        raise ValueError("Plan root differs from the audited source checkout")
    entry.verify_plan(plan)
    arm = next(row for row in plan["arms"] if row["name"] == args.arm)
    run = Path(arm["run"])
    saved_path = run / "step_000000.pt"
    parent_path = Path(plan["parent"]["checkpoint"]["path"])
    initial = torch.load(saved_path, map_location="cpu", mmap=True, weights_only=False)
    parent = torch.load(parent_path, map_location="cpu", mmap=True, weights_only=False)
    manifest = initial["manifest"]
    entry.contract.validate_contract(parent["manifest"], manifest)
    audit = entry.load_delegate(root / "script/supermimic_q1_extension/audit.py", "q1_initial_audit_helpers")
    keys = entry.contract.initialization.MODEL_KEYS + entry.contract.initialization.OPTIMIZER_KEYS + ("bin_return", "bin_memory")
    retained = {key: audit.equal_tree(parent[key], initial[key]) for key in keys}
    gates = {"all_eight_states_retained": all(retained.values()),
             "local_counters_zero": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0,
             "manifest_matches_admission": audit.equal_tree(json.loads(json.dumps(manifest)),
                                                            json.loads((run / "admission.json").read_text())),
             "registered_binding": manifest["horizon_experiment"]["plan"] == entry.contract.identity(args.plan)
                  and manifest["horizon_experiment"]["arm"] == args.arm,
             "parent_lineage_matches": manifest["initialization"]["evidence"] == plan["parent"],
             "cumulative_schedule": manifest["learning_rate_step_offset"] == initial["total_collection_rounds"]
                  == parent["total_collection_rounds"],
             "finite_saved_state": audit.finite_tree({key: initial[key] for key in keys})}
    for item in manifest["source_files"]:
        entry.contract.check_identity(item)
    result = {"status": "HORIZON_INITIAL_STATE_AUDIT_PASSED" if all(gates.values()) else "HORIZON_INITIAL_STATE_AUDIT_FAILED",
        "arm": args.arm, "gates": gates, "retained": retained,
        "initial": entry.contract.identity(saved_path), "parent": plan["parent"]["checkpoint"],
        "source": entry.contract.identity(__file__), "plan": entry.contract.identity(args.plan),
        "initial_optimizer_steps": manifest["initialization"]["initial_optimizer_steps"],
        "learning_rate_step_offset": manifest["learning_rate_step_offset"],
        "new_physics_steps": 0, "new_network_calls": 0, "new_optimizer_updates": 0,
        "limits": "Only the immutable initial saved state; not validation of later training or its final effect.",
        "storage_status": "LOCAL_ONLY"}
    entry.runner.write_json(args.output, result, exclusive=True)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
