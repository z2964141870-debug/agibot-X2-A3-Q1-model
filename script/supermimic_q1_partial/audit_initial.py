#!/usr/bin/env python3
"""Check the new step0 against the genuine partial parent, without stepping."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import parent


sys.path.insert(0, str(parent.EXTENSION))
spec = importlib.util.spec_from_file_location("q1_retention_helpers", parent.EXTENSION / "audit.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    torch.set_num_threads(1)
    spec = json.loads((args.experiment / "spec.json").read_text())
    initial_path = args.experiment / "run/step_000000.pt"
    initial_record = parent.identity(initial_path)
    observed = parent.validated_parent(spec["partial_development_directory"], spec["parent_run"])
    if observed != spec["partial_parent"]:
        raise ValueError("Registered partial parent changed")
    before = torch.load(observed["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    after = torch.load(initial_path, map_location="cpu", mmap=True, weights_only=False)
    retained = {key: helpers.equal_tree(before[key], after[key])
                for key in parent.MODEL_KEYS + parent.OPTIMIZER_KEYS + ("bin_return", "bin_memory")}
    parent.validate_contract(before["manifest"], after["manifest"])
    frozen_changes = [row["path"] for row in spec["frozen_files"] if parent.identity(row["path"]) != row]
    gates = {"all_eight_parent_trees_exact": all(retained.values()),
             "initial_local_counters_zero": after["step"] == after["wm_updates"] == after["policy_updates"] == 0,
             "cumulative_offset5000": after["total_collection_rounds"] == after["manifest"]["learning_rate_step_offset"] == 5000,
             "fixed_remaining7000": after["manifest"]["args"]["steps"] == spec["steps"] == 7000,
             "parent_failure_retained": after["manifest"]["initialization"]["parent_terminal_state"] == "FAILED_UNCHANGED_NOT_COMPLETED",
             "fresh_explicit_seed": after["manifest"]["args"]["seed"] == spec["seed"] != before["manifest"]["args"]["seed"],
             "source_code_commit": after["manifest"]["code_commit"] == spec["code_commit"],
             "manifest_matches_admission": json.loads(json.dumps(after["manifest"]))
                 == json.loads((args.experiment / "run/admission.json").read_text()),
             "all_frozen_unchanged": not frozen_changes,
             "finite_state": helpers.finite_tree({key: after[key] for key in retained}),
             "initial_save_stable": parent.identity(initial_path) == initial_record,
             "no_cuda_context": not torch.cuda.is_initialized()}
    report = {"status": "PARTIAL_CONTINUATION_INITIAL_RETENTION_PASSED" if all(gates.values()) else "PARTIAL_CONTINUATION_INITIAL_RETENTION_FAILED",
              "gates": gates, "retained": retained, "parent": observed["checkpoint"], "initial": initial_record,
              "frozen_files": len(spec["frozen_files"]), "frozen_changes": frozen_changes,
              "source": parent.identity(__file__), "new_physics_steps": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
