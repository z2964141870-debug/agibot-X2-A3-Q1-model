#!/usr/bin/env python3
"""Check saved training evidence in a fresh CPU process without optimizer updates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.q1_assets import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    receipt = json.loads((args.run / "receipt.json").read_text())
    initial_path, final_path = args.run / "step_000000.pt", args.run / "final.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    manifest = final["manifest"]
    expected_steps = manifest["args"]["steps"]
    expected_wm = 0
    accumulator = 0.
    for step in range(1, expected_steps + 1):
        if step >= manifest["args"]["warmup_steps"]:
            accumulator += .4
            if accumulator >= 1.:
                accumulator -= 1.
                expected_wm += 1
    expected_actor = max(0, expected_wm - manifest["args"]["model_warmup_updates"])
    expected = {"wm_opt": expected_wm * manifest["wm_trainer"]["batches_per_update"],
                "policy_opt": expected_actor,
                "critic_opt": expected_actor * manifest["policy_trainer"]["critic_epochs"]
                              * manifest["policy_trainer"]["critic_minibatches"]}
    counters, parameter_changes, finite = {}, {}, True
    for name, count in expected.items():
        states = final[name]["state"]
        values = [int(state["step"].item()) for state in states.values()]
        finite = finite and all(torch.isfinite(value).all().item()
                                for state in states.values() for value in state.values()
                                if isinstance(value, torch.Tensor))
        counters[name] = {"expected": count, "actual_distinct": sorted(set(values)),
                          "parameter_states": len(values), "initial_states": len(initial[name]["state"]),
                          "passed": bool(values) and set(values) == {count} and not initial[name]["state"]}
    for name in ("wm", "policy", "critic"):
        changes = [(key, float((value - initial[name][key]).abs().max()))
                   for key, value in final[name].items() if value.is_floating_point()]
        finite = finite and all(torch.isfinite(value).all().item() for value in final[name].values())
        parameter_changes[name] = {"max_abs": max(delta for _, delta in changes),
                                   "changed_tensors": sum(delta > 0 for _, delta in changes),
                                   "tensors": len(changes)}
    for artifact in receipt["artifacts"]:
        if identity(artifact["path"])["sha256"] != artifact["sha256"]:
            raise ValueError("Saved checkpoint identity differs from training receipt")
    changed_sources = [source["path"] for source in manifest["source_files"]
                       if identity(source["path"])["sha256"] != source["sha256"]]
    changed_inputs = [source["path"] for source in manifest["train_clips"]
                      if identity(source["path"])["sha256"] != source["sha256"]]
    metrics = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]
    for row in metrics:
        finite = finite and all(np.isfinite(value) for value in row.values()
                                if isinstance(value, (int, float)))
    gates = {"native_finished": receipt["status"] == "TRAINING_FINISHED_PENDING_EVALUATION",
             "fixed_budget": final["step"] == receipt["step"] == expected_steps
                             and receipt["env_steps"] == expected_steps * manifest["args"]["num_envs"],
             "initial_is_step_zero": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0,
             "update_counters": final["wm_updates"] == receipt["wm_updates"] == expected_wm
                                and final["policy_updates"] == receipt["policy_updates"] == expected_actor,
             "optimizer_steps": all(value["passed"] for value in counters.values()),
             "all_networks_changed": all(value["max_abs"] > 0 for value in parameter_changes.values()),
             "finite": bool(finite), "source_unchanged": not changed_sources,
             "input_unchanged": not changed_inputs,
             "metrics_reach_budget": bool(metrics) and metrics[-1]["step"] == expected_steps,
             "no_new_oom": all(row["health"]["oom_kill"] == manifest["baseline_health"]["oom_kill"] for row in metrics)}
    report = {"status": "TRAINING_SAVE_AUDIT_PASSED" if all(gates.values()) else "TRAINING_SAVE_AUDIT_FAILED",
              "gates": gates, "initial": identity(initial_path), "final": identity(final_path),
              "training_receipt": identity(args.run / "receipt.json"), "code_commit": manifest["code_commit"],
              "optimizer_counters": counters, "parameter_changes": parameter_changes,
              "changed_sources": changed_sources, "changed_inputs": changed_inputs,
              "metrics_rows": len(metrics), "physics_steps": 0, "optimizer_updates": 0,
              "source": identity(__file__), "storage_status": "LOCAL_ONLY"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
