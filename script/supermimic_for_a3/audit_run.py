#!/usr/bin/env python3
"""Verify a completed run and compare its matched deterministic evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.a3_assets import identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    torch.set_num_threads(2)
    receipt = json.loads((args.run / "receipt.json").read_text())
    require(receipt["status"] == "TRAINING_FINISHED_PENDING_EVALUATION", "Run did not finish its budget")
    initial_path, final_path = args.run / "step_000000.pt", args.run / "final.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    require(initial["step"] == initial["policy_updates"] == initial["wm_updates"] == 0,
            "Baseline is not this run's untrained initialization")
    require(initial["manifest"] == final["manifest"], "Checkpoint contracts differ")
    manifest = final["manifest"]
    require(final["step"] == manifest["args"]["steps"] == receipt["step"], "Collection budget mismatch")
    require(receipt["env_steps"] == final["step"] * manifest["sim"]["num_envs"], "Interaction count mismatch")
    for counter in ("wm_updates", "policy_updates"):
        require(final[counter] == receipt[counter] and final[counter] > 0, f"Invalid {counter}")
    final_identity = identity(final_path)
    require(final_identity in receipt["artifacts"], "Final checkpoint is not bound to the training receipt")

    networks = {}
    for name in ("wm", "policy", "critic"):
        before, after = initial[name], final[name]
        require(before.keys() == after.keys(), f"{name} tensor keys differ")
        changed = 0
        max_delta = 0.
        for key, tensor in after.items():
            require(tensor.shape == before[key].shape, f"{name}/{key} shape differs")
            require(bool(torch.isfinite(tensor).all()), f"Nonfinite {name}/{key}")
            if not torch.equal(tensor, before[key]):
                changed += 1
                max_delta = max(max_delta, float((tensor.double() - before[key].double()).abs().max()))
        require(changed > 0, f"No {name} tensors changed")
        networks[name] = {"tensors": len(after), "changed_tensors": changed, "max_abs_delta": max_delta}

    cfg = manifest["policy_trainer"]
    expected_steps = {"wm_opt": final["wm_updates"] * manifest["wm_trainer"]["batches_per_update"],
                      "policy_opt": final["policy_updates"],
                      "critic_opt": final["policy_updates"] * cfg["critic_epochs"] * cfg["critic_minibatches"]}
    optimizers = {}
    for name, expected in expected_steps.items():
        states = final[name]["state"]
        counts = [int(state["step"]) for state in states.values()]
        require(counts and all(count == expected for count in counts), f"{name} step count mismatch")
        for state in states.values():
            require(all(bool(torch.isfinite(value).all()) for value in state.values() if torch.is_tensor(value)),
                    f"Nonfinite {name} state")
        optimizers[name] = {"parameter_states": len(states), "steps_per_parameter": expected}

    evaluations = [json.loads(path.read_text()) for path in (args.baseline, args.trained)]
    for evaluation, path in zip(evaluations, (initial_path, final_path)):
        require(evaluation["status"] == "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "Incomplete evaluation")
        require(evaluation["checkpoint"] == identity(path), "Evaluation checkpoint identity mismatch")
        for row in evaluation["rows"]:
            require(identity(row["trace"]["path"]) == row["trace"], "Evaluation trace changed")
            require(identity(row["reference"]["path"]) == row["reference"], "Evaluation reference changed")
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                count = len(trace["reward"])
                require(0 < count == row["executed_steps"] <= row["expected_steps"], "Trace length mismatch")
                require(trace["state"].shape == (count, 83) and trace["actions"].shape == (count, 29),
                        "Trace state/action dimensions differ")
                require(all(np.isfinite(trace[key]).all() for key in trace.files), "Nonfinite evaluation trace")
                require(np.isclose(trace["reward"].sum() / (.1 * row["expected_steps"]), row["normalized_return"],
                                   rtol=1e-7, atol=1e-9), "Full-clip return differs from trace")
                require(row["full_clip_completion"] == count / row["expected_steps"], "Completion differs from trace")
                require(count == row["expected_steps"] or row["failed"], "Early termination missing from failures")
    require(evaluations[0]["protocol"] == evaluations[1]["protocol"], "Evaluation protocols differ")
    before_rows, after_rows = ({row["clip"]: row for row in ev["rows"]} for ev in evaluations)
    require(before_rows.keys() == after_rows.keys(), "Evaluation clip sets differ")
    training_names = {Path(record["path"]).stem for record in manifest["train_clips"]}
    require(set(before_rows) == training_names | set(manifest["heldout_clips"]), "Expected all training and reserved clips")
    rows = []
    for clip, before in before_rows.items():
        after = after_rows[clip]
        for key in ("reference", "expected_steps", "split"):
            require(before[key] == after[key], f"Unmatched {clip}/{key}")
        require(before["split"] == ("training" if clip in training_names else "heldout"), "Split mismatch")
        row = {"clip": clip, "split": before["split"]}
        for key in ("normalized_return", "full_clip_completion", "failed", "executed_steps"):
            row[key] = {"initial": before[key], "trained": after[key]}
        row["return_delta"] = after["normalized_return"] - before["normalized_return"]
        rows.append(row)
    groups = {}
    for split in ("training", "heldout"):
        group = [row for row in rows if row["split"] == split]
        require(group, f"Missing {split} evaluation")
        for evaluation in evaluations:
            selected = [row for row in evaluation["rows"] if row["split"] == split]
            expected = {"clips": len(selected), "failures": sum(row["failed"] for row in selected),
                        "mean_normalized_return": float(np.mean([row["normalized_return"] for row in selected])),
                        "mean_completion": float(np.mean([row["full_clip_completion"] for row in selected]))}
            require(expected == evaluation["groups"][split], "Evaluation group differs from clip results")
        groups[split] = {"clips": len(group), "mean_return_delta": float(np.mean([r["return_delta"] for r in group])),
                         "initial": evaluations[0]["groups"][split], "trained": evaluations[1]["groups"][split]}
    result = {"status": "TRAINING_AND_MATCHED_EVALUATION_VERIFIED", "run_receipt": identity(args.run / "receipt.json"),
              "initial_checkpoint": identity(initial_path), "final_checkpoint": final_identity,
              "audit_entry": identity(__file__), "networks": networks, "optimizers": optimizers,
              "groups": groups, "rows": rows, "evaluation_receipts": [identity(p) for p in (args.baseline, args.trained)],
              "storage_status": "LOCAL_ONLY", "not_hardware_validated": True,
              "interpretation": "Verification confirms execution and matched evaluation; performance is reported without a success override."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "groups": groups, "networks": networks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
