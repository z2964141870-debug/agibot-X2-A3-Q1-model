#!/usr/bin/env python3
"""Apply the preregistered four-training-clip policy gate to verified traces."""

import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.a3_assets import identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Assessment output must be new")
    audit = json.loads(args.audit.read_text())
    require(audit["status"] == "TRAINING_AND_MATCHED_EVALUATION_VERIFIED", "Matched run audit did not pass")
    final_receipt = audit["evaluation_receipts"][1]
    require(identity(final_receipt["path"]) == final_receipt, "Final evaluation receipt changed")
    evaluation = json.loads(Path(final_receipt["path"]).read_text())
    require(evaluation["status"] == "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "Final evaluation did not finish")
    require(evaluation["checkpoint"] == audit["final_checkpoint"], "Final evaluation checkpoint differs")
    audited = {row["clip"]: row for row in audit["rows"]}
    require(len(audited) == len(audit["rows"]) == len(evaluation["rows"]) == 8, "Expected eight unique clips")
    rows = []
    for row in evaluation["rows"]:
        require(row["clip"] in audited and row["split"] == audited[row["clip"]]["split"], "Clip or split differs")
        for key in ("normalized_return", "full_clip_completion", "failed", "executed_steps"):
            require(row[key] == audited[row["clip"]][key]["trained"], f"Audited metric differs: {key}")
        require(identity(row["trace"]["path"]) == row["trace"], "Final trace changed")
        with np.load(row["trace"]["path"], allow_pickle=False) as trace:
            reward = trace["reward"]
            joint = trace["joint_rmse_rad"]
            require(reward.shape == joint.shape == (row["executed_steps"],), "Reward/joint trace shape differs")
            require(np.isfinite(reward).all() and np.isfinite(joint).all(), "Nonfinite policy metric")
            joint_rmse = float(np.sqrt(np.mean(np.square(joint))))
            normalized_return = float(reward.sum() / (.1 * row["expected_steps"]))
        require(np.isclose(joint_rmse, row["executed_joint_rmse_rad"], rtol=1e-7, atol=1e-9),
                "Joint error differs from trace")
        require(np.isclose(normalized_return, row["normalized_return"], rtol=1e-7, atol=1e-9),
                "Return differs from trace")
        rows.append({"clip": row["clip"], "split": row["split"], "failed": row["failed"],
                     "executed_steps": row["executed_steps"], "expected_steps": row["expected_steps"],
                     "normalized_return": normalized_return, "executed_joint_rmse_rad": joint_rmse,
                     "complete_without_failure": not row["failed"] and row["executed_steps"] == row["expected_steps"]})
    training = [row for row in rows if row["split"] == "training"]
    development = [row for row in rows if row["split"] == "heldout"]
    require(len(training) == len(development) == 4, "Expected four training and four development clips")
    decision = {"four_training_clips_complete": all(row["complete_without_failure"] for row in training),
                "mean_full_clip_return_at_least_half": float(np.mean([r["normalized_return"] for r in training])) >= .5,
                "every_training_joint_rmse_at_most_015rad": all(row["executed_joint_rmse_rad"] <= .15 for row in training)}
    result = {"status": "VERIFIED_TRACE_POLICY_ASSESSMENT_COMPLETED", "audit": identity(args.audit),
              "assessment_entry": identity(__file__), "final_evaluation": final_receipt,
              "criteria": {"training_clips": 4, "maximum_early_failures": 0,
                           "minimum_mean_full_clip_normalized_return": .5,
                           "maximum_each_training_executed_joint_rmse_rad": .15},
              "decision": decision, "passed_four_training_clip_gate": all(decision.values()), "rows": rows,
              "development_failures": sum(row["failed"] for row in development),
              "scope": "Four known training clips in deterministic MuJoCo only; development clips were examined previously. Not a generalization, hardware or paper-reproduction claim.",
              "storage_status": "LOCAL_ONLY"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
