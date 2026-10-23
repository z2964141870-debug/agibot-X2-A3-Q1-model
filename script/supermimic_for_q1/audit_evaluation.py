#!/usr/bin/env python3
"""Recompute complete-clip metrics from saved traces without stepping physics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.world_model.wm_trainer import reference_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    receipt_path = args.evaluation / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assets = Q1Assets(args.root)
    heads = assets.heads()
    results = []
    with torch.no_grad():
        for row in receipt["rows"]:
            for artifact in (row["reference"], row["trace"]):
                if identity(artifact["path"])["sha256"] != artifact["sha256"]:
                    raise ValueError("Evaluation artifact identity changed")
            library = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                                    device="cpu", active_indices=assets.active_indices)
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                state = torch.from_numpy(trace["state"])
                action = torch.from_numpy(trace["actions"]).clamp(-20., 20.)
                count = len(state)
                frames = torch.arange(count)
                previous = torch.cat([torch.zeros_like(action[:1]), action[:-1]], dim=0)
                reference = reference_at(library, frames)
                bodies = heads.robot_bodies(state)
                reward = heads.reward(state, action, previous, reference, bodies)["total"]
                terminated = heads.terminated(state, reference)
                joints = ((state[:, 7:29] - library.ref(frames).joint_pos) ** 2).mean(-1).sqrt()
                body = (bodies["pos"] - reference.body_pos_w).norm(dim=-1).mean(-1)
                expected = library.num_frames
                recomputed = {"expected_steps": expected, "executed_steps": count,
                              "failed": bool(terminated[-1]),
                              "normalized_return": float(reward.double().sum()) / (heads.reward_cfg.max_step_reward * expected),
                              "full_clip_completion": count / expected,
                              "executed_joint_rmse_rad": float(joints.double().square().mean().sqrt()),
                              "executed_body_error_mean_m": float(body.double().mean())}
                errors = {"reward": float(np.abs(reward.numpy() - trace["reward"]).max()),
                          "joint_rmse": float(np.abs(joints.numpy() - trace["joint_rmse_rad"]).max()),
                          "body_error": float(np.abs(body.numpy() - trace["body_error_m"]).max())}
                exact = all(recomputed[key] == row[key] for key in ("expected_steps", "executed_steps", "failed"))
                scalar_error = max(abs(recomputed[key] - row[key]) for key in recomputed
                                   if key not in ("expected_steps", "executed_steps", "failed"))
                passed = (exact and scalar_error < 2e-5 and max(errors.values()) < 2e-5
                          and not bool(terminated[:-1].any()) and count <= expected)
                result = {"clip": row["clip"], "split": row["split"], "passed": passed,
                          "recomputed": recomputed, "max_scalar_error": scalar_error,
                          "trace_field_errors": errors}
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
    gates = {"all_complete_without_early_failure": all(not r["recomputed"]["failed"]
                 and r["recomputed"]["executed_steps"] == r["recomputed"]["expected_steps"] for r in results),
             "mean_full_clip_return_at_least_0p5": float(np.mean([r["recomputed"]["normalized_return"] for r in results])) >= .5,
             "each_full_clip_joint_rmse_at_most_0p15": all(not r["recomputed"]["failed"]
                 and r["recomputed"]["executed_steps"] == r["recomputed"]["expected_steps"]
                 and r["recomputed"]["executed_joint_rmse_rad"] <= .15 for r in results)}
    groups = {}
    for split in ("training", "heldout"):
        selected = [r["recomputed"] for r in results if r["split"] == split]
        if selected:
            groups[split] = {"clips": len(selected), "failures": sum(r["failed"] for r in selected),
                             "mean_normalized_return": float(np.mean([r["normalized_return"] for r in selected])),
                             "mean_completion": float(np.mean([r["full_clip_completion"] for r in selected]))}
    group_error = max(abs(value - receipt["groups"][split][key])
                      for split, group in groups.items() for key, value in group.items())
    passed = all(r["passed"] for r in results) and gates == receipt["usability_gates"] and group_error < 2e-5
    report = {"status": "EVALUATION_TRACE_AUDIT_PASSED" if passed else "EVALUATION_TRACE_AUDIT_FAILED",
              "verified": passed, "usability_gates": gates, "groups": groups,
              "group_max_error": group_error, "clips": results, "receipt": identity(receipt_path),
              "source": identity(__file__), "physics_steps": 0, "optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "groups": groups, "usability_gates": gates}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
