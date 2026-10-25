#!/usr/bin/env python3
"""Check saved evaluation identities and arithmetic without a performance gate."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_reproduction import aggregate_rows, summarize_trace
from contact_protocol import ContactCorrectedAssets
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import TRACKED, identity
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
    if receipt["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("A completed physical evaluation is required")
    for artifact in (receipt["checkpoint"], receipt["reference_manifest"], receipt["reference_receipt"]):
        if identity(artifact["path"]) != artifact:
            raise ValueError("Evaluation input identity changed")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != receipt["assets"]:
        raise ValueError("Evaluation asset/control identity changed")
    heads = assets.heads("cpu")
    results, recomputed_rows = [], []
    with torch.no_grad():
        for row in receipt["rows"]:
            for artifact in (row["reference"], row["trace"]):
                if identity(artifact["path"]) != artifact:
                    raise ValueError("Saved evaluation artifact changed")
            library = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                                    device="cpu", active_indices=assets.active_indices)
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                state = torch.from_numpy(trace["state"])
                action = torch.from_numpy(trace["actions"])
                count = len(state)
                if (state.shape != (count, 57) or action.shape != (count, 22)
                        or count < 1 or count > library.num_frames
                        or not torch.isfinite(state).all() or not torch.isfinite(action).all()
                        or bool((action.abs() > receipt["action_limit"]).any())
                        or not np.array_equal(trace["reference_frames"], np.arange(count))):
                    raise ValueError("Malformed saved physical trace")
                frames = torch.arange(count)
                previous = torch.cat([torch.zeros_like(action[:1]), action[:-1]], dim=0)
                reference = reference_at(library, frames)
                bodies = heads.robot_bodies(state)
                reward = heads.reward(state, action, previous, reference, bodies)["total"]
                terminated = heads.terminated(state, reference)
                joints = ((state[:, 7:29] - library.ref(frames).joint_pos) ** 2).mean(-1).sqrt()
                body = (bodies["pos"] - reference.body_pos_w).norm(dim=-1).mean(-1)
                qpos, qvel = library.ref_qpos_qvel(torch.tensor([0]))
                initial = torch.cat([qpos, qvel], dim=-1)[0].numpy()
                recomputed = summarize_trace(state.numpy(), reward.numpy(), joints.numpy(), body.numpy(),
                                            terminated.numpy(), library.root_pos_w.numpy(),
                                            initial[:3], heads.reward_cfg.max_step_reward)
                errors = {"reward": float(np.abs(reward.numpy() - trace["reward"]).max()),
                          "joint_rmse": float(np.abs(joints.numpy() - trace["joint_rmse_rad"]).max()),
                          "body_error": float(np.abs(body.numpy() - trace["body_error_m"]).max()),
                          "initial_state": float(np.abs(initial - trace["initial_state"]).max())}
                exact_fields = ("expected_steps", "executed_steps", "failed", "early_terminated", "termination_step")
                exact = all(recomputed[key] == row[key] for key in exact_fields)
                scalar_error = max(float(np.abs(np.asarray(value) - np.asarray(row[key])).max())
                                   for key, value in recomputed.items() if key not in exact_fields)
                passed = (exact and scalar_error < 2e-5 and max(errors.values()) < 2e-5
                          and np.array_equal(terminated.numpy(), trace["terminated"])
                          and not bool(terminated[:-1].any()))
                result = {"clip": row["clip"], "split": row["split"], "arithmetic_verified": passed,
                          "recomputed": recomputed, "max_scalar_error": scalar_error,
                          "trace_field_errors": errors}
                results.append(result)
                recomputed_rows.append({"split": row["split"], **recomputed})
                print(json.dumps(result, allow_nan=False), flush=True)
    groups = aggregate_rows(recomputed_rows)
    same_groups = set(groups) == set(receipt["groups"])
    group_error = max(abs(value - receipt["groups"][split][key])
                      for split, group in groups.items() for key, value in group.items()) if same_groups else 1.
    passed = all(row["arithmetic_verified"] for row in results) and same_groups and group_error < 2e-5
    report = {
        "status": "EVALUATION_TRACE_AUDIT_PASSED" if passed else "EVALUATION_TRACE_AUDIT_FAILED",
        "verified": passed, "validation_scope": "Artifact identity, initial frame, termination accounting and saved metric arithmetic only; no performance threshold",
        "groups": groups, "group_max_error": group_error, "clips": results,
        "receipt": identity(receipt_path), "source": identity(__file__),
        "physics_steps": 0, "optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "groups": groups}, allow_nan=False), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
