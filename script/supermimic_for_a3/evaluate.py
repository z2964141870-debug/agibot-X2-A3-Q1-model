#!/usr/bin/env python3
"""Deterministic full-clip evaluation in actual MuJoCo, including early failures."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.a3_sim import A3Sim, SimCfg
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.policy.nets import Policy, PolicyCfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clips", nargs="+", default=None)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = PolicyCfg(**ck["manifest"]["policy"])
    policy = Policy(cfg).to(args.device).eval()
    policy.load_state_dict(ck["policy"], strict=True)
    assets = A3Assets(args.root)
    if assets.manifest() != ck["manifest"]["assets"]:
        raise ValueError("Evaluation asset/control contract differs from training")
    for source in json.loads(Path(ck["manifest"]["backend_receipt"]["path"]).read_text())["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError("Backend changed since verification")
    heads = assets.heads(args.device)
    names = args.clips or [f.stem for f in sorted(args.motions.glob("*.npz"))]
    training_names = {Path(r["path"]).stem for r in ck["manifest"]["train_clips"]}
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for name in names:
            file = args.motions / (name + ".npz")
            lib = MotionLibrary([file], TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
            n = lib.num_frames
            sim = A3Sim(assets, lib, SimCfg(num_envs=1, threads=1, domain_randomization=False,
                          observation_noise=False, action_noise=0., episode_steps=n + 1), heads, seed=0)
            sim.reset(sim.ids, torch.tensor([0], device=args.device))
            reward_sum = 0.
            joint_errors, body_errors, states, actions, rewards = [], [], [], [], []
            failed = False
            try:
                for t in range(n):
                    obs, goal = sim.policy_inputs()
                    action = policy(obs, goal)
                    result = sim.step(action, auto_reset=False)
                    reward = float(result["reward"][0])
                    reward_sum += reward
                    joint_errors.append(float(result["joint_rmse"][0]))
                    body_errors.append(float(result["body_error_m"][0]))
                    states.append(result["state"][0].cpu().numpy())
                    actions.append(action[0].cpu().numpy())
                    rewards.append(reward)
                    if bool(result["terminated"][0]):
                        failed = True
                        break
            finally:
                sim.close()
            trace = args.output / (name + ".npz")
            np.savez_compressed(trace, state=np.asarray(states), actions=np.asarray(actions),
                                reward=np.asarray(rewards), joint_rmse_rad=np.asarray(joint_errors),
                                body_error_m=np.asarray(body_errors))
            row = {"clip": name, "split": "training" if name in training_names else "heldout",
                   "expected_steps": n, "executed_steps": len(states), "failed": failed,
                   "normalized_return": reward_sum / (.1 * n), "full_clip_completion": len(states) / n,
                   "executed_joint_rmse_rad": float(np.sqrt(np.mean(np.square(joint_errors)))),
                   "executed_body_error_mean_m": float(np.mean(body_errors)),
                   "reference": identity(file), "trace": identity(trace)}
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    groups = {}
    for split in ("training", "heldout"):
        selected = [r for r in rows if r["split"] == split]
        if selected:
            groups[split] = {"clips": len(selected), "failures": sum(r["failed"] for r in selected),
                             "mean_normalized_return": float(np.mean([r["normalized_return"] for r in selected])),
                             "mean_completion": float(np.mean([r["full_clip_completion"] for r in selected]))}
    receipt = {"status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "checkpoint": identity(args.checkpoint),
               "training_step": ck["step"], "policy_updates": ck["policy_updates"], "groups": groups,
               "rows": rows, "seconds": time.monotonic() - started, "simulator": "MuJoCo 3.3.7 CPU native rollout",
               "protocol": "first frame, full clip denominator including early falls; no noise or DR; friction 1",
               "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
