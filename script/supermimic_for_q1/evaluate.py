#!/usr/bin/env python3
"""Deterministic full-clip evaluation in actual MuJoCo, including early failures."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
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
    assets = Q1Assets(args.root)
    if assets.manifest() != ck["manifest"]["assets"]:
        raise ValueError("Evaluation asset/control contract differs from training")
    for source in json.loads(Path(ck["manifest"]["backend_receipt"]["path"]).read_text())["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError("Backend changed since verification")
    heads = assets.heads(args.device)
    source_manifest = json.loads((args.motions / "manifest.json").read_text())
    expected = {Path(row["output"]["path"]).stem: row for row in source_manifest["motions"] if "output" in row}
    checks_path = Path(ck["manifest"]["reference_receipt"]["path"])
    if identity(checks_path)["sha256"] != ck["manifest"]["reference_receipt"]["sha256"]:
        raise ValueError("Independent reference receipt changed since training")
    checked = json.loads(checks_path.read_text())
    if identity(args.motions / "manifest.json")["sha256"] != checked["input_manifest"]["sha256"]:
        raise ValueError("Evaluation references differ from the independently verified dataset")
    qualified = {row["clip"] for row in checked["clips"] if row["qualified_for_training_pilot"]}
    names = args.clips or sorted(qualified)
    if not names or not set(names) <= qualified:
        raise ValueError("Evaluation requires independently qualified references")
    training_names = {Path(r["path"]).stem for r in ck["manifest"]["train_clips"]}
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for name in names:
            file = args.motions / (name + ".npz")
            if identity(file)["sha256"] != expected[name]["output"]["sha256"]:
                raise ValueError("Q1 evaluation reference identity differs from the prepared library")
            lib = MotionLibrary([file], TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
            n = lib.num_frames
            sim = Q1Sim(assets, lib, SimCfg(num_envs=1, threads=1, domain_randomization=False,
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
                   "normalized_return": reward_sum / (heads.reward_cfg.max_step_reward * n), "full_clip_completion": len(states) / n,
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
               "protocol": "Reference frame0 initial state; n targets/controls; complete frame denominator including early failures; no noise or DR; friction1; free base",
               "usability_gates": {
                   "all_complete_without_early_failure": all(not row["failed"] and row["executed_steps"] == row["expected_steps"] for row in rows),
                   "mean_full_clip_return_at_least_0p5": float(np.mean([row["normalized_return"] for row in rows])) >= .5,
                   "each_full_clip_joint_rmse_at_most_0p15": all(not row["failed"] and row["executed_steps"] == row["expected_steps"] and row["executed_joint_rmse_rad"] <= .15 for row in rows)},
               "evaluation_entry": identity(__file__),
               "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt["groups"], indent=2), flush=True)


if __name__ == "__main__":
    main()
