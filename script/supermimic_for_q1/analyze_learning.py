#!/usr/bin/env python3
"""Describe learning logs and reward terms on preserved Q1 physics traces."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.world_model.wm_trainer import reference_at


def summarize_training(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "RUNNING" and "reward_mean" in row:
            rows.append(row)
    if not rows or [row["step"] for row in rows] != list(range(25, 3001, 25)):
        raise ValueError("Expected the 120 preserved R01 logging samples")
    keys = ["reward_mean", "joint_rmse_rad", "termination_fraction",
            "policy_grad_norm", "action_abs_mean", "imag_reward_mean",
            "wm/state_loss", "wm/nse_1step", "wm/nse_kstep", "critic_loss"]
    keys += sorted(key for key in rows[-1] if key.startswith("imag_r/"))
    windows = []
    for first, last in ((1, 250), (251, 750), (751, 1500),
                        (1501, 2250), (2251, 3000)):
        selected = [row for row in rows if first <= row["step"] <= last]
        means = {key: float(np.mean([row[key] for row in selected if key in row]))
                 for key in keys if any(key in row for row in selected)}
        if not all(np.isfinite(list(means.values()))):
            raise ValueError("Nonfinite training summary")
        windows.append({"first_collection_round": first, "last_collection_round": last,
                        "logged_samples": len(selected), "mean": means})
    return {"source": identity(path), "logged_samples": len(rows), "windows": windows,
            "last_env_steps": rows[-1]["env_steps"],
            "last_actor_updates": rows[-1]["policy_updates"],
            "limits": "Every 25th collection round; changing training distributions, not paired full-clip evaluation or evidence of convergence."}


def describe_evaluation(path, assets, heads):
    receipt = json.loads(path.read_text())
    rows = []
    maxima = {"motion_global_root_pos": heads.reward_cfg.dt * heads.reward_cfg.w_root_pos,
              "motion_global_root_ori": heads.reward_cfg.dt * heads.reward_cfg.w_root_ori,
              "motion_body_pos": heads.reward_cfg.dt * heads.reward_cfg.w_body_pos,
              "motion_body_ori": heads.reward_cfg.dt * heads.reward_cfg.w_body_ori,
              "motion_body_lin_vel": heads.reward_cfg.dt * heads.reward_cfg.w_body_lin_vel,
              "motion_body_ang_vel": heads.reward_cfg.dt * heads.reward_cfg.w_body_ang_vel}
    for row in receipt["rows"]:
        for key in ("reference", "trace"):
            if identity(row[key]["path"]) != row[key]:
                raise ValueError("Preserved evaluation input changed")
        library = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                                device="cpu", active_indices=assets.active_indices)
        with np.load(row["trace"]["path"], allow_pickle=False) as trace:
            state = torch.from_numpy(trace["state"])
            action = torch.from_numpy(trace["actions"]).clamp(-20., 20.)
            count = len(state)
            if count != row["executed_steps"] or library.num_frames != row["expected_steps"]:
                raise ValueError("Full-clip denominator changed")
            frames = torch.arange(count)
            previous = torch.cat([torch.zeros_like(action[:1]), action[:-1]], dim=0)
            ref = reference_at(library, frames)
            bodies = heads.robot_bodies(state)
            terms = heads.reward(state, action, previous, ref, bodies)
            recorded_error = float(np.abs(terms["total"].numpy() - trace["reward"]).max())
        means = {key: float(value.double().mean()) for key, value in terms.items()}
        if not all(np.isfinite(list(means.values()))):
            raise ValueError("Nonfinite reward terms")
        term_sum_error = abs(sum(v for k, v in means.items() if k != "total") - means["total"])
        normalized = float(terms["total"].double().sum()) / (
            heads.reward_cfg.max_step_reward * row["expected_steps"])
        if (recorded_error > 2e-5 or term_sum_error > 2e-5
                or abs(normalized - row["normalized_return"]) > 2e-5):
            raise ValueError("Reward decomposition disagrees with preserved evaluation")
        root_score = terms["motion_global_root_pos"] / maxima["motion_global_root_pos"]
        rows.append({"clip": row["clip"], "split": row["split"],
                     "executed_steps": count, "expected_steps": row["expected_steps"],
                     "failed": row["failed"], "reference": row["reference"], "trace": row["trace"],
                     "normalized_return": normalized, "mean_step_rewards": means,
                     "normalized_score_contributions": {
                         key: value / heads.reward_cfg.max_step_reward for key, value in means.items()},
                     "positive_term_fraction_of_own_maximum": {
                         key: means[key] / maximum for key, maximum in maxima.items()},
                     "root_position_score_fraction_below_0p01": float((root_score < .01).double().mean()),
                     "recorded_reward_max_error": recorded_error,
                     "sum_of_terms_error": term_sum_error})
    return {"receipt": identity(path), "checkpoint": receipt["checkpoint"],
            "training_step": receipt["training_step"], "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True, nargs="+")
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    assets = Q1Assets(args.root)
    heads = assets.heads()
    with torch.no_grad():
        evaluations = [describe_evaluation(path, assets, heads) for path in args.evaluation]
    package = Path(__file__).parent
    report = {"status": "SAVED_LEARNING_ANALYSIS_COMPLETED", "source": identity(__file__),
              "learning_sources": [identity(package / relative) for relative in (
                  "supermimic/q1_assets.py", "supermimic/world_model/heads.py",
                  "supermimic/envs/mdp/rewards.py", "supermimic/envs/mdp/motion_library.py",
                  "supermimic/world_model/wm_trainer.py")],
              "training": summarize_training(args.training_log), "evaluations": evaluations,
              "reward_config": asdict(heads.reward_cfg),
              "max_positive_step_reward": heads.reward_cfg.max_step_reward,
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY",
              "limits": "Descriptive analysis of existing development traces. Reward decomposition does not identify a unique causal mechanism or verify world-model action gradients. No networks loaded or training settings changed."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for evaluation in evaluations:
        for row in evaluation["rows"]:
            print(json.dumps({"training_step": evaluation["training_step"], "clip": row["clip"],
                              "normalized_return": row["normalized_return"],
                              "reward_fractions": row["positive_term_fraction_of_own_maximum"],
                              "root_score_below_0p01": row["root_position_score_fraction_below_0p01"]},
                             allow_nan=False), flush=True)
    print(json.dumps({"status": report["status"], "output": identity(args.output)}, allow_nan=False))


if __name__ == "__main__":
    main()
