"""Decompose recorded native actions and reward terms without new inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from audit_q1_hp_artifacts import identity


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "diagnosis", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    torch.set_num_threads(2)
    sys.path[:0] = [str(args.runtime/"src"), str(args.runtime)]
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, default_joint_pos_list

    library = json.loads(args.library.read_text())["splits"]
    rows = library["train"]+library["validation"]
    kin = Q1Kinematics(args.runtime/"q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    default = torch.tensor(default_joint_pos_list())
    alpha = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES])
    mdp = MDP(kin, motion, default, observation_mode="root_xy")
    diagnosis = json.loads((args.diagnosis/"status.json").read_text())
    with np.load(args.diagnosis/"native_trace.npz", allow_pickle=False) as saved:
        trace = {name:saved[name] for name in saved.files}
    outcomes = []
    with torch.inference_mode():
        for i, name in enumerate(diagnosis["selected"]):
            count = int(trace["steps"][i])
            frames = torch.tensor(trace["frames"][:count, i])+1
            action = torch.tensor(trace["actions"][:count, i])
            previous = torch.tensor(trace["previous"][:count, i])
            state = unflatten_state(torch.tensor(trace["states"][:count, i]))
            reward, terms = mdp.reward(state, frames, action, previous, terms=True)
            penalty_rate = mdp.cfg.w_action_rate*(action-previous).square().sum(-1)*.02
            penalty_limit = mdp.cfg.w_joint_limit*(torch.relu(state[4]-mdp.soft_hi)+torch.relu(mdp.soft_lo-state[4])).sum(-1)*.02
            reconstructed = terms["track"]+penalty_rate+penalty_limit
            torch.testing.assert_close(reward, reconstructed, atol=2e-6, rtol=2e-5)
            target = default+alpha*action
            target_delta = alpha*(action-previous)
            result = {"name":name, "steps":count, "failed":bool(trace["failed"][i]),
                "mean_reward_total":float(reward.mean()), "mean_tracking":float(terms["track"].mean()),
                "mean_action_rate_penalty":float(penalty_rate.mean()), "mean_soft_limit_penalty":float(penalty_limit.mean()),
                "action_absolute_p95":float(torch.quantile(action.abs().flatten(), .95)),
                "action_delta_rms":float((action-previous).square().mean().sqrt()),
                "physical_target_delta_rms_rad":float(target_delta.square().mean().sqrt()),
                "upper_target_reference_rmse_rad":float((target[:,14:]-motion.q[frames,14:]).square().mean().sqrt()),
                "upper_actual_target_rmse_rad":float((state[4][:,14:]-target[:,14:]).square().mean().sqrt()),
                "mean_abs_target_by_axis_rad":target.abs().mean(0).tolist(),
                "mean_abs_action_delta_by_axis":(action-previous).abs().mean(0).tolist()}
            outcomes.append(result)
    if torch.cuda.is_initialized():
        raise AssertionError("Recorded-action analysis should remain CPU-only")
    result = {"status":"RECORDED_NATIVE_ACTION_REWARD_DECOMPOSITION_COMPLETED", "created_utc":datetime.now(timezone.utc).isoformat(),
              "clips":outcomes, "source":identity(__file__), "sources":[identity(path) for path in (
                  args.library, args.diagnosis/"status.json", args.diagnosis/"native_trace.npz")],
              "physics_steps":0, "policy_inference_calls":0, "optimizer_updates":0,
              "limits":"Four recorded native rollout prefixes; full-library and pairwise common-prefix outcomes are reported separately. These terms describe the executed actions, not a unique causal attribution to the candidate's combined changes."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
