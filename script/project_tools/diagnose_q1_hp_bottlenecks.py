"""Decompose saved Q1 tracking outcomes without stepping physics or learning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from analyze_q1_hp_trial import identity, support, validate_trace


def mean_rows(rows, key):
    return float(np.mean([row[key] for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import mujoco
    import torch

    torch.set_num_threads(8)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from tools.x2_geometry import CollisionGeometry
    from unisupermimic.rl.algos.supermimic.geometry import rot_angle_sq, unflatten_state, yaw_mat
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, default_joint_pos_list
    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy

    folder = args.trial / "evaluation"
    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"] + library["splits"]["validation"]
    outcome = json.loads((folder / "final.json").read_text())
    with np.load(folder / "final_trajectory.npz", allow_pickle=False) as saved:
        trace = {key: saved[key] for key in saved.files}
    validate_trace(trace, len(rows))
    device = torch.device("cuda:0")
    scene = args.runtime / "q1_assets/scene.xml"
    kin = Q1Kinematics(scene, device)
    motion = MotionLib([row["path"] for row in rows], kin, device, src_fps=30.)
    default_q = torch.tensor(default_joint_pos_list(), device=device)
    mdp = MDP(kin, motion, default_q, observation_mode="root_xy")
    checkpoint_path = args.trial / "run/final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy = load_policy(SimpleNamespace(supermimic_mdp=mdp, device=device), checkpoint)
    alpha = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES], device=device)
    context = torch.zeros(len(rows), mdp.context_dim, device=device)
    context[:, 0] = 1.0
    previous_action = torch.zeros(len(rows), 22, device=device)
    history = None
    names = ("body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity", "root_position", "root_orientation", "action_rate_penalty", "soft_limit_penalty", "total")
    values = {name: np.zeros((len(trace["states"]), len(rows))) for name in names}
    upper_fixed = np.zeros_like(values["total"])
    inferred = np.zeros((len(trace["states"]), len(rows), 22), dtype=np.float32)
    targets = np.zeros_like(inferred)
    torque_estimate = np.zeros_like(inferred)
    model = mujoco.MjModel.from_xml_path(str(scene))
    kp = torch.tensor(model.actuator_gainprm[:, 0], device=device, dtype=torch.float32)
    kd = torch.tensor(-model.actuator_biasprm[:, 2], device=device, dtype=torch.float32)
    effort = model.actuator_forcerange[:, 1].copy()
    reconstructed_returns = np.zeros(len(rows))
    max_reward_formula_error = 0.0
    with torch.inference_mode():
        for step in range(int(trace["steps"].max())):
            before = trace["initial"] if step == 0 else trace["states"][step - 1]
            state = unflatten_state(torch.tensor(before, device=device))
            frames = torch.minimum(motion.clip_start + step, motion.clip_end - 1)
            obs = mdp.obs(state, previous_action, context, frames)
            history = obs[:, None].repeat(1, int(checkpoint["args"]["hist"]), 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            action = policy(history, mdp.goal(frames, int(checkpoint["args"]["goal_frames"])))
            target = default_q + alpha * action
            inferred[step] = action.cpu().numpy()
            targets[step] = target.cpu().numpy()
            # This is a pre-step PD estimate, not a recorded Warp substep force.
            torque_estimate[step] = (kp * (target - state[4]) - kd * state[5]).cpu().numpy()
            after = unflatten_state(torch.tensor(trace["states"][step], device=device))
            next_frames = frames + 1
            position, rotation, v, w, q, qd = after
            bp, br, bv, bw = kin.forward(*after)
            rp, rr = motion.root_p[next_frames], motion.root_R[next_frames]
            delta = yaw_mat(rotation @ rr.transpose(-1, -2))
            anchor = torch.cat((position[:, :2], rp[:, 2:3]), -1)
            reference_bp = anchor[:, None] + (delta[:, None] @ (motion.body_p[next_frames] - rp[:, None]).unsqueeze(-1)).squeeze(-1)
            terms = {
                "body_position": torch.exp(-((reference_bp - bp) ** 2).sum(-1).mean(-1) / mdp.cfg.body_pos_std ** 2) * .02,
                "body_orientation": torch.exp(-rot_angle_sq(delta[:, None] @ motion.body_R[next_frames], br).mean(-1) / mdp.cfg.body_ori_std ** 2) * .02,
                "body_linear_velocity": torch.exp(-((motion.body_v[next_frames] - bv) ** 2).sum(-1).mean(-1) / mdp.cfg.body_linvel_std ** 2) * .02,
                "body_angular_velocity": torch.exp(-((motion.body_w[next_frames] - bw) ** 2).sum(-1).mean(-1) / mdp.cfg.body_angvel_std ** 2) * .02,
                "root_position": .5 * torch.exp(-((rp - position) ** 2).sum(-1) / mdp.cfg.root_pos_std ** 2) * .02,
                "root_orientation": .5 * torch.exp(-rot_angle_sq(rr, rotation) / mdp.cfg.root_ori_std ** 2) * .02,
                "action_rate_penalty": mdp.cfg.w_action_rate * ((action - previous_action) ** 2).sum(-1) * .02,
                "soft_limit_penalty": mdp.cfg.w_joint_limit * (torch.relu(q - mdp.soft_hi) + torch.relu(mdp.soft_lo - q)).sum(-1) * .02,
            }
            terms["total"] = sum(terms.values())
            direct = mdp.reward(after, next_frames, action, previous_action)
            active = torch.tensor(trace["alive"][step], device=device)
            max_reward_formula_error = max(max_reward_formula_error, float((terms["total"][active] - direct[active]).abs().max()))
            for key, value in terms.items():
                values[key][step] = value.cpu().numpy()
            reconstructed_returns += direct.cpu().numpy() * trace["alive"][step]
            fixed = list(after)
            fixed[4], fixed[5] = q.clone(), qd.clone()
            fixed[4][:, 14:] = motion.q[next_frames, 14:]
            fixed[5][:, 14:] = motion.qd[next_frames, 14:]
            upper_fixed[step] = mdp.reward(tuple(fixed), next_frames, action, previous_action).cpu().numpy()
            previous_action = action

    return_error = float(np.max(np.abs(reconstructed_returns - trace["returns"])))
    if return_error > 5e-4 or max_reward_formula_error > 2e-6:
        raise ValueError(f"Saved-replay reward mismatch: returns={return_error}, formula={max_reward_formula_error}")
    data = mujoco.MjData(model)
    feet = [CollisionGeometry(model, (side + "_ankle_roll_link",)) for side in ("left", "right")]
    per_clip = []
    for clip, row in enumerate(rows):
        full = outcome["per_clip"][clip]
        if (full["name"], full["split"]) != (row["name"], row["split"]):
            raise ValueError("Saved clip order differs")
        count = int(trace["steps"][clip])
        frame_ids = torch.tensor(trace["frames"][:count, clip], device=device)
        reference_q = motion.q[frame_ids].cpu().numpy()
        actual_q = trace["states"][:count, clip, 13:35]
        command_q = targets[:count, clip]
        reference_soft = (torch.relu(motion.q[frame_ids] - mdp.soft_hi) + torch.relu(mdp.soft_lo - motion.q[frame_ids])).cpu().numpy()
        item = {key: full[key] for key in ("name", "split", "group", "failed", "steps", "reference_steps", "root_xy_rmse_m", "upper_limb_rmse_rad", "geometric_double_near_ground_fraction")}
        item["mean_reward_terms"] = {key: float(values[key][:count, clip].mean()) for key in names}
        item["upper_joint_target_rmse_rad"] = float(np.sqrt(np.mean((command_q[:, 14:] - reference_q[:, 14:]) ** 2)))
        item["upper_joint_command_response_rmse_rad"] = float(np.sqrt(np.mean((command_q[:, 14:] - actual_q[:, 14:]) ** 2)))
        item["upper_corrected_static_reward_delta"] = float((upper_fixed[:count, clip] - values["total"][:count, clip]).mean())
        item["nominal_pd_effort_exceed_fraction"] = (np.abs(torque_estimate[:count, clip]) > effort).mean(axis=0).tolist()
        item["joint_rmse_rad"] = np.sqrt(np.mean((actual_q - reference_q) ** 2, axis=0)).tolist()
        item["joint_bias_rad"] = (actual_q - reference_q).mean(axis=0).tolist()
        item["reference_soft_limit_mean_rad"] = reference_soft.mean(axis=0).tolist()
        item["reference_soft_limit_frame_fraction"] = (reference_soft > 0).mean(axis=0).tolist()
        item["actual_upper_range_rad"] = np.ptp(actual_q[:, 14:], axis=0).tolist()
        item["reference_upper_range_rad"] = np.ptp(reference_q[:, 14:], axis=0).tolist()
        if row["group"] == "walking" and (row["split"] == "validation" or row["name"] in ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071")):
            reference_states = torch.cat((motion.root_p[frame_ids], torch.tensor(trace["states"][:count, clip, 3:7], device=device), motion.root_v[frame_ids], motion.root_w[frame_ids], motion.q[frame_ids], motion.qd[frame_ids]), -1).cpu().numpy()
            from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
            reference_states[:, 3:7] = mat_to_quat(motion.root_R[frame_ids]).cpu().numpy()
            item["reference_geometric_support_same_prefix"] = support(model, data, feet, reference_states)
        per_clip.append(item)

    summary = {}
    for split in ("train", "validation"):
        summary[split] = {}
        for group in ("all", "walking", "gesture"):
            chosen = [row for row in per_clip if row["split"] == split and (group == "all" or row["group"] == group)]
            if not chosen:
                continue
            summary[split][group] = {
                "clips": len(chosen),
                "mean_reward_terms": {key: float(np.mean([row["mean_reward_terms"][key] for row in chosen])) for key in names},
                **{key: mean_rows(chosen, key) for key in ("upper_limb_rmse_rad", "upper_joint_target_rmse_rad", "upper_joint_command_response_rmse_rad", "upper_corrected_static_reward_delta")},
                "mean_joint_rmse_rad": np.mean([row["joint_rmse_rad"] for row in chosen], axis=0).tolist(),
                "mean_reference_soft_limit_rad": np.mean([row["reference_soft_limit_mean_rad"] for row in chosen], axis=0).tolist(),
            }
    result = {
        "status": "SAVED_TRACE_POLICY_REWARD_DIAGNOSIS_COMPLETED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "per_clip": per_clip, "joint_names": Q1_JOINT_NAMES,
        "checks": {"max_absolute_return_reconstruction_error": return_error, "max_absolute_reward_formula_error": max_reward_formula_error},
        "control": {"kp": kp.cpu().tolist(), "kd": kd.cpu().tolist(), "effort": effort.tolist(), "action_scale": alpha.cpu().tolist()},
        "physics_integration_steps": 0, "optimizer_updates": 0,
        "policy_inference_calls": int(trace["steps"].max()),
        "sources": [identity(path) for path in (args.library, checkpoint_path, scene, folder / "final.json", folder / "final_trajectory.npz", Path(__file__))],
        "limits": "Policy actions reconstructed only on saved deterministic evaluation prefixes; accumulated rewards must match recorded returns. Counterfactual upper-joint replacement measures the objective at a synthetic state, not a reachable rollout. PD values are pre-step estimates, not measured contact or actuator substep forces. Geometric near-ground is not force contact. This does not reconstruct training episode accounting or validate model action gradients.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": identity(args.output), "checks": result["checks"], "summary": summary}))


if __name__ == "__main__":
    main()
