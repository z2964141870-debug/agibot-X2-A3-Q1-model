"""Measure frozen-policy contact forces and matched-state model derivatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from analyze_q1_hp_trial import identity


def agreement(predicted, physical):
    predicted, physical = np.asarray(predicted), np.asarray(physical)
    valid = (np.abs(predicted) > 1e-7) & (np.abs(physical) > 1e-7)
    denominator = np.linalg.norm(predicted) * np.linalg.norm(physical)
    return {
        "cosine": float(predicted @ physical / denominator) if denominator > 1e-15 else None,
        "sign_fraction": float(np.mean(np.sign(predicted[valid]) == np.sign(physical[valid]))) if valid.any() else None,
        "informative_axes": int(valid.sum()),
        "model_norm": float(np.linalg.norm(predicted)),
        "physical_norm": float(np.linalg.norm(physical)),
        "model": predicted.tolist(), "physical": physical.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "library", "suite", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp

    torch.set_num_threads(8)
    torch.manual_seed(0)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from unisupermimic.rl.algos.supermimic.geometry import flatten_state, mat_to_quat, quat_to_mat, unflatten_state
    from unisupermimic.rl.algos.supermimic import networks
    from unisupermimic.rl.algos.supermimic.networks import WorldModel
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, default_joint_pos_list
    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy

    device = torch.device("cuda:0")
    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"] + library["splits"]["validation"]
    names = [c["name"] for c in json.loads(args.suite.read_text())["clips"][:4]]
    selected = [next(i for i, row in enumerate(rows) if row["name"] == name) for name in names]
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", device)
    motion = MotionLib([rows[i]["path"] for i in selected], kin, device, src_fps=30.)
    default = torch.tensor(default_joint_pos_list(), device=device)
    alpha = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=device)
    mdp = MDP(kin, motion, default, observation_mode="root_xy")
    checkpoint = torch.load(args.trial / "run/final.pt", map_location=device, weights_only=True)
    policy = load_policy(SimpleNamespace(supermimic_mdp=mdp, device=device), checkpoint)
    for p in policy.parameters():
        p.requires_grad_(False)
    wm = WorldModel(hidden=tuple(checkpoint["args"]["wm_hidden"]), num_joints=22, context_dim=26).to(device)
    wm.load_state_dict(checkpoint["wm"])
    for p in wm.parameters():
        p.requires_grad_(False)
    model = mujoco.MjModel.from_xml_path(str(args.runtime / "q1_assets/scene.xml"))
    model.opt.timestep = .005
    gpu_model = mw.put_model(model)
    cuda = "cuda:0"
    worlds = len(selected)
    gpu_data = mw.put_data(model, mujoco.MjData(model), nworld=worlds, nconmax=128, njmax=500)
    context = torch.zeros(worlds, 26, device=device)
    context[:, 0] = 1.
    p, rotation, v, w, q, qd = motion.ref_state(motion.clip_start)
    initial = flatten_state((p, rotation, v, w, q, qd))

    def set_state(data, flat):
        p, r, v, w, q, qd = unflatten_state(flat)
        wp.to_torch(data.qpos).copy_(torch.cat((p, mat_to_quat(r), q), -1))
        wp.to_torch(data.qvel).copy_(torch.cat((v, (r.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1), qd), -1))
        wp.to_torch(data.qacc_warmstart).zero_()

    def state_of(data):
        qpos, qvel = wp.to_torch(data.qpos), wp.to_torch(data.qvel)
        r = quat_to_mat(qpos[:, 3:7])
        return (qpos[:, :3], r, qvel[:, :3], (r @ qvel[:, 3:6, None]).squeeze(-1), qpos[:, 7:], qvel[:, 6:])

    set_state(gpu_data, initial)
    mw.forward(gpu_model, gpu_data)
    foot_bodies = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    body_lookup = {body: kin.body_names.index(model.body(body).name) for body in foot_bodies}
    geom_body = torch.tensor(model.geom_bodyid, device=device)
    contact_ids = wp.array(np.arange(gpu_data.naconmax, dtype=np.int32), dtype=int, device=cuda)
    forces = wp.zeros(gpu_data.naconmax, dtype=wp.spatial_vector, device=cuda)
    history = None
    previous = torch.zeros(worlds, 22, device=device)
    alive = torch.ones(worlds, dtype=torch.bool, device=device)
    physical_states, actions, alive_rows, normal_rows, slip_rows = [], [], [], [], []
    max_steps = int((motion.clip_len - 1).max())
    with torch.inference_mode():
        for step in range(max_steps):
            frames = torch.minimum(motion.clip_start + step, motion.clip_end)
            state = state_of(gpu_data)
            observation = mdp.obs(state, previous, context, frames)
            history = observation[:, None].repeat(1, checkpoint["args"]["hist"], 1) if history is None else torch.cat((history[:, 1:], observation[:, None]), 1)
            action = policy(history, mdp.goal(frames, checkpoint["args"]["goal_frames"]))
            wp.to_torch(gpu_data.ctrl).copy_(default + alpha * action)
            for _ in range(4):
                mw.step(gpu_model, gpu_data)
            # Recompute forces at the final state, without another integration.
            mw.forward(gpu_model, gpu_data)
            after = state_of(gpu_data)
            forces.zero_()
            mw.contact_force(gpu_model, gpu_data, contact_ids, True, forces)
            count = int(wp.to_torch(gpu_data.nacon)[0])
            world = wp.to_torch(gpu_data.contact.worldid)[:count].long()
            geoms = wp.to_torch(gpu_data.contact.geom)[:count].long()
            points = wp.to_torch(gpu_data.contact.pos)[:count]
            contact_force = wp.to_torch(forces)[:count, :3]
            bp, _, bv, bw = kin.forward(*after, all_bodies=True)
            normal, slip_sum = torch.zeros(worlds, 2, device=device), torch.zeros(worlds, 2, device=device)
            for side, foot_body in enumerate(foot_bodies):
                bodies = geom_body[geoms]
                mask = ((bodies[:, 0] == foot_body) & (bodies[:, 1] == 0)) | ((bodies[:, 1] == foot_body) & (bodies[:, 0] == 0))
                wi, cp, force = world[mask], points[mask], contact_force[mask]
                body = int(body_lookup[foot_body])
                speed = (bv[wi, body] + torch.linalg.cross(bw[wi, body], cp - bp[wi, body]))[:, :2].norm(dim=-1)
                vertical = force[:, 2].abs()
                normal[:, side].index_add_(0, wi, vertical)
                slip_sum[:, side].index_add_(0, wi, vertical * speed)
            slip = slip_sum / normal.clamp(min=1e-6)
            physical_states.append(flatten_state(after).clone())
            actions.append(action.clone())
            alive_rows.append(alive.clone())
            normal_rows.append(normal)
            slip_rows.append(slip)
            alive &= ~(mdp.termination(after, torch.minimum(frames+1, motion.clip_end)) | (frames+1 >= motion.clip_end))
            previous = action
            if not bool(alive.any()):
                break
    states = torch.stack(physical_states)
    all_actions = torch.stack(actions)
    active = torch.stack(alive_rows)
    normal, slip = torch.stack(normal_rows), torch.stack(slip_rows)
    with np.load(args.trial / "evaluation/final_trajectory.npz", allow_pickle=False) as saved:
        original_states = saved["states"][:, selected]
        original_steps = saved["steps"][selected]
    contacts = []
    picks = []
    for i, name in enumerate(names):
        count = int(active[:, i].sum())
        force = normal[:count, i].cpu().numpy()
        speed = slip[:count, i].cpu().numpy()
        stance = force > 5.
        codes = stance[:, 0].astype(int) + 2*stance[:, 1].astype(int)
        single = codes[(codes == 1) | (codes == 2)]
        shared = min(count, int(original_steps[i]))
        contacts.append({"name": name, "steps": count, "reference_steps": int(motion.clip_len[i])-1,
            "saved_trace_common_state_max_abs": float(np.max(np.abs(states[:shared, i].cpu().numpy()-original_states[:shared, i]))),
            "mean_vertical_force_n": force.mean(0).tolist(), "double_support_fraction": float((codes == 3).mean()),
            "single_support_fraction": float(((codes == 1) | (codes == 2)).mean()), "flight_fraction": float((codes == 0).mean()),
            "single_support_side_changes": int(np.count_nonzero(np.diff(single))),
            "loaded_contact_slip_mean_mps": float(speed[stance].mean()) if stance.any() else None,
            "loaded_contact_slip_p95_mps": float(np.quantile(speed[stance], .95)) if stance.any() else None})
        for fraction in ((.1, .35, .6, .85) if i < 2 else (.25, .7)):
            k = min(int((count-9)*fraction), count-9)
            if k >= 1:
                picks.append((i, k))

    # Matched cold starts isolate action response; no history or warm start is
    # borrowed from another perturbation branch. Future actions stay fixed.
    horizons = (1, 4, 8)
    del gpu_data
    epsilons = (.02, .08)
    offsets = [torch.zeros(22, device=device)]
    descriptors = [(0., -1, 0)]
    for eps in epsilons:
        for j in range(22):
            for sign in (-1, 1):
                offset = torch.zeros(22, device=device)
                offset[j] = sign*eps
                offsets.append(offset)
                descriptors.append((eps, j, sign))
    variants = len(offsets)
    offsets = torch.stack(offsets)
    starts = torch.stack([states[k-1, i] for i, k in picks])
    branch = starts[:, None].expand(-1, variants, -1).reshape(-1, 57)
    data = mw.put_data(model, mujoco.MjData(model), nworld=len(branch), nconmax=128, njmax=500)
    set_state(data, branch)
    mw.forward(gpu_model, data)
    finite_values = {}

    def metrics(state, frames, action, prev):
        reward = mdp.reward(state, frames, action, prev)
        tracking = reward - mdp.cfg.w_action_rate*((action-prev)**2).sum(-1)*.02 - mdp.cfg.w_joint_limit*(torch.relu(state[4]-mdp.soft_hi)+torch.relu(mdp.soft_lo-state[4])).sum(-1)*.02
        return {"tracking_reward": tracking, "upper_score": -((state[4][:, 14:]-motion.q[frames, 14:])**2).mean(-1), "leg_score": -((state[4][:, :12]-motion.q[frames, :12])**2).mean(-1)}

    for h in range(1, max(horizons)+1):
        base_action = torch.stack([all_actions[k+h-1, i] for i, k in picks])
        action = (base_action[:, None] + (offsets[None] if h == 1 else 0)).expand(-1, variants, -1).reshape(-1, 22)
        wp.to_torch(data.ctrl).copy_(default+alpha*action)
        for _ in range(4):
            mw.step(gpu_model, data)
        if h in horizons:
            frames = torch.tensor([int(motion.clip_start[i])+k+h for i, k in picks], device=device).repeat_interleave(variants)
            prev = torch.stack([all_actions[k+h-2, i] for i, k in picks]).repeat_interleave(variants, 0)
            with torch.no_grad():
                finite_values[h] = {key: value.reshape(len(picks), variants).cpu().numpy() for key, value in metrics(state_of(data), frames, action, prev).items()}
    comparisons = []
    for sample, (i, k) in enumerate(picks):
        first_action = all_actions[k, i:i+1].detach().clone().requires_grad_()
        current = unflatten_state(starts[sample:sample+1])
        xi = context[i:i+1]
        for h in range(1, max(horizons)+1):
            action = first_action if h == 1 else all_actions[k+h-1, i:i+1]
            current = wm.step(current, action, mdp.normalize_context(xi))
            if h not in horizons:
                continue
            frames = motion.clip_start[i:i+1]+k+h
            prev = all_actions[k+h-2, i:i+1]
            model_metrics = metrics(current, frames, action, prev)
            for key, score in model_metrics.items():
                derivative = torch.autograd.grad(score.sum(), first_action, retain_graph=True)[0][0].cpu().numpy()
                for eps_index, eps in enumerate(epsilons):
                    base = 1+eps_index*44
                    physical = (finite_values[h][key][sample, base+1:base+44:2]-finite_values[h][key][sample, base:base+44:2])/(2*eps)
                    comparisons.append({"clip": names[i], "step": k, "horizon": h, "epsilon": eps, "metric": key,
                        "contact_vertical_force_n": normal[k-1, i].cpu().tolist(), "all_axes": agreement(derivative, physical),
                        "upper_axes": agreement(derivative[14:], physical[14:]), "leg_axes": agreement(derivative[:12], physical[:12])})
    summary = {}
    for key in ("tracking_reward", "upper_score", "leg_score"):
        for h in horizons:
            group = [c for c in comparisons if c["metric"] == key and c["horizon"] == h and c["epsilon"] == .08]
            summary[f"{key}_H{h}"] = {"samples": len(group), "mean_cosine": float(np.mean([c["all_axes"]["cosine"] for c in group if c["all_axes"]["cosine"] is not None])), "mean_sign_fraction": float(np.mean([c["all_axes"]["sign_fraction"] for c in group if c["all_axes"]["sign_fraction"] is not None]))}
    reference_action = (motion.q-default)/alpha
    reset_cost = -.1*(reference_action**2).sum(-1)*.02
    result = {"status": "FROZEN_POLICY_PHYSICAL_AND_DIRECTION_DIAGNOSIS_COMPLETED", "optimizer_steps": 0,
        "scope": "Standalone Q1 Warp using XML friction defaults, asset PD and frozen policy; not the native Manager backend. Actual solved ground-contact forces are refreshed without integration after each 20ms control step. Branch finite differences start from matched state with cleared acceleration warm start and fixed future actions. Network AMP inherits the checkpoint through load_policy; this is not an H32 closed-loop policy/critic derivative test.",
        "model_arithmetic": {"state_dtype": str(starts.dtype), "mlp_autocast_dtype": str(networks.AMP_DTYPE)},
        "standalone_geom_friction_unique": np.unique(model.geom_friction, axis=0).tolist(),
        "sources": [identity(p) for p in (Path(__file__), args.library, args.suite, args.trial/"run/final.pt", args.runtime/"q1_assets/scene.xml")],
        "force_threshold_n": 5., "contact": contacts, "direction_summary": summary, "direction_comparisons": comparisons,
        "reference_reset_action_rate_penalty": {"mean": float(reset_cost.mean()), "p05": float(torch.quantile(reset_cost, .05)), "fraction_penalty_exceeds_max_tracking": float((reset_cost < -.1).float().mean()), "max_tracking_per_step": .1},
        "physics_integrated_control_steps": len(physical_states)*worlds + len(branch)*max(horizons),
        "limits": "Local fixed-action probes do not prove H32 closed-loop training gradients or global causality; no training or candidate performance claim."}
    (args.output/"result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(args.output/"contact_trace.npz", states=states.cpu().numpy(), actions=all_actions.cpu().numpy(), alive=active.cpu().numpy(), vertical_force_n=normal.cpu().numpy(), slip_mps=slip.cpu().numpy())
    print(json.dumps({key: result[key] for key in ("status", "contact", "direction_summary", "reference_reset_action_rate_penalty")}), flush=True)


if __name__ == "__main__":
    main()
