#!/usr/bin/env python3
"""Compare three torque-cap interventions with a fixed RF04 closed-loop actor."""

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = args.output.resolve()
    if not output.is_relative_to(root / "data/supermimic_for_q1") or output.exists():
        parser.error("Use a new independent Q1 diagnostic output")
    for directory in ("supermimic_for_q1", "supermimic_q1_reproduction",
                      "supermimic_q1_stability", "supermimic_q1_analysis",
                      "supermimic_q1_reproduction/leg_control"):
        sys.path.insert(0, str(root / "script" / directory))
    import mujoco
    import numpy as np
    import torch
    from bounded_policy import BoundedPolicy
    from contact_protocol import ContactCorrectedAssets
    from control import Leg50Assets, leg_mask
    from diagnose_fullbody import kinematic_metrics
    from evaluate_reproduction import summarize_trace
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.policy.nets import PolicyCfg
    from supermimic.q1_assets import TRACKED, identity
    from supermimic.q1_sim import Q1Sim, SimCfg

    def health():
        base = Path("/sys/fs/cgroup/memory")
        memory = int((base / "memory.usage_in_bytes").read_text())
        oom = dict(line.split() for line in (base / "memory.oom_control").read_text().splitlines())
        disk = os.statvfs(root)
        free = disk.f_bavail * disk.f_frsize
        if (memory >= 30 * 1024 ** 3 or int(oom["oom_kill"]) != args.expected_oom_kill
                or int(oom["under_oom"]) or free < 100 * 1024 ** 3):
            raise RuntimeError("CPU diagnostic resource protection")
        return {"memory_bytes": memory, "oom_kill": int(oom["oom_kill"]),
                "available_disk_bytes": free}

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    started = time.monotonic()
    initial_health = health()
    run = root / "data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025"
    evaluation_path = run / "evaluation_final/receipt.json"
    checkpoint_path = run / "run/final.pt"
    evaluation = json.loads(evaluation_path.read_text())
    row, = evaluation["rows"]
    reference_path = Path(row["reference"]["path"])
    saved_trace_path = Path(row["trace"]["path"])
    inputs = [identity(path) for path in (evaluation_path, checkpoint_path,
                                         reference_path, saved_trace_path)]
    if inputs[1] != evaluation["checkpoint"] or inputs[2:] != [row["reference"], row["trace"]]:
        raise ValueError("The preserved RF04 checkpoint/reference/trace identities changed")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = BoundedPolicy(PolicyCfg(**saved["manifest"]["policy"])).eval()
    policy.load_state_dict(saved["policy"], strict=True)
    del saved
    with np.load(saved_trace_path, allow_pickle=False) as trace:
        saved_states = trace["state"].copy()
        saved_actions = trace["actions"].copy()
        saved_initial = trace["initial_state"].copy()
    with np.load(reference_path, allow_pickle=False) as reference:
        ref_qpos = reference["qpos"].copy()
        ref_body = {"pos": reference["body_pos_w"].copy(),
                    "quat": reference["body_quat_w"].copy()}
    baseline = ContactCorrectedAssets(root)
    if baseline.manifest() != evaluation["assets"]:
        raise ValueError("The original RF04 control manifest differs")
    legs = leg_mask(baseline.names)
    output.mkdir()
    rows, arrays = [], {}
    with torch.inference_mode():
        for label in ("baseline25", "legs50_other25", "all50"):
            health()
            assets = Leg50Assets(root) if label == "legs50_other25" else ContactCorrectedAssets(root)
            if label == "all50":
                assets.effort *= 2.
            if (not np.array_equal(assets.scale, baseline.scale)
                    or not np.array_equal(assets.kp, baseline.kp)
                    or not np.array_equal(assets.kd, baseline.kd)):
                raise ValueError("The intervention also changed action scales or PD")
            library = MotionLibrary([reference_path], TRACKED, assets.q0, device="cpu",
                                    active_indices=assets.active_indices)
            heads = assets.heads()
            sim = Q1Sim(assets, library, SimCfg(num_envs=1, threads=1,
                        domain_randomization=False, observation_noise=False,
                        action_noise=0., episode_steps=library.num_frames + 1), heads, seed=0)
            sim.reset(sim.ids, torch.tensor([0]))
            initial = sim.state[0].numpy().copy()
            if not np.array_equal(initial, saved_initial):
                raise ValueError("The intervention does not start at the preserved RF04 state")
            model = sim.models[0]
            probe = mujoco.MjData(model)
            floor = model.geom("floor").id
            side = {int(g): int(model.body(int(model.geom_bodyid[g])).name.startswith("right"))
                    for g in assets.foot_geoms}
            states, actions, rewards, joints, bodies, terminated, forces, ratios = ([] for _ in range(8))
            try:
                for index in range(library.num_frames):
                    observation, goal = sim.policy_inputs()
                    action = policy(observation, goal)
                    target = assets.q0 + action[0].clamp(-20., 20.).numpy() * assets.scale
                    previous = sim.state[0].numpy().copy()
                    demand = assets.kp * (target - previous[assets.qadr]) - (
                        assets.kd * previous[29 + assets.vadr])
                    result = sim.step(action, auto_reset=False)
                    state = result["state"][0].numpy().copy()
                    probe.qpos[:], probe.qvel[:] = state[:29], state[29:]
                    probe.ctrl[assets.aids] = target
                    mujoco.mj_forward(model, probe)
                    support = np.zeros(2)
                    for contact_index, contact in enumerate(probe.contact):
                        other = int(contact.geom2) if contact.geom1 == floor else (
                            int(contact.geom1) if contact.geom2 == floor else -1)
                        if other in side:
                            wrench = np.empty(6)
                            mujoco.mj_contactForce(model, probe, contact_index, wrench)
                            support[side[other]] += max(0., float(wrench[0]))
                    states.append(state)
                    actions.append(action[0].numpy().copy())
                    rewards.append(float(result["reward"][0]))
                    joints.append(float(result["joint_rmse"][0]))
                    bodies.append(float(result["body_error_m"][0]))
                    terminated.append(bool(result["terminated"][0]))
                    forces.append(support)
                    ratios.append(np.abs(demand) / assets.effort)
                    if terminated[-1]:
                        break
            finally:
                sim.close()
            states, actions, forces, ratios = map(np.asarray, (states, actions, forces, ratios))
            if not all(np.isfinite(value).all() for value in (states, actions, forces, ratios)):
                raise ValueError("The intervention generated nonfinite physical values")
            supported = forces > 1.
            reference_root = library.root_pos_w.numpy()
            direction = reference_root[-1, :2] - reference_root[0, :2]
            unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
            displacement = states[-1, :2] - initial[:2]
            actual_bodies = {k: v.numpy() for k, v in heads.robot_bodies(torch.from_numpy(states)).items()}
            fractions = assets.effort / assets.model.actuator_ctrlrange[assets.aids, 1]
            result = {"trial": label,
                **summarize_trace(states, rewards, joints, bodies, terminated, reference_root,
                                  initial[:3], heads.reward_cfg.max_step_reward),
                "torque_fractions_by_joint": fractions.tolist(),
                "effective_effort_nm": assets.effort.tolist(),
                "action_scale": assets.scale.tolist(), "pd_unchanged": True,
                "root_forward_displacement_m": float(displacement @ unit),
                "root_mean_forward_velocity_m_s": float(displacement @ unit) / (len(states) * .02),
                "final_root_height_m": float(states[-1, 2]),
                "leg_boundary_saturation_fraction": float((ratios[:, legs] > 1. + 1e-10).mean()),
                "joint_saturation_fraction": {name: float((ratios[:, j] > 1. + 1e-10).mean())
                                              for j, name in enumerate(assets.names)},
                "double_support_boundary_fraction": float(supported.all(axis=1).mean()),
                "no_support_boundary_fraction": float((~supported.any(axis=1)).mean()),
                "left_only_support_boundary_fraction": float((supported[:, 0] & ~supported[:, 1]).mean()),
                "right_only_support_boundary_fraction": float((supported[:, 1] & ~supported[:, 0]).mean()),
                "support_state_changes": int(np.any(supported[1:] != supported[:-1], axis=1).sum()),
                "executed_kinematics": kinematic_metrics(states, ref_qpos, actual_bodies, ref_body, assets.names)}
            if label == "baseline25":
                result["saved_state_max_abs_error"] = float(np.max(np.abs(states - saved_states))) if states.shape == saved_states.shape else None
                result["saved_action_max_abs_error"] = float(np.max(np.abs(actions - saved_actions))) if actions.shape == saved_actions.shape else None
                if (result["saved_state_max_abs_error"] is None or result["saved_state_max_abs_error"] > 2e-5
                        or result["saved_action_max_abs_error"] is None or result["saved_action_max_abs_error"] > 2e-5):
                    raise ValueError("The original closed-loop actor does not reproduce preserved RF04 evidence")
            trace_path = output / (label + ".npz")
            payload = {"state": states, "actions": actions, "reward": np.asarray(rewards),
                       "terminated": np.asarray(terminated), "initial_state": initial,
                       "reference_root": reference_root, "post_boundary_normal_force_n": forces,
                       "pre_boundary_demand_to_cap": ratios}
            np.savez_compressed(trace_path, **payload)
            result["trace"] = identity(trace_path)
            arrays[label] = {"state": states, "body": actual_bodies}
            rows.append(result)
            print(json.dumps({key: result[key] for key in ("trial", "executed_steps", "failed",
                "normalized_return", "root_xy_displacement_m", "root_forward_displacement_m",
                "leg_boundary_saturation_fraction", "double_support_boundary_fraction")}), flush=True)
    prefix = min(row["executed_steps"] for row in rows)
    for row in rows:
        data = arrays[row["trial"]]
        row["common_prefix_kinematics"] = kinematic_metrics(data["state"][:prefix], ref_qpos,
            {key: value[:prefix] for key, value in data["body"].items()}, ref_body, baseline.names)
    if inputs != [identity(record["path"]) for record in inputs]:
        raise ValueError("Preserved RF04 input evidence changed during the closed-loop probe")
    receipt = {"status": "Q1_LEG_CLOSED_LOOP_DIAGNOSTICS_COMPLETED", "source": identity(__file__),
        "control_source": identity(root / "script/supermimic_q1_reproduction/leg_control/control.py"),
        "inputs": inputs, "baseline_assets": baseline.manifest(), "trials": rows,
        "common_prefix_steps": prefix, "control_steps": sum(row["executed_steps"] for row in rows),
        "substeps_per_control": 20, "optimizer_updates": 0, "cuda_used": torch.cuda.is_initialized(),
        "elapsed_seconds": time.monotonic() - started, "initial_health": initial_health,
        "final_health": health(), "hostname": os.uname().nodename,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "protocol": "Fixed RF04 trained policy; recomputed action at every control; original full reference and frame-zero state; no noise or DR; stop at physical termination or 365 controls; only actuator force caps differ",
        "limits": "Simulation intervention on a policy trained under baseline caps; not retraining, hardware validation, or a data gate. Boundary contact forces and torque demand do not cover every physics substep. Executed errors have different horizons; common-prefix errors are also reported.",
        "storage_status": "LOCAL_ONLY"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "common_prefix_steps": prefix,
                      "control_steps": receipt["control_steps"], "elapsed_seconds": receipt["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
