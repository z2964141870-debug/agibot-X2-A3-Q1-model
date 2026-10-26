#!/usr/bin/env python3
"""Stream short deterministic leg/control diagnostics without changing source files."""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--entry-sha256", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "script/supermimic_for_q1"))
    import mujoco
    import numpy as np
    import torch
    from contact_protocol import ContactCorrectedAssets
    from prepare_motion import support_height
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.q1_assets import TRACKED, identity
    from supermimic.q1_sim import Q1Sim, SimCfg

    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    started = time.monotonic()
    root = args.root
    run = root / "data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025"
    evaluation_path = run / "evaluation_final/receipt.json"
    evaluation = json.loads(evaluation_path.read_text())
    row, = evaluation["rows"]
    reference = Path(row["reference"]["path"])
    trace_path = Path(row["trace"]["path"])
    frozen = [identity(path) for path in (evaluation_path, reference, trace_path)]
    if frozen[1] != row["reference"] or frozen[2] != row["trace"]:
        raise ValueError("The saved RF04 evidence has changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        saved_actions, saved_states = trace["actions"].copy(), trace["state"].copy()
    assets = ContactCorrectedAssets(root)
    if assets.manifest() != evaluation["assets"]:
        raise ValueError("The RF04 asset/control baseline differs")
    baseline_assets = assets.manifest()
    legs = np.array([any(s in name for s in ("hip", "knee", "ankle")) for name in assets.names])
    data = mujoco.MjData(assets.model)
    with np.load(reference, allow_pickle=False) as ref:
        reference_qpos = ref["qpos"].copy()
    heights = []
    for q in reference_qpos:
        data.qpos[:] = q
        mujoco.mj_fwdPosition(assets.model, data)
        heights.append(support_height(assets.model, data, assets.foot_geoms))
    median = float(np.median(heights))
    trials = [("saved_actions_cap25", "saved", .25, 0.),
              ("saved_actions_cap50", "saved", .50, 0.),
              ("reference_pd_cap25", "reference", .25, 0.),
              ("reference_pd_cap50", "reference", .50, 0.),
              ("reference_pd_cap25_lowered", "reference", .25, -median),
              ("reference_pd_cap50_lowered", "reference", .50, -median)]
    results, arrays = [], {}
    for label, mode, fraction, dz in trials:
        trial_assets = ContactCorrectedAssets(root)
        trial_assets.effort *= fraction / .25
        library = MotionLibrary([reference], TRACKED, trial_assets.q0, device="cpu",
                                active_indices=trial_assets.active_indices)
        library._ref[:, 2] += dz
        library.body_pos_w[:, :, 2] += dz
        library.goal_feat[:, -1] += dz
        sim = Q1Sim(trial_assets, library, SimCfg(num_envs=1, threads=1,
                    domain_randomization=False, observation_noise=False, action_noise=0.,
                    episode_steps=library.num_frames + 1), trial_assets.heads(), seed=0)
        sim.reset(sim.ids, torch.tensor([0]))
        initial = sim.state[0].numpy().copy()
        model, probe = sim.models[0], mujoco.MjData(sim.models[0])
        floor = model.geom("floor").id
        side = {int(g): int(model.body(int(model.geom_bodyid[g])).name.startswith("right"))
                for g in trial_assets.foot_geoms}
        states, forces, saturated, rewards, errors = [], [], [], [], []
        early = False
        limit = len(saved_actions) if mode == "saved" else library.num_frames
        try:
            for index in range(limit):
                if mode == "saved":
                    action = torch.as_tensor(saved_actions[index:index + 1])
                else:
                    target = library.joint_pos[index].numpy()
                    action = torch.as_tensor(((target - trial_assets.q0) / trial_assets.scale)[None],
                                             dtype=torch.float32)
                previous = sim.state[0].numpy().copy()
                target = trial_assets.q0 + action[0].clamp(-20., 20.).numpy() * trial_assets.scale
                demand = trial_assets.kp * (target - previous[trial_assets.qadr]) - (
                    trial_assets.kd * previous[29 + trial_assets.vadr])
                result = sim.step(action, auto_reset=False)
                state = result["state"][0].numpy().copy()
                probe.qpos[:], probe.qvel[:] = state[:29], state[29:]
                probe.ctrl[trial_assets.aids] = target
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
                forces.append(support)
                saturated.append(np.abs(demand) / trial_assets.effort)
                rewards.append(float(result["reward"][0]))
                errors.append(state[7:29] - library.joint_pos[index].numpy())
                early = bool(result["terminated"][0])
                if early:
                    break
        finally:
            sim.close()
        states, forces, saturated, errors = map(np.asarray, (states, forces, saturated, errors))
        supported = forces > 1.
        actual_root = states[:, :3]
        reference_root = library.root_pos_w[:len(states)].numpy()
        result = {"trial": label, "mode": mode, "torque_fraction_of_model": fraction,
                  "reference_constant_z_shift_m": dz, "steps": len(states),
                  "reference_frames": library.num_frames, "early_terminated": early,
                  "completion": len(states) / library.num_frames,
                  "leg_rmse_rad": float(np.sqrt(np.mean(errors[:, legs] ** 2))),
                  "root_xy_error_m": float(np.linalg.norm(actual_root[:, :2] - reference_root[:, :2], axis=1).mean()),
                  "root_xy_displacement_m": float(np.linalg.norm(actual_root[-1, :2] - initial[:2])),
                  "final_root_height_m": float(actual_root[-1, 2]),
                  "normalized_return": sum(rewards) / (library.num_frames * sim.heads.reward_cfg.max_step_reward),
                  "leg_boundary_saturation_fraction": float((saturated[:, legs] > 1. + 1e-10).mean()),
                  "leg_demand_to_cap_p95": float(np.quantile(saturated[:, legs], .95)),
                  "double_support_boundary_fraction": float(supported.all(axis=1).mean()),
                  "no_support_boundary_fraction": float((~supported.any(axis=1)).mean()),
                  "left_support_boundary_fraction": float(supported[:, 0].mean()),
                  "right_support_boundary_fraction": float(supported[:, 1].mean()),
                  "support_state_changes": int(np.any(supported[1:] != supported[:-1], axis=1).sum()),
                  "joint_saturation_fraction": {name: float((saturated[:, j] > 1. + 1e-10).mean())
                                                for j, name in enumerate(trial_assets.names)}}
        if label == "saved_actions_cap25":
            result["saved_state_max_abs_error"] = float(np.abs(states - saved_states[:len(states)]).max())
            if len(states) != len(saved_states) or result["saved_state_max_abs_error"] > 2e-5:
                raise ValueError("The native baseline replay does not match the preserved RF04 trace")
        arrays[label] = {"state": states, "reference_root": reference_root, "initial_state": initial,
                         "post_boundary_normal_force_n": forces, "pre_boundary_demand_to_cap": saturated,
                         "joint_error_rad": errors, "rewards": np.asarray(rewards)}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "joint_saturation_fraction"}),
              file=sys.stderr, flush=True)
    prefix = min(row["steps"] for row in results)
    for row in results:
        data = arrays[row["trial"]]
        row["common_prefix_steps"] = prefix
        row["common_prefix_leg_rmse_rad"] = float(np.sqrt(np.mean(data["joint_error_rad"][:prefix, legs] ** 2)))
        row["common_prefix_root_xy_error_m"] = float(np.linalg.norm(
            data["state"][:prefix, :2] - data["reference_root"][:prefix, :2], axis=1).mean())
    if frozen != [identity(record["path"]) for record in frozen]:
        raise ValueError("Preserved source evidence changed during the probe")
    receipt = {"status": "Q1_LEG_CONTROL_DIAGNOSTICS_COMPLETED", "entry_sha256": args.entry_sha256,
               "inputs": frozen, "baseline_assets": baseline_assets, "trials": results,
               "reference_lower_sole_median_m": median, "common_prefix_steps": prefix,
               "control_steps": sum(row["steps"] for row in results), "substeps_per_control": 20,
               "optimizer_updates": 0, "cuda_used": False, "source_files_changed": 0,
               "elapsed_seconds": time.monotonic() - started, "hostname": os.uname().nodename,
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "limits": "Short CPU diagnostics, not new training or a feasibility/data gate. Reference PD has no balance policy or inverse-dynamics feedforward. Saved actions in cap50 are open-loop, not a recomputed policy. Contact forces are reconstructed at post-control boundaries, not all substeps. Lowered variants use one constant source-reference median shift only in memory. Trial torque caps are simulation probes, not verified Q1 hardware settings.",
               "storage_status": "LOCAL_ONLY"}
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
        for name, payload in [("receipt.json", (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())] + [
                (label + ".npz", values) for label, values in arrays.items()]:
            if isinstance(payload, dict):
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **payload)
                payload = buffer.getvalue()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(payload), 0o644
            archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    main()
