#!/usr/bin/env python3
"""Bounded RF02 contact replay and local open-loop reward/action diagnostics."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
from diagnose_fullbody import checked, write_json

HORIZONS = (1, 8, 32)
MAGNITUDES = (.02, .10)
GAMMA = .99


def health(root, baseline_oom=14):
    memory = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
    values = dict(line.split() for line in Path("/sys/fs/cgroup/memory/memory.oom_control").read_text().splitlines())
    sample = {"memory_bytes": memory, "disk_free_bytes": shutil.disk_usage(root).free,
              "oom_kill": int(values["oom_kill"]), "under_oom": int(values["under_oom"])}
    if (memory > 30 * 1024**3 - 896 * 1024**2 or sample["disk_free_bytes"] < 100 * 1024**3
            or sample["oom_kill"] != baseline_oom or sample["under_oom"]):
        raise RuntimeError("Resource guard refused: " + json.dumps(sample))
    return sample


def prepare(args):
    from supermimic.q1_assets import identity
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("A completed physical evaluation is required")
    checked(evaluation["checkpoint"])
    for row in evaluation["rows"]:
        checked(row["trace"])
        checked(row["reference"])
    chosen = sorted(evaluation["rows"], key=lambda r: (-r["reference_root_xy_displacement_full_m"], r["clip"]))[:3]
    rng = np.random.default_rng(20261025)
    samples = []
    for row in chosen:
        for fraction in (.25, .65):
            start = int(np.floor(fraction * (row["executed_steps"] - 32)))
            if start < 1:
                raise ValueError("Insufficient saved trace for this registered state")
            direction = rng.normal(size=22)
            direction /= np.linalg.norm(direction)
            samples.append({"clip": row["clip"], "start": start, "fraction": fraction,
                            "reference": row["reference"], "trace": row["trace"],
                            "random_direction": direction.tolist()})
    plan = {"status": "PREPARED_NOT_EXECUTED", "source": identity(__file__),
            "helper": identity(HERE / "diagnose_fullbody.py"), "evaluation": identity(args.evaluation),
            "checkpoint": evaluation["checkpoint"], "root": str(args.root.resolve()),
            "samples": samples, "contact_rows": evaluation["rows"],
            "max_probe_control_steps": len(samples) * 17 * 32,
            "max_contact_control_steps": sum(r["executed_steps"] for r in evaluation["rows"]),
            "horizons": HORIZONS, "magnitudes": MAGNITUDES, "discount": GAMMA,
            "sampling": "3 fixed-eight final train clips with greatest reference net xy displacement, two interior fractions .25/.65; no model or perturbation results used",
            "protocol": "Only first saved action perturbed; later actions fixed; 3 normalized model raw-discounted-reward gradient directions H1/8/32 plus seeded random; +/- .02/.10; native baseline must reproduce saved 32 steps first",
            "limits": "Local open-loop raw reward sensitivities, no termination survival weighting, critic bootstrap or policy feedback; not full actor-gradient correctness. Contacts are last solver-substep dynamic forces, not reference support labels.",
            "initial_health": health(args.root), "new_optimizer_updates": 0}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    print(json.dumps({"plan": identity(args.output / "plan.json"), "samples": samples,
                      "probe_controls": plan["max_probe_control_steps"]}, indent=2), flush=True)


def initialize_sim(assets, reference, heads):
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.q1_assets import TRACKED
    from supermimic.q1_sim import Q1Sim, SimCfg
    lib = MotionLibrary([reference], TRACKED, assets.q0, device="cpu", active_indices=assets.active_indices)
    sim = Q1Sim(assets, lib, SimCfg(num_envs=1, threads=1, domain_randomization=False,
                observation_noise=False, action_noise=0., episode_steps=lib.num_frames + 1), heads, seed=0)
    return sim


def native(sim, state, previous, actions, start, counter):
    import torch
    sim.reset(sim.ids, torch.tensor([start]))
    sim.state.copy_(state)
    sim.previous_action.copy_(previous)
    states, rewards, terms, failures = [], [], [], []
    for action in actions:
        if counter[0] >= counter[1]:
            raise RuntimeError("Registered control budget exhausted")
        result = sim.step(action[None], auto_reset=False)
        counter[0] += 1
        states.append(result["state"][0].numpy().copy())
        rewards.append(float(result["reward"][0]))
        terms.append({key: float(value[0]) for key, value in result["terms"].items()})
        failures.append(bool(result["terminated"][0] or result["timeout"][0]))
        if failures[-1]:
            break
    return np.asarray(states), np.asarray(rewards), terms, np.asarray(failures)


def model_rewards(model, state, previous, actions, references, delta):
    import torch
    from supermimic.robot_spec import XI_DIM
    from supermimic.world_model.heads import ReferenceFrame
    batch = delta.shape[0]
    state = state.expand(batch, -1)
    previous = previous.expand(batch, -1)
    xi = torch.zeros(batch, XI_DIM, device=state.device, dtype=state.dtype)
    xi[:, 0] = 1.
    rewards, positions = [], []
    for t in range(32):
        action = actions[t:t + 1].expand(batch, -1)
        if t == 0:
            action = action + delta
        action = action.clamp(-20, 20)
        state = model.step(state, action, xi)
        ref = ReferenceFrame(**{key: value[t:t + 1].expand(batch, *value.shape[1:])
                               for key, value in vars(references).items()})
        reward = model.heads.reward(state, action, previous, ref)["total"]
        previous = action
        rewards.append(reward)
        positions.append(state[:, :3])
    return torch.stack(rewards, dim=1), torch.stack(positions, dim=1)


def contact_replay(plan, assets, heads, output):
    import mujoco
    import torch
    from supermimic.q1_assets import identity
    from verify_reference import lowest_foot
    rows = []
    controls = 0
    for row in plan["contact_rows"]:
        health(plan["root"])
        with np.load(checked(row["trace"]), allow_pickle=False) as trace:
            actions = torch.from_numpy(trace["actions"].copy())
            saved = trace["state"].copy()
        with np.load(checked(row["reference"]), allow_pickle=False) as ref:
            qpos, qvel = ref["qpos"].copy(), ref["qvel"].copy()
        sim = initialize_sim(assets, row["reference"]["path"], heads)
        sim.reset(sim.ids, torch.tensor([0]))
        model = sim.models[0]
        floor = {gid for gid in range(model.ngeom) if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE}
        foot_sets = [{int(g) for g in assets.foot_geoms if model.body(int(model.geom_bodyid[g])).name == side + "_ankle_roll_link"}
                     for side in ("left", "right")]
        force, torque, clearance, ref_clearance, ref_speed, observed = [], [], [], [], [], []
        ref_data = mujoco.MjData(model)
        body_ids = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        state_error, data_state_error = 0., 0.
        try:
            for t, action in enumerate(actions):
                result = sim.step(action[None], auto_reset=False)
                controls += 1
                if controls > plan["max_contact_control_steps"]:
                    raise RuntimeError("Contact control budget exceeded")
                post = result["state"][0].numpy()
                state_error = max(state_error, float(np.max(np.abs(post - saved[t]))))
                dynamic = sim.data[0]
                data_state_error = max(data_state_error, float(np.max(np.abs(
                    np.concatenate([dynamic.qpos, dynamic.qvel]) - post))))
                if state_error > 2e-5 or data_state_error > 2e-5:
                    raise AssertionError("Contact replay does not reproduce the saved physical trace")
                normal = np.zeros(2)
                for i in range(dynamic.ncon):
                    c = dynamic.contact[i]
                    pair = {int(c.geom1), int(c.geom2)}
                    for side, geoms in enumerate(foot_sets):
                        if pair & floor and pair & geoms:
                            wrench = np.zeros(6)
                            mujoco.mj_contactForce(model, dynamic, i, wrench)
                            normal[side] += max(float(wrench[0]), 0.)
                force.append(normal)
                torque.append(dynamic.actuator_force[assets.aids].copy())
                clearance.append([lowest_foot(model, dynamic, list(g)) for g in foot_sets])
                ref_data.qpos[:], ref_data.qvel[:] = qpos[t], qvel[t]
                mujoco.mj_forward(model, ref_data)
                ref_clearance.append([lowest_foot(model, ref_data, list(g)) for g in foot_sets])
                velocity = np.zeros((2, 6))
                for i, bid in enumerate(body_ids):
                    mujoco.mj_objectVelocity(model, ref_data, mujoco.mjtObj.mjOBJ_XBODY, bid, velocity[i], 0)
                ref_speed.append(np.linalg.norm(velocity[:, 3:], axis=-1))
                observed.append(post)
        finally:
            sim.close()
        force, torque, clearance, ref_clearance, ref_speed = map(np.asarray, (force, torque, clearance, ref_clearance, ref_speed))
        ratio = np.abs(torque) / assets.effort
        raw = output / (row["clip"] + "_contacts.npz")
        np.savez_compressed(raw, normal_force_n=force, actuator_force_nm=torque,
                            actual_clearance_m=clearance, reference_clearance_m=ref_clearance,
                            reference_foot_speed_mps=ref_speed)
        near = ref_clearance <= .015
        supported = force > 1.
        result = {"clip": row["clip"], "steps": len(actions), "reference": row["reference"],
            "saved_trace": row["trace"], "replay_max_abs_state_error": state_error,
            "data_final_state_max_abs_error": data_state_error,
            "fraction_both_feet_force_above_1N": float(np.mean(supported.all(-1))),
            "fraction_neither_foot_force_above_1N": float(np.mean(~supported.any(-1))),
            "per_foot_force_above_1N_fraction": supported.mean(0).tolist(),
            "reference_per_foot_clearance_below_15mm_fraction": near.mean(0).tolist(),
            "reference_per_foot_clearance_quantiles_m": np.quantile(ref_clearance, [.0, .5, .95, 1.], axis=0).tolist(),
            "reference_mean_foot_speed_mps": ref_speed.mean(0).tolist(),
            "force_and_geometric_near_disagreement_fraction": float(np.mean(supported != near)),
            "actuator_saturation_fraction": float(np.mean(ratio >= .99)),
            "leg_saturation_fraction": float(np.mean(ratio[:, :12] >= .99)),
            "arm_saturation_fraction": float(np.mean(ratio[:, 14:22] >= .99)),
            "raw": identity(raw)}
        rows.append(result)
        print(json.dumps({"contact_replay": result}), flush=True)
    receipt = {"status": "SAVED_ACTION_CONTACT_REPLAY_COMPLETED", "plan": identity(output / "plan.json"),
               "rows": rows, "control_steps": controls,
               "force_timing": "Dynamic forces from native rollout's final solver substep; geometry also from that solver stage, before final integration, not a forward-only reconstructed force",
               "diagnostic_thresholds": "1N indicates nontrivial force; 15mm reference clearance is a descriptive comparison only, not true reference support or an admission gate",
               "optimizer_updates": 0}
    write_json(output / "contacts.json", receipt)


def probe(plan, assets, output, device):
    import torch
    from supermimic.q1_assets import identity
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.wm_trainer import reference_at
    checkpoint = torch.load(checked(plan["checkpoint"]), map_location=device, weights_only=False)
    if checkpoint["manifest"]["assets"] != assets.manifest():
        raise ValueError("World model/control contract changed")
    cfg = dict(checkpoint["manifest"]["world_model"])
    cfg["state_spec"] = StateSpecCfg(**cfg["state_spec"])
    model = WorldModel(WorldModelCfg(**cfg), assets.heads(device), device).eval()
    model.load_state_dict(checkpoint["wm"], strict=True)
    model.requires_grad_(False)
    del checkpoint
    counter = [0, plan["max_probe_control_steps"]]
    rows = []
    for index, sample in enumerate(plan["samples"]):
        health(plan["root"])
        with np.load(checked(sample["trace"]), allow_pickle=False) as saved:
            states = torch.from_numpy(saved["state"].copy())
            actions = torch.from_numpy(saved["actions"].copy()).clamp(-20, 20)
        start = sample["start"]
        initial, previous, window = states[start - 1:start], actions[start - 1:start], actions[start:start + 32]
        sim = initialize_sim(assets, checked(sample["reference"]), assets.heads())
        try:
            baseline, rewards, _, failures = native(sim, initial, previous, window, start, counter)
            replay_error = float(np.max(np.abs(baseline - states[start:start + 32].numpy()))) if len(baseline) == 32 else None
            if replay_error is None or replay_error > 2e-5 or failures.any():
                raise AssertionError("Saved 32-step baseline did not reproduce")
            references = reference_at(sim.lib, torch.arange(start, start + 32))
            for key, value in vars(references).items():
                setattr(references, key, value.to(device))
            s, a, p = initial.to(device), window.to(device), previous.to(device)
            delta = torch.zeros(1, 22, device=device, requires_grad=True)
            predicted_rewards, _ = model_rewards(model, s, p, a, references, delta)
            discount = GAMMA ** torch.arange(32, device=device)
            objective = (predicted_rewards * discount).cumsum(1)
            gradients = [torch.autograd.grad(objective[0, h - 1], delta, retain_graph=h != 32)[0][0].detach().cpu().numpy()
                         for h in HORIZONS]
            norms = np.linalg.norm(gradients, axis=-1)
            if np.min(norms) <= 1e-10 or not np.isfinite(norms).all():
                raise ValueError("Degenerate model reward gradient; retain failure without resampling")
            directions = np.vstack([np.asarray(gradients) / norms[:, None], sample["random_direction"]])
            variants = [(None, 0., 0)] + [(d, m, sign) for d in range(4) for m in MAGNITUDES for sign in (-1, 1)]
            deltas = np.zeros((17, 22), np.float32)
            native_states = np.full((17, 32, 57), np.nan, np.float32)
            native_rewards = np.zeros((17, 32), np.float64)
            lengths = np.zeros(17, int)
            failed = np.zeros(17, bool)
            native_states[0], native_rewards[0], lengths[0] = baseline, rewards, 32
            for vi, (di, magnitude, sign) in enumerate(variants[1:], 1):
                deltas[vi] = directions[di] * magnitude * sign
                perturbed = window.clone()
                perturbed[0] += torch.from_numpy(deltas[vi])
                native_s, native_r, _, termination = native(sim, initial, previous, perturbed, start, counter)
                native_states[vi, :len(native_s)], native_rewards[vi, :len(native_r)] = native_s, native_r
                lengths[vi], failed[vi] = len(native_s), bool(termination.any())
            with torch.no_grad():
                model_r, model_pos = model_rewards(model, s, p, a, references, torch.tensor(deltas, device=device))
            mr, mp = model_r.cpu().numpy(), model_pos.cpu().numpy()
            actual_return = (native_rewards * (GAMMA ** np.arange(32))).cumsum(1)
            predicted_return = (mr.astype(float) * (GAMMA ** np.arange(32))).cumsum(1)
            responses = []
            for di in range(4):
                for magnitude in MAGNITUDES:
                    minus, plus = [i for i, v in enumerate(variants) if v[0] == di and v[1] == magnitude]
                    for h in HORIZONS:
                        valid = bool(lengths[minus] >= h and lengths[plus] >= h and not (failed[minus] or failed[plus]))
                        native_fd = float((actual_return[plus, h - 1] - actual_return[minus, h - 1]) / (2 * magnitude)) if valid else None
                        predicted_fd = float((predicted_return[plus, h - 1] - predicted_return[minus, h - 1]) / (2 * magnitude))
                        ad = float(np.dot(gradients[HORIZONS.index(h)], directions[di]))
                        responses.append({"direction": "gradient_H" + str(HORIZONS[di]) if di < 3 else "random",
                            "magnitude": magnitude, "horizon": h, "valid": valid,
                            "model_directional_derivative": ad, "model_central_fd": predicted_fd,
                            "physics_central_fd": native_fd,
                            "positive_step_physics_return_gain": float(actual_return[plus, h - 1] - actual_return[0, h - 1]) if valid else None,
                            "positive_step_model_return_gain": float(predicted_return[plus, h - 1] - predicted_return[0, h - 1]),
                            "positive_step_physics_root_xy_change_m": (native_states[plus, h - 1, :2] - native_states[0, h - 1, :2]).tolist() if valid else None,
                            "physics_fd_matches_ad_sign": bool(native_fd * ad > 0) if valid and abs(native_fd) > 1e-7 and abs(ad) > 1e-7 else None})
            raw = output / f"sample_{index:02d}.npz"
            np.savez_compressed(raw, native_state=native_states, native_reward=native_rewards,
                model_reward=mr, model_root_position=mp, gradients=np.asarray(gradients), directions=directions,
                delta=deltas, lengths=lengths, failed=failed)
            row = {"sample": index, "clip": sample["clip"], "start": start,
                   "baseline_max_abs_state_error": replay_error, "gradient_norms": norms.tolist(),
                   "model_baseline_return": {str(h): float(predicted_return[0, h - 1]) for h in HORIZONS},
                   "physics_baseline_return": {str(h): float(actual_return[0, h - 1]) for h in HORIZONS},
                   "responses": responses, "raw": identity(raw)}
            rows.append(row)
            print(json.dumps({"sample_finished": index, "controls": counter[0], "gradient_norms": norms.tolist()}), flush=True)
        finally:
            sim.close()
    summary = {}
    for h in HORIZONS:
        selected = [r for sample in rows for r in sample["responses"]
                    if r["horizon"] == h and r["direction"] == "gradient_H" + str(h)]
        valid = [r for r in selected if r["valid"]]
        signs = [r["physics_fd_matches_ad_sign"] for r in valid if r["physics_fd_matches_ad_sign"] is not None]
        summary["H" + str(h)] = {"pairs": len(selected), "valid": len(valid),
            "nondegenerate_signs": len(signs), "correct_direction": sum(signs),
            "positive_step_improves_real_reward": sum(r["positive_step_physics_return_gain"] > 1e-7 for r in valid),
            "median_physics_return_gain": float(np.median([r["positive_step_physics_return_gain"] for r in valid])),
            "median_model_return_gain": float(np.median([r["positive_step_model_return_gain"] for r in valid]))}
    receipt = {"status": "LOCAL_OPEN_LOOP_REWARD_DIRECTION_PROBE_COMPLETED", "plan": identity(output / "plan.json"),
               "rows": rows, "summary": summary, "control_steps": counter[0],
               "physics_substeps": counter[0] * 20, "new_optimizer_updates": 0,
               "limits": plan["limits"], "final_health": health(plan["root"])}
    write_json(output / "reward_directions.json", receipt)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    import torch
    from contact_protocol import ContactCorrectedAssets
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "contacts", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
        return
    plan = json.loads((args.output / "plan.json").read_text())
    for key in ("source", "helper", "evaluation", "checkpoint"):
        checked(plan[key])
    health(args.root)
    started = time.monotonic()
    assets = ContactCorrectedAssets(args.root)
    if args.mode == "contacts":
        contact_replay(plan, assets, assets.heads(), args.output)
    else:
        probe(plan, assets, args.output, args.device)
    print(json.dumps({"mode": args.mode, "seconds": time.monotonic() - started,
                      "final_health": health(args.root)}), flush=True)


if __name__ == "__main__":
    main()
