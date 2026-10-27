"""Observe native solved contacts and test closed-loop action directions."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from worker import identity, now, prepare, save
from diagnose_q1_hp_physics import agreement


def contact_summary(force, speed):
    loaded = force > 5.0
    code = loaded[:, 0].astype(int) + 2*loaded[:, 1].astype(int)
    single = code[(code == 1) | (code == 2)]
    return {
        "double_support_fraction": float((code == 3).mean()),
        "single_support_fraction": float(((code == 1) | (code == 2)).mean()),
        "flight_fraction": float((code == 0).mean()),
        "single_support_side_changes": int(np.count_nonzero(np.diff(single))),
        "mean_vertical_force_n": force.mean(0).tolist(),
        "loaded_contact_slip_mean_mps": float(speed[loaded].mean()) if loaded.any() else None,
        "loaded_contact_slip_p95_mps": float(np.quantile(speed[loaded], .95)) if loaded.any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "library", "suite", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--contacts-only", action="store_true",
                        help="Repeat unobserved rollout and verify read-only contact observation; omit completed direction probe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp
    from evaluate_trial import rollout
    from run_trial import Q1Budget

    torch.set_num_threads(8)
    options = SimpleNamespace(runtime=args.runtime, output=args.output,
                              mode="diagnosis", expected_oom_kill=0)
    budget = Q1Budget(options)
    resources = budget.check()
    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"] + library["splits"]["validation"]
    names = [row["name"] for row in json.loads(args.suite.read_text())["clips"][:4]]
    selected = [next(i for i, row in enumerate(rows) if row["name"] == name) for name in names]
    combined = args.output/"evaluation_library.json"
    save(combined, {"splits": {"train": rows}})
    result = {"status": "RUNNING", "started_utc": now(), "optimizer_updates": 0,
              "runtime": str(args.runtime), "library": identity(args.library),
              "checkpoint": identity(args.trial/"run/final.pt"),
              "selected": names, "resources_initial": resources}
    save(args.output/"status.json", result)
    env = None
    try:
        _, env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=combined,
                                           output=args.output), num_envs=len(rows), eval_mode=True)
        from unisupermimic.rl.algos.supermimic.geometry import flatten_state, mat_to_quat, quat_to_mat, unflatten_state
        from unisupermimic.rl.algos.supermimic.networks import WorldModel
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale, default_joint_pos_list
        evaluate_module = importlib.import_module("unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate")

        device = torch.device(env.device)
        ids = torch.tensor(selected, device=device)
        backend = env.backend
        model, gpu_model, data = backend._cpu_model, backend._device_model, backend._device_data
        mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
        default = torch.tensor(default_joint_pos_list(), device=device)
        alpha = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES], device=device)
        checkpoint = torch.load(args.trial/"run/final.pt", map_location=device, weights_only=True)
        print("NATIVE_PREPARED", flush=True)
        baseline = rollout(env, checkpoint)
        print("UNOBSERVED_ROLLOUT_COMPLETED", flush=True)
        repeated = rollout(env, checkpoint) if args.contacts_only else None

        # Check the COM-centered spatial-velocity convention against MuJoCo.
        cpu = mujoco.MjData(model)
        cpu.qpos[:] = np.r_[baseline["initial"][selected[0], :7], baseline["initial"][selected[0], 13:35]]
        cpu.qvel[:] = np.linspace(-.2, .3, model.nv)
        mujoco.mj_forward(model, cpu)
        point_velocity_error = 0.0
        foot_ids = []
        for side in ("left", "right"):
            matches = [i for i in range(model.nbody)
                       if model.body(i).name.endswith(side+"_ankle_roll_link")]
            if len(matches) != 1:
                raise ValueError("Ambiguous native foot body: "+side)
            body = matches[0]
            foot_ids.append(body)
            for object_type, position in ((mujoco.mjtObj.mjOBJ_BODY, cpu.xipos[body]),
                                          (mujoco.mjtObj.mjOBJ_XBODY, cpu.xpos[body])):
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, cpu, object_type, body, velocity, 0)
                predicted = cpu.cvel[body, 3:] + np.cross(
                    cpu.cvel[body, :3], position-cpu.subtree_com[model.body_rootid[body]])
                point_velocity_error = max(point_velocity_error, float(np.max(np.abs(predicted-velocity[3:]))))
        if point_velocity_error > 1e-10:
            raise AssertionError("Spatial velocity convention differs")

        geom_body = torch.tensor(model.geom_bodyid, device=device)
        root_ids = torch.tensor(model.body_rootid, device=device)
        foot_owner = torch.full((model.nbody,), -1, dtype=torch.long, device=device)
        for side, root in enumerate(foot_ids):
            for body in range(model.nbody):
                parent = body
                while parent and parent != root:
                    parent = int(model.body_parentid[parent])
                if parent == root:
                    foot_owner[body] = side
        contact_ids = wp.array(np.arange(data.naconmax, dtype=np.int32), dtype=int, device=str(device))
        forces = wp.zeros(data.naconmax, dtype=wp.spatial_vector, device=str(device))
        observations = {name: [] for name in ("force", "slip", "torque", "before", "frames", "context", "previous", "history", "actions")}
        limit = torch.as_tensor(np.max(np.abs(model.actuator_forcerange), axis=1), device=device)
        original_step = backend.step_tensor
        original_loader = evaluate_module.load_policy
        observer_state_checks = 0

        def policy_loader(owner, payload, device=None):
            policy = original_loader(owner, payload, device)
            def record_policy(module, inputs, output):
                observations["history"].append(inputs[0][ids].detach().clone())
                observations["actions"].append(output[ids].detach().clone())
            policy.register_forward_hook(record_policy)
            return policy

        def observed_step(ctrl, nsteps=1):
            nonlocal observer_state_checks
            torch.testing.assert_close(ctrl[ids], default+alpha*observations["actions"][-1], atol=1e-6, rtol=1e-6)
            outcome = original_step(ctrl, nsteps)
            # Read before manager kinematics refresh and autoreset; no forward.
            torch.cuda.synchronize()
            wp.synchronize()
            preserved = {name: wp.to_torch(getattr(data, name))[ids].clone()
                         for name in ("qpos", "qvel", "qacc_warmstart")}
            count = int(wp.to_torch(data.nacon)[0])
            forces.zero_()
            mw.contact_force(gpu_model, data, contact_ids, True, forces)
            wp.synchronize()
            world = wp.to_torch(data.contact.worldid)[:count].long()
            geoms = wp.to_torch(data.contact.geom)[:count].long()
            point = wp.to_torch(data.contact.pos)[:count]
            contact_force = wp.to_torch(forces)[:count, :3]
            bodies = geom_body[geoms]
            normal = torch.zeros(env.num_envs, 2, device=device)
            weighted_speed = torch.zeros_like(normal)
            cvel, com = wp.to_torch(data.cvel), wp.to_torch(data.subtree_com)
            for side in range(2):
                foot_first = (foot_owner[bodies[:, 0]] == side) & (bodies[:, 1] == 0)
                foot_second = (foot_owner[bodies[:, 1]] == side) & (bodies[:, 0] == 0)
                keep = foot_first | foot_second
                wi, cp = world[keep], point[keep]
                body = torch.where(foot_first[keep], bodies[keep, 0], bodies[keep, 1])
                velocity = cvel[wi, body, 3:] + torch.linalg.cross(cvel[wi, body, :3], cp-com[wi, root_ids[body]])
                vertical = contact_force[keep, 2].abs()
                normal[:, side].index_add_(0, wi, vertical)
                weighted_speed[:, side].index_add_(0, wi, vertical*velocity[:, :2].norm(dim=-1))
            pre = env._pre_action
            observations["force"].append(normal[ids].clone())
            observations["slip"].append((weighted_speed/normal.clamp(min=1e-8))[ids].clone())
            observations["torque"].append(wp.to_torch(data.actuator_force)[ids].clone())
            observations["before"].append(flatten_state(pre["state"])[ids].clone())
            observations["frames"].append(pre["frames"][ids].clone())
            observations["context"].append(pre["context"][ids].clone())
            observations["previous"].append(pre["previous_action"][ids].clone())
            for name, before in preserved.items():
                if not torch.equal(before, wp.to_torch(getattr(data, name))[ids]):
                    raise AssertionError("Contact observer changed native "+name)
            observer_state_checks += 1
            if len(observations["force"]) % 100 == 0:
                budget.check()
            return outcome

        try:
            backend.step_tensor = observed_step
            evaluate_module.load_policy = policy_loader
            observed = rollout(env, checkpoint)
        finally:
            backend.step_tensor = original_step
            evaluate_module.load_policy = original_loader
        print("OBSERVED_ROLLOUT_COMPLETED", flush=True)
        values = {key: torch.stack(value) for key, value in observations.items()}
        with np.load(args.trial/"evaluation/final_trajectory.npz", allow_pickle=False) as saved:
            old_steps = saved["steps"][selected]
            old_states = saved["states"][:, selected].copy()
        contacts = []
        def prefix_errors(first_trace, second_trace, index):
            return {str(n): float(np.max(np.abs(first_trace["states"][:min(n, int(first_trace["steps"][index]), int(second_trace["steps"][index])), index]
                                      -second_trace["states"][:min(n, int(first_trace["steps"][index]), int(second_trace["steps"][index])), index])))
                    for n in (1, 10, 50, 100)}
        for i, index in enumerate(selected):
            count = int(observed["steps"][index])
            shared = min(count, int(baseline["steps"][index]))
            old_shared = min(count, int(old_steps[i]))
            contacts.append({"name": names[i], "steps": count,
                "reference_steps": int(motion.clip_len[index])-1,
                "failed": bool(observed["failed"][index]),
                "unobserved_steps": int(baseline["steps"][index]),
                "observation_common_state_max_abs": float(np.max(np.abs(observed["states"][:shared, index]-baseline["states"][:shared, index]))),
                "RF09_saved_steps": int(old_steps[i]),
                "RF09_common_state_max_abs": float(np.max(np.abs(observed["states"][:old_shared, index]-old_states[:old_shared, i]))),
                "unobserved_vs_observed_prefix_max_abs": prefix_errors(baseline, observed, index),
                "unobserved_vs_repeat_prefix_max_abs": prefix_errors(baseline, repeated, index) if repeated is not None else None,
                "repeated_unobserved_steps": int(repeated["steps"][index]) if repeated is not None else None,
                "torque_saturation_fraction": float((values["torque"][:count, i].abs() >= .98*limit).float().mean()),
                **contact_summary(values["force"][:count, i].cpu().numpy(), values["slip"][:count, i].cpu().numpy())})
        save(args.output/"contact.json", {"status": "NATIVE_SOLVED_CONTACT_OBSERVED", "contacts": contacts,
            "point_velocity_contract_max_abs": point_velocity_error,
            "observer_selected_state_invariance_checks": observer_state_checks,
            "observer_state_fields": ["qpos", "qvel", "qacc_warmstart"],
            "phase": "Last 5ms solver substep of each 20ms control interval, before sensor refresh or autoreset; no extra forward or integration",
            "limits": "Snapshot contact force, not time-integrated 20ms impulse. Old-GPU RF09 agreement is reported separately from same-process observation invariance."})
        saved_arrays = {key: value.cpu().numpy() for key, value in values.items()}
        saved_arrays.update(states=observed["states"][:, selected], alive=observed["alive"][:, selected],
                            steps=observed["steps"][selected], failed=observed["failed"][selected])
        np.savez_compressed(args.output/"native_trace.npz", **saved_arrays)
        if args.contacts_only:
            result.update(status="NATIVE_READ_ONLY_CONTACT_DIAGNOSIS_COMPLETED", completed_utc=now(),
                          contact=identity(args.output/"contact.json"),
                          trace=identity(args.output/"native_trace.npz"), resources_final=budget.check())
            save(args.output/"status.json", result)
            print(json.dumps({"status":result["status"], "contact":contacts}), flush=True)
            return

        # Branches reuse this native model and control path, with matched cold
        # starts; each branch recomputes future policy actions for all 32 steps.
        uniform = {}
        for name in ("body_mass", "body_inertia", "body_ipos", "dof_armature", "geom_friction", "actuator_gainprm", "actuator_biasprm"):
            tensor = wp.to_torch(getattr(gpu_model, name))
            variation = float((tensor-tensor[:1]).abs().max()) if tensor.ndim > 1 and tensor.shape[0] == env.num_envs else 0.0
            uniform[name] = {"shape": list(tensor.shape), "world_variation_max_abs": variation}
            if variation > 1e-7:
                raise ValueError("Branch protocol requires nominal identical per-world parameters: "+name)
        policy = original_loader(env, checkpoint)
        wm = WorldModel(hidden=tuple(checkpoint["args"]["wm_hidden"]), num_joints=22, context_dim=26).to(device)
        wm.load_state_dict(checkpoint["wm"])
        for module in (policy, wm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        cases = []
        for i, row in enumerate(contacts):
            fractions = (.2, .55) if i < 2 else (.4,)
            latest = min(row["steps"]-1, row["reference_steps"]-33)
            for fraction in fractions:
                if latest >= 2:
                    cases.append((i, max(1, int(latest*fraction))))
        variants, epsilon = 45, .08
        if len(cases)*variants > env.num_envs or not cases:
            raise ValueError("Insufficient native worlds for the preselected direction cases")
        starts = torch.stack([values["before"][k, i] for i, k in cases])
        first = torch.stack([values["actions"][k, i] for i, k in cases])
        history = torch.stack([values["history"][k, i] for i, k in cases])
        frames = torch.stack([values["frames"][k, i] for i, k in cases])
        previous = torch.stack([values["previous"][k, i] for i, k in cases])
        context = torch.stack([values["context"][k, i] for i, k in cases])
        active_worlds = len(cases)*variants
        selector = torch.arange(env.num_envs, device=device).clamp(max=active_worlds-1)//variants
        branch = starts[selector]
        p, rotation, velocity, angular, joints, joint_velocity = unflatten_state(branch)
        wp.to_torch(data.qpos).copy_(torch.cat((p, mat_to_quat(rotation), joints), -1))
        wp.to_torch(data.qvel).copy_(torch.cat((velocity, (rotation.transpose(-1,-2)@angular.unsqueeze(-1)).squeeze(-1), joint_velocity), -1))
        wp.to_torch(data.qacc_warmstart).zero_()
        mw.forward(gpu_model, data)
        wp.synchronize()
        offsets = torch.zeros(variants, 22, device=device)
        for joint in range(22):
            offsets[1+2*joint, joint], offsets[2+2*joint, joint] = -epsilon, epsilon
        perturb = offsets.repeat(len(cases), 1)
        if active_worlds < env.num_envs:
            perturb = torch.cat((perturb, perturb[-1:].repeat(env.num_envs-active_worlds, 1)))
        branch_history = history[selector].clone()
        branch_previous = previous[selector].clone()
        branch_context = context[selector]
        branch_frames = frames[selector]
        model_state = unflatten_state(starts)
        model_history, model_previous = history.clone(), previous.clone()
        first_variable = first.detach().clone().requires_grad_(True)
        horizons = (1, 4, 8, 32)
        totals_physical = {name: torch.zeros(env.num_envs, device=device) for name in ("tracking", "upper_pose", "leg_pose")}
        totals_model = {name: torch.zeros(len(cases), device=device) for name in totals_physical}
        comparisons = []
        def metrics(state, frame, action, prev):
            reward, terms = mdp.reward(state, frame, action, prev, terms=True)
            del reward
            return {"tracking": terms["track"],
                    "upper_pose": -(state[4][:, 14:]-motion.q[frame, 14:]).square().mean(-1)*.02,
                    "leg_pose": -(state[4][:, :12]-motion.q[frame, :12]).square().mean(-1)*.02}

        # The backend action term supplies the same target conversion used by
        # env.step, without invoking manager resets or changing reference time.
        for horizon in range(1, 33):
            with torch.no_grad():
                action = first[selector]+perturb if horizon == 1 else policy(branch_history, mdp.goal(branch_frames+horizon-1, checkpoint["args"]["goal_frames"]))
                original_step(default+alpha*action, 4)
                torch.cuda.synchronize()
                qpos, qvel = wp.to_torch(data.qpos), wp.to_torch(data.qvel)
                r = quat_to_mat(qpos[:, 3:7])
                state = (qpos[:, :3], r, qvel[:, :3], (r@qvel[:, 3:6,None]).squeeze(-1), qpos[:,7:], qvel[:,6:])
                for name, value in metrics(state, branch_frames+horizon, action, branch_previous).items():
                    totals_physical[name] += value
                obs = mdp.obs(state, action, branch_context, branch_frames+horizon)
                branch_history = torch.cat((branch_history[:,1:], obs[:,None]),1)
                branch_previous = action
            model_action = first_variable if horizon == 1 else policy(model_history, mdp.goal(frames+horizon-1, checkpoint["args"]["goal_frames"]))
            model_state = wm.step(model_state, model_action, mdp.normalize_context(context))
            for name, value in metrics(model_state, frames+horizon, model_action, model_previous).items():
                totals_model[name] = totals_model[name]+value
            model_obs = mdp.obs(model_state, model_action, context, frames+horizon)
            model_history = torch.cat((model_history[:,1:], model_obs[:,None]),1)
            model_previous = model_action
            if horizon in horizons:
                for name in totals_model:
                    gradient = torch.autograd.grad(totals_model[name].sum(), first_variable, retain_graph=True)[0]
                    physical = totals_physical[name][:active_worlds].reshape(len(cases),variants)
                    difference = (physical[:,2::2]-physical[:,1::2])/(2*epsilon)
                    for i, (clip, step) in enumerate(cases):
                        comparisons.append({"clip":names[clip], "step":step, "horizon":horizon, "metric":name,
                            "force_before_action_n":values["force"][step-1,clip].cpu().tolist(),
                            "all_axes":agreement(gradient[i].detach().cpu().numpy(), difference[i].cpu().numpy()),
                            "upper_axes":agreement(gradient[i,14:].detach().cpu().numpy(), difference[i,14:].cpu().numpy()),
                            "leg_axes":agreement(gradient[i,:12].detach().cpu().numpy(), difference[i,:12].cpu().numpy())})
                budget.check()
        summary = {}
        for metric in totals_model:
            for horizon in horizons:
                group = [item for item in comparisons if item["metric"] == metric and item["horizon"] == horizon]
                cosine = [item["all_axes"]["cosine"] for item in group if item["all_axes"]["cosine"] is not None]
                signs = [item["all_axes"]["sign_fraction"] for item in group if item["all_axes"]["sign_fraction"] is not None]
                summary[f"{metric}_H{horizon}"] = {"samples":len(group), "mean_cosine":float(np.mean(cosine)) if cosine else None,
                    "mean_sign_fraction":float(np.mean(signs)) if signs else None}
        save(args.output/"directions.json", {"status":"NATIVE_CLOSED_LOOP_DIRECTION_DIAGNOSIS_COMPLETED",
            "horizons":list(horizons), "epsilon":epsilon, "cases":len(cases), "comparisons":comparisons,
            "summary":summary, "nominal_model_uniformity":uniform,
            "scope":"Cumulative tracking and pose scores; first-action perturbation, frozen policy recomputed thereafter; matched cold starts in native backend. BF16 policy/model inherited from RF09.",
            "limits":"Not the complete noisy actor objective: excludes learned critic bootstrap, discount and survival weights. No full-pipeline causality or training improvement claim."})
        result.update(status="NATIVE_CONTACT_AND_CLOSED_LOOP_DIRECTION_COMPLETED", completed_utc=now(),
                      contact=identity(args.output/"contact.json"), directions=identity(args.output/"directions.json"),
                      trace=identity(args.output/"native_trace.npz"), resources_final=budget.check())
        save(args.output/"status.json", result)
        print(json.dumps({"status":result["status"], "contact":contacts, "direction_summary":summary}), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        save(args.output/"status.json", result)
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()


if __name__ == "__main__":
    main()
