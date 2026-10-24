#!/usr/bin/env python3
"""Probe bounded H32 differentiation using the audited saved policy and real replay."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from prepare_amass_body import identity


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def probe(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget

    root = args.root.resolve(strict=True)
    experiment = root / "data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025"
    budget = MemoryBoundBudget(StorageBudget(experiment, experiment, 100 * 1024 ** 3,
                                            12 * 1024 ** 3, 12 * 1024 ** 3),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    before = budget.check()
    audit = json.loads(args.checkpoint_audit.read_text())
    if audit["status"] != "FAILED_R02C_SAVED1500_AUDIT_PASSED" or not all(audit["gates"].values()):
        raise ValueError("Require the independently audited saved1500 state")
    import numpy as np
    import torch

    from bounded_policy import ACTION_LIMIT, BoundedPolicy
    from supermimic.data.replay_buffer import ReplayBuffer
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.imagination.compiled import ImagStep
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
    from supermimic.q1_assets import TRACKED
    from supermimic.q1_sim import SimCfg
    from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from contact_protocol import ContactCorrectedAssets

    sys.path.insert(0, str(HERE.parent / "supermimic_q1_expanded"))
    from slotted_sim import SlottedQ1Sim

    torch.set_num_threads(2)
    torch.manual_seed(2026102510)
    np.random.seed(2026102510)
    record = audit["checkpoint"]
    if identity(record["path"]) != record:
        raise ValueError("Audited checkpoint changed")
    saved = torch.load(record["path"], map_location="cpu", mmap=True, weights_only=False)
    manifest = saved["manifest"]
    spec = json.loads((experiment / "spec.json").read_text())
    motion_manifest = json.loads(Path(manifest["motion_manifest"]["path"]).read_text())
    rows = {row["clip"]: row for row in motion_manifest["motions"]}
    ordered = sorted(spec["split_clips"]["train"], key=lambda name: hashlib.sha256(
        ("2026102510:" + name).encode()).digest())
    selected = ([name for name in ordered if rows[name]["bucket"] == "locomotion"][:2]
                + [name for name in ordered if rows[name]["bucket"] != "locomotion"][:2])
    if len(selected) != 4:
        raise ValueError("Need two train displacement and two other train references")
    references = [rows[name]["output"] for name in selected]
    inputs = [record, identity(args.checkpoint_audit), identity(experiment / "spec.json"),
              manifest["motion_manifest"], *references,
              *[identity(HERE / name) for name in ("bounded_policy.py", "probe_bounded_imagination.py")]]
    for item in inputs:
        if identity(item["path"]) != item:
            raise ValueError("Probe input changed")
    device = "cuda:0"
    assets = ContactCorrectedAssets(root)
    if assets.manifest() != manifest["assets"]:
        raise ValueError("Saved Q1 control assets changed")
    heads = assets.heads(device)
    library = MotionLibrary([Path(row["path"]) for row in references], TRACKED, assets.q0,
                            device=device, active_indices=assets.active_indices)
    wcfg = WorldModelCfg(**{**manifest["world_model"],
                           "state_spec": StateSpecCfg(**manifest["world_model"]["state_spec"])})
    pcfg = PolicyCfg(**manifest["policy"])
    wm = WorldModel(wcfg, heads, device)
    original, bounded = Policy(pcfg).to(device), BoundedPolicy(pcfg).to(device)
    critic = Critic(CriticCfg(**manifest["critic"]), pcfg=pcfg).to(device)
    for key, module in (("wm", wm), ("policy", original), ("policy", bounded), ("critic", critic)):
        module.load_state_dict(saved[key], strict=True)
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    simulator = SlottedQ1Sim(assets, library, SimCfg(**{**manifest["sim"], "num_envs": 16, "threads": 4}),
                            heads, 2026102510, slots=4)
    replay = ReplayBuffer(16, 1000, STATE_DIM, ACT_DIM, XI_DIM, device)
    try:
        with torch.no_grad():
            for step in range(128):
                obs, goal = simulator.policy_inputs()
                simulator.step(original(obs, goal), replay)
                if step % 16 == 0:
                    budget.memory()
        start = replay.sample_starts(128, pcfg.history)
        step_fn = ImagStep(wm, library.default_joint_pos, compile_mode=None)
        cases = []
        with torch.no_grad():
            for seed in range(2026102510, 2026102516):
                results, rollouts = {}, []
                for label, policy in (("original", original), ("bounded", bounded)):
                    torch.manual_seed(seed)
                    ro = imagine(wm, policy, library, start, 32, obs_noise=True, step_fn=step_fn)
                    action_max = finite_number(ro.actions.abs().max())
                    results[label] = {"action_abs_max": action_max,
                                      "outside_real_action_domain_fraction": float((ro.actions.abs() > ACTION_LIMIT).float().mean()),
                                      "states_finite": bool(torch.isfinite(ro.states).all()),
                                      "rewards_finite": bool(torch.isfinite(ro.rewards).all()),
                                      "mean_action_rate_reward": finite_number(ro.reward_terms["action_rate_l2"])}
                    rollouts.append(ro)
                results["state_max_abs_difference"] = finite_number((rollouts[0].states - rollouts[1].states).abs().max())
                results["seed"] = seed
                cases.append(results)
                budget.memory()
        torch.manual_seed(2026102510)
        bounded.zero_grad(set_to_none=True)
        ro = imagine(wm, bounded, library, start, 32, obs_noise=True, step_fn=step_fn)
        final_values = critic(wm.spec.canonical(ro.states[-1]), ro.goals[-1],
                              wm.spec.normalize_xi(start.xi), ro.e_root[-1])
        objective = policy_objective(ro, final_values, manifest["policy_trainer"]["gamma"])
        (-objective / 32).backward()
        norms = [parameter.grad.norm() for parameter in bounded.parameters() if parameter.grad is not None]
        gradient_norm = float(torch.stack(norms).norm())
        gates = {"real_replay_collected": replay.total_rows >= 2048,
                 "bounded_h32_cases_finite": all(row["bounded"]["states_finite"]
                                                 and row["bounded"]["rewards_finite"] for row in cases),
                 "bounded_h32_actions": all(row["bounded"]["outside_real_action_domain_fraction"] == 0 for row in cases),
                 "bounded_action_rate_lower_bound": all(row["bounded"]["mean_action_rate_reward"] is not None
                                                         and row["bounded"]["mean_action_rate_reward"] >= -70.4001
                                                         for row in cases),
                 "bounded_actor_gradient_finite_nonzero": np.isfinite(gradient_norm) and gradient_norm > 1e-10,
                 "wm_and_critic_no_parameter_gradients": not any(parameter.grad is not None
                                                                  for module in (wm, critic) for parameter in module.parameters()),
                 "inputs_unchanged": all(identity(item["path"]) == item for item in inputs)}
        report = {"status": "BOUNDED_SAVED_POLICY_H32_PROBE_PASSED" if all(gates.values()) else "BOUNDED_SAVED_POLICY_H32_PROBE_FAILED",
                  "captured_utc": datetime.now(timezone.utc).isoformat(), "gates": gates,
                  "inputs": inputs, "train_clips": selected, "cases": cases,
                  "actor_gradient_norm": gradient_norm, "imagined_objective": float(objective.detach()),
                  "new_physics_control_steps": 2048, "new_optimizer_updates": 0,
                  "horizon": 32, "batch_size": 128, "health_before": before, "health_after": budget.check(),
                  "limits": "Fresh train-only replay and six paired imagined forward cases, plus one actor backward. This is not an exact replay of failed iteration1810, a training experiment, or proof of tracking improvement.",
                  "storage_status": "LOCAL_ONLY"}
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, allow_nan=False), flush=True)
        return 0 if all(gates.values()) else 2
    finally:
        simulator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raise SystemExit(probe(args))
