#!/usr/bin/env python3
"""Numerical contracts for full A3 state, rewards, replay and policy gradients."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.a3_sim import A3Sim, SimCfg
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
from supermimic.world_model.heads import analytic_observation
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg, reference_at


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(1404)
    torch.set_num_threads(2)
    assets = A3Assets(args.root)
    cpu_heads = assets.heads("cpu", dtype=torch.float64)
    rng = np.random.default_rng(1404)
    position_error = velocity_error = rotation_error = 0.
    for _ in range(12):
        d = mujoco.MjData(assets.model)
        d.qpos[:] = assets.model.qpos0
        d.qpos[:3] = [.1, -.2, 1.2]
        quat = rng.normal(size=4)
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        d.qpos[7:] = rng.uniform(-.03, .03, size=35)
        d.qvel[:] = rng.normal(size=41) * .2
        mujoco.mj_forward(assets.model, d)
        out = cpu_heads.fk(torch.tensor(d.qpos[None]), torch.tensor(d.qvel[None]))
        position_error = max(position_error, float(np.abs(out["pos"].numpy()[0] - d.xpos[1:]).max()))
        products = np.abs((out["quat"].numpy()[0] * d.xquat[1:]).sum(-1))
        rotation_error = max(rotation_error, float(np.abs(1. - products).max()))
        for idx in range(1, assets.model.nbody):
            actual = np.empty(6)
            mujoco.mj_objectVelocity(assets.model, d, mujoco.mjtObj.mjOBJ_XBODY, idx, actual, 0)
            pred = np.concatenate([out["ang_vel"][0, idx - 1].numpy(), out["lin_vel"][0, idx - 1].numpy()])
            velocity_error = max(velocity_error, float(np.abs(actual - pred).max()))
    print(json.dumps({"fk_position_max_abs_m": position_error, "fk_rotation_max_abs": rotation_error,
                      "fk_velocity_max_abs": velocity_error}), flush=True)
    check(position_error < 1e-9 and rotation_error < 1e-9 and velocity_error < 1e-9, "Differentiable A3 FK disagrees with MuJoCo")
    device = args.device
    heads = assets.heads(device)
    lib = MotionLibrary([args.motions / "042_squat_shallow_repeated.npz"], TRACKED, assets.q0,
                        device=device, active_indices=assets.active_indices)
    frames = torch.arange(10, device=device)
    qp, qv = lib.ref_qpos_qvel(frames)
    state = torch.cat([qp, qv], -1)
    ref = reference_at(lib, frames)
    pred = heads.robot_bodies(state)
    ref_error = float((pred["pos"] - ref.body_pos_w).abs().max())
    ref_velocity_error = float((pred["lin_vel"] - ref.body_lin_vel_w).abs().max())
    check(ref_error < 2e-6 and ref_velocity_error < 2e-6, "Saved motion state and saved body reference disagree")
    xi = torch.zeros(10, 33, device=device)
    xi[:, 0] = 1.
    zero_a = torch.zeros(10, 29, device=device)
    obs = analytic_observation(state, zero_a, xi, heads.default_joint_pos, heads.active_indices)
    check(obs.shape == (10, 93) and bool(torch.isfinite(obs).all()), "Actor observation contract")
    expected_pos = state[:, 7:42][:, heads.active_indices] - heads.default_joint_pos
    check(bool(torch.equal(obs[:, :29], expected_pos)), "Passive/head joints leaked into actor observation")
    # Positive terminal label belongs to transition t, not the action after that transition.
    replay = ReplayBuffer(1, 32, 83, 29, 33, device)
    ids = torch.tensor([0], device=device)
    for t in range(4):
        a = torch.full((1, 29), float(t), device=device)
        replay.add(ids, state[:1], a, a, xi[:1], torch.ones(1, device=device),
                   torch.tensor([float(t == 2)], device=device), torch.tensor([t], device=device),
                   ids, torch.tensor([t == 3], device=device), torch.tensor([t == 0], device=device))
    sample = replay.sample_windows(8, 3)
    check(bool((sample.done[2] == 1.).all()) and bool((sample.done[:2] == 0.).all()), "Replay terminal label off by one")
    wm = WorldModel(WorldModelCfg(hidden=(128, 128, 128)), heads, device)
    output = wm.step(state, zero_a, xi)
    check(output.shape == (10, 83) and bool(torch.isfinite(output).all()), "Full model state shape")
    sim = A3Sim(assets, lib, SimCfg(num_envs=8, threads=4, domain_randomization=False,
                                  observation_noise=False, action_noise=0.), heads)
    buffer = ReplayBuffer(8, 256, 83, 29, 33, device)
    start = time.perf_counter()
    try:
        for _ in range(48):
            sim.step(torch.randn(8, 29, device=device) * .05, buffer)
        physics_s = time.perf_counter() - start
        policy_cfg = PolicyCfg(encoder_hidden=(128, 128), decoder_hidden=(128, 128))
        policy = Policy(policy_cfg).to(device)
        critic = Critic(CriticCfg(), pcfg=policy_cfg).to(device)
        wm_trainer = WorldModelTrainer(wm, lib, WMTrainerCfg(batch_size=16, batches_per_update=2, compile=None))
        wm_metrics = wm_trainer.update(buffer)
        trainer = PolicyTrainer(wm, policy, critic, lib,
            PolicyTrainerCfg(batch_size=8, horizon=32, compile=None, critic_epochs=1, critic_minibatches=2))
        before = {n: x.detach().clone() for n, x in policy.named_parameters()}
        wm_before = {n: x.detach().clone() for n, x in wm.named_parameters()}
        metrics = trainer.update(buffer)
        changed = [n for n, x in policy.named_parameters() if not torch.equal(x, before[n])]
        check(len(changed) > 0 and metrics["policy_grad_norm"] > 0., "No BPTT policy update")
        check(all(torch.equal(x, wm_before[n]) for n, x in wm.named_parameters()), "World model changed during actor optimization")
        check(all(np.isfinite(v) for v in wm_metrics.values()) and all(np.isfinite(v) for v in metrics.values()), "Nonfinite model/actor/critic metrics")
        receipt = {"status": "BACKEND_AND_GRADIENT_CONTRACT_PASSED", "assets": assets.manifest(),
                   "fk_position_max_abs_m": position_error, "fk_rotation_max_abs": rotation_error,
                   "fk_velocity_max_abs": velocity_error, "saved_reference_position_max_abs_m": ref_error,
                   "saved_reference_velocity_max_abs": ref_velocity_error,
                   "physics_env_steps": 48 * 8, "physics_seconds": physics_s,
                   "model_metrics": wm_metrics, "actor_critic_metrics": metrics,
                   "actor_parameters_changed": changed, "actor_horizon": 32, "model_window": 8,
                   "terminal_alignment_passed": True, "wm_frozen_during_actor": True,
                   "implementation": identity(__file__), "not_a_training_effect_result": True}
        receipt["source_files"] = [identity(f) for f in sorted((Path(__file__).parent / "supermimic").rglob("*.py"))]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
