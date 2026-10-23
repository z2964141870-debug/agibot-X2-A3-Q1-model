#!/usr/bin/env python3
"""Check Q1 FK, native physics, state integration and frozen-model BPTT."""

import argparse
import copy
from dataclasses import fields
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np
import torch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, GOAL_DIM, NQ, OBS_DIM, STATE_DIM, XI_DIM
from supermimic.world_model.state_spec import DELTA_DIM, StateSpec, StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def diagnostic_reference(assets, destination):
    model, data = assets.model, mujoco.MjData(assets.model)
    data.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(model, data)
    count = 200
    repeat = lambda value: np.repeat(np.asarray(value)[None], count, axis=0)
    shape = (count, len(TRACKED), 3)
    np.savez_compressed(destination, qpos=repeat(data.qpos), qvel=repeat(data.qvel),
                        joint_pos=repeat(data.qpos[7:]), joint_vel=repeat(data.qvel[6:]),
                        body_names=np.array(TRACKED), fps=np.array(50.),
                        body_pos_w=repeat(data.xpos[assets.body_ids]),
                        body_quat_w=repeat(data.xquat[assets.body_ids]),
                        body_lin_vel_w=np.zeros(shape), body_ang_vel_w=np.zeros(shape),
                        scope=np.array("DIAGNOSTIC_ONLY_NOT_TRAINING"))


def check_fk(assets):
    rng = np.random.default_rng(20261023)
    model, data = assets.model, mujoco.MjData(assets.model)
    fk = assets.heads(dtype=torch.float64).fk
    positions, quats, linear, angular, qs, vs = [], [], [], [], [], []
    for _ in range(8):
        q = model.qpos0.copy()
        q[:3] += rng.normal(0, .05, 3)
        q[3:7] = rng.normal(size=4)
        q[3:7] /= np.linalg.norm(q[3:7])
        limits = model.jnt_range[assets.jids]
        q[assets.qadr] = rng.uniform(limits[:, 0] + .02, limits[:, 1] - .02)
        v = rng.normal(0, .3, model.nv)
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        velocities = []
        for bid in range(1, model.nbody):
            value = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, bid, value, 0)
            velocities.append(value)
        qs.append(q)
        vs.append(v)
        positions.append(data.xpos[1:].copy())
        quats.append(data.xquat[1:].copy())
        angular.append(np.asarray(velocities)[:, :3])
        linear.append(np.asarray(velocities)[:, 3:])
    q = torch.tensor(np.asarray(qs), dtype=torch.float64)
    v = torch.tensor(np.asarray(vs), dtype=torch.float64)
    out = fk(q, v)
    errors = {key: float(np.abs(out[key].detach().numpy() - np.asarray(value)).max())
              for key, value in (("pos", positions), ("lin_vel", linear), ("ang_vel", angular))}
    predicted, true = out["quat"].numpy(), np.asarray(quats)
    errors["quat"] = float(np.minimum(np.abs(predicted-true).max(-1), np.abs(predicted+true).max(-1)).max())
    if max(errors.values()) > 1e-8:
        raise AssertionError(f"Q1 FK differs from MuJoCo: {errors}")
    variable = q[:1].clone().requires_grad_(True)
    weight = torch.tensor(rng.normal(size=(1, model.nbody-1, 3)), dtype=torch.float64)
    gradient = torch.autograd.grad((fk(variable)["pos"] * weight).sum(), variable)[0]
    direction = torch.zeros_like(variable)
    direction[:, 7:] = torch.tensor(rng.normal(size=(1, ACT_DIM)))
    direction /= direction.norm()
    epsilon = 1e-6
    with torch.no_grad():
        fd = (((fk(variable + epsilon*direction)["pos"] - fk(variable - epsilon*direction)["pos"])
               * weight).sum() / (2*epsilon))
    ad = (gradient * direction).sum()
    errors["fk_directional_gradient_abs"] = float((fd-ad).abs())
    if errors["fk_directional_gradient_abs"] > 1e-7:
        raise AssertionError("Q1 FK directional derivative failed")
    spec = StateSpec(StateSpecCfg(), "cpu", torch.float64)
    state = torch.cat([q, v], -1)
    integrated = spec.integrate(state, torch.zeros(8, DELTA_DIM, dtype=torch.float64))
    expected = np.asarray(qs).copy()
    for i in range(8):
        mujoco.mj_integratePos(model, expected[i], np.asarray(vs)[i], .02)
    errors["zero_residual_integration_abs"] = float(np.abs(integrated[:, :NQ].numpy() - expected).max())
    if errors["zero_residual_integration_abs"] > 1e-8:
        raise AssertionError("Q1 state slices/integration differ from MuJoCo")
    return errors


def check_graph(assets, library, replay):
    torch.manual_seed(20261023)
    heads = assets.heads(dtype=torch.float64)
    wm = WorldModel(WorldModelCfg(hidden=(64, 64), term_hidden=(32, 32)), heads, "cpu").double()
    policy = Policy(PolicyCfg(encoder_hidden=(64, 32), decoder_hidden=(64, 32))).double()
    last = [layer for layer in wm.dynamics.modules() if isinstance(layer, torch.nn.Linear)][-1]
    # A fixture with action dependence tests the graph; it is never saved as trained weights.
    torch.nn.init.normal_(last.weight, std=1e-4)
    wm.requires_grad_(False)
    sample = replay.sample_starts(2, 10)
    start = StartBatch(**{field.name: (getattr(sample, field.name).double()
                                    if getattr(sample, field.name).is_floating_point()
                                    else getattr(sample, field.name)) for field in fields(sample)})
    action = torch.zeros(2, ACT_DIM, dtype=torch.float64, requires_grad=True)
    weights = torch.linspace(.1, 1., STATE_DIM, dtype=torch.float64)[None]
    score = (wm.step(start.state, action, start.xi) * weights).sum()
    gradient = torch.autograd.grad(score, action)[0]
    direction = torch.randn_like(action)
    direction /= direction.norm()
    epsilon = 1e-5
    with torch.no_grad():
        fd = ((wm.step(start.state, action + epsilon*direction, start.xi)
               - wm.step(start.state, action - epsilon*direction, start.xi)) * weights).sum() / (2*epsilon)
    ad = (gradient * direction).sum()
    if not torch.isfinite(gradient).all() or gradient.norm() <= 1e-10 or (fd-ad).abs() > 1e-7:
        raise AssertionError("Frozen Q1 world-model action gradient failed")
    rollout = imagine(wm, policy, library, start, 32, obs_noise=False)
    score = rollout.states[-1, :, :NQ].square().sum()
    policy.zero_grad(set_to_none=True)
    score.backward()
    actor_norm = sum(float(parameter.grad.square().sum()) for parameter in policy.parameters()
                     if parameter.grad is not None) ** .5
    if not np.isfinite(actor_norm) or actor_norm <= 1e-10 or any(p.grad is not None for p in wm.parameters()):
        raise AssertionError("H32 Q1 actor graph or world-model freeze failed")
    return {"horizon": 32, "world_model_fitted": False, "optimizer_updates": 0,
            "action_gradient_norm": float(gradient.norm()), "action_gradient_fd_abs": float((fd-ad).abs()),
            "actor_terminal_state_gradient_norm": actor_norm, "world_model_parameter_gradients": 0,
            "fixture_networks": "64-wide; verifies differentiation, not learned model accuracy or main training performance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    result = {"status": "FAILED", "optimizer_updates": 0, "physics_control_steps": 0,
              "hardware_control": False, "storage_status": "LOCAL_ONLY"}
    sim = None
    try:
        assets = Q1Assets(args.root)
        result["assets"] = assets.manifest()
        result["fk"] = check_fk(assets)
        diagnostic_reference(assets, args.output / "diagnostic_only.npz")
        heads = assets.heads()
        lib = MotionLibrary([args.output / "diagnostic_only.npz"], TRACKED, assets.q0,
                            device="cpu", active_indices=assets.active_indices)
        sim = Q1Sim(assets, lib, SimCfg(num_envs=2, threads=2, domain_randomization=False,
                                       observation_noise=False, action_noise=0.), heads, 20261023)
        sim.reset(sim.ids, torch.zeros(2, dtype=torch.long))
        initial = sim.state[0].numpy().copy()
        action = torch.full((2, ACT_DIM), .05)
        model = copy.copy(sim.models[0])
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = initial[:NQ], initial[NQ:]
        data.ctrl[assets.aids] = assets.q0 + action[0].numpy() * assets.scale
        for _ in range(20):
            mujoco.mj_step(model, data)
        actual = sim.step(action)["state"][0].numpy()
        result["physics_control_steps"] += 2
        error = float(np.abs(actual - np.concatenate([data.qpos, data.qvel])).max())
        if error > 1e-5 or not np.isclose(data.time, .02):
            raise AssertionError(f"Q1 native rollout/explicit PD mismatch: {error}")
        result["native_explicit_state_max_abs"] = error
        replay = ReplayBuffer(2, 200, STATE_DIM, ACT_DIM, XI_DIM, "cpu")
        for _ in range(40):
            sim.step(torch.randn(2, ACT_DIM) * .05, replay)
            result["physics_control_steps"] += 2
        if sim.policy_inputs()[0].shape != (2, OBS_DIM*10) or sim.policy_inputs()[1].shape != (2, GOAL_DIM):
            raise AssertionError("Q1 observation/goal dimension mismatch")
        graph_lib = MotionLibrary([args.output / "diagnostic_only.npz"], TRACKED, assets.q0,
                                  device="cpu", dtype=torch.float64, active_indices=assets.active_indices)
        result["graph"] = check_graph(assets, graph_lib, replay)
        result["status"] = "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if sim is not None:
            sim.close()
        root = Path(__file__).parent
        sources = sorted((root / "supermimic").rglob("*.py")) + [Path(__file__)]
        result["source_files"] = [identity(path) for path in sources]
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: value for key, value in result.items() if key not in ("source_files", "assets", "traceback")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
