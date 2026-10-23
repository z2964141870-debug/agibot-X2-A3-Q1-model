#!/usr/bin/env python3
"""Check H8 actor gradients on one saved state without physics or optimization."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import ImagStep
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.diagnostic / "plan.json").read_text())
    if identity(plan["checkpoint"]["path"]) != plan["checkpoint"]:
        raise ValueError("Source checkpoint changed")
    torch.set_num_threads(2)
    torch.manual_seed(8723)
    device = "cuda:0"
    ck = torch.load(plan["checkpoint"]["path"], map_location="cpu", weights_only=False, mmap=True)
    assets = A3Assets(plan["root"])
    if assets.manifest() != ck["manifest"]["assets"]:
        raise ValueError("Source asset contract changed")
    heads = assets.heads(device)
    sample = plan["samples"][0]
    for key in ("reference", "trace"):
        if identity(sample[key]["path"]) != sample[key]:
            raise ValueError("Saved input changed")
    lib = MotionLibrary([Path(sample["reference"]["path"])], TRACKED, assets.q0,
                        device=device, active_indices=assets.active_indices)
    with np.load(sample["trace"]["path"], allow_pickle=False) as raw:
        states = torch.tensor(raw["state"], device=device)
        actions = torch.tensor(raw["actions"], device=device).clamp(-20, 20)
    t = sample["start"]
    xi = torch.zeros(1, 33, device=device)
    xi[:, 0] = 1.
    start = StartBatch(state=states[t - 1:t], xi=xi, frame=torch.tensor([t], device=device),
                       prev_action=actions[t - 1:t], hist_states=states[t - 10:t, None],
                       hist_actions=actions[t - 10:t, None])
    # Post-transition state row t-1 is state s_t; history actions precede each state.
    wcfg = dict(ck["manifest"]["world_model"])
    wcfg["state_spec"] = StateSpecCfg(**wcfg["state_spec"])
    wm = WorldModel(WorldModelCfg(**wcfg), heads, device)
    pcfg = PolicyCfg(**ck["manifest"]["policy"])
    policy = Policy(pcfg).to(device)
    critic = Critic(CriticCfg(**ck["manifest"]["critic"]), pcfg=pcfg).to(device)
    for name, module in (("wm", wm), ("policy", policy), ("critic", critic)):
        module.load_state_dict(ck[name], strict=True)
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    ro = imagine(wm, policy, lib, start, 8, obs_noise=False,
                 step_fn=ImagStep(wm, lib.default_joint_pos, compile_mode=None))
    feat = wm.spec.canonical(ro.states[-1])
    value = critic(feat, ro.goals[-1], wm.spec.normalize_xi(xi), ro.e_root[-1])
    loss = -policy_objective(ro, value, .99) / 8
    loss.backward()
    gradients = [p.grad for p in policy.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Nonfinite or absent actor gradient")
    norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
    if norm <= 0 or any(p.grad is not None for p in wm.parameters()):
        raise ValueError("Missing actor gradient or leaked world-model gradient")
    for name, module in (("wm", wm), ("policy", policy), ("critic", critic)):
        if not all(torch.equal(value.cpu(), ck[name][key]) for key, value in module.state_dict().items()):
            raise ValueError("Gradient-only control changed model parameters")
    result = {"status": "H8_FROZEN_MODEL_ACTOR_GRADIENT_PASSED", "entry": identity(__file__),
              "checkpoint": plan["checkpoint"], "diagnostic_plan": identity(args.diagnostic / "plan.json"),
              "sample": sample["sample"], "start": t, "horizon": 8, "batch_size": 1,
              "loss": float(loss.detach()), "actor_gradient_norm": norm,
              "actor_gradient_tensors": len(gradients), "wm_gradient_tensors": 0,
              "all_network_tensors_unchanged": True, "physics_steps": 0, "optimizer_updates": 0,
              "interpretation": "Eager H8 graph/critic bootstrap control, not compiled full-batch training or physical success."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
