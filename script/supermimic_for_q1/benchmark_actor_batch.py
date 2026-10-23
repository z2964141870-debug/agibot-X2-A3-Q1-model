#!/usr/bin/env python3
"""Bounded H32 actor forward/backward benchmark with no optimizer or physics."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import policy_objective
from supermimic.imagination.rollout import imagine
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.robot_spec import XI_DIM
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def health(device):
    root = Path("/sys/fs/cgroup/memory")
    usage = int((root / "memory.usage_in_bytes").read_text())
    limit = int((root / "memory.limit_in_bytes").read_text())
    counters = dict(line.split() for line in (root / "memory.oom_control").read_text().splitlines())
    temperature = int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu",
                      "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    free, total = torch.cuda.mem_get_info(device)
    if usage >= 30 * 1024 ** 3 or temperature >= 86 or free < 2 * 1024 ** 3:
        raise RuntimeError("Bounded benchmark resource protection")
    return {"cgroup_memory_bytes": usage, "cgroup_limit_bytes": limit,
            "oom_kill": int(counters["oom_kill"]), "gpu_temperature_c": temperature,
            "cuda_free_bytes": free, "cuda_total_bytes": total}


def digest_modules(modules):
    result = {}
    for label, module in modules.items():
        digest = hashlib.sha256()
        for name, value in module.state_dict().items():
            digest.update(name.encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        result[label] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(20261023)
    evaluation = json.loads(args.evaluation.read_text())
    for item in (evaluation["checkpoint"],):
        if identity(item["path"]) != item:
            raise ValueError("Checkpoint changed")
    row = next(r for r in evaluation["rows"] if r["clip"] == "001_walk_front_slow")
    for key in ("reference", "trace"):
        if identity(row[key]["path"]) != row[key]:
            raise ValueError("Saved input changed")
    receipt = {"status": "FAILED", "source": identity(__file__),
               "evaluation": identity(args.evaluation), "checkpoint": evaluation["checkpoint"],
               "trace": row["trace"], "reference": row["reference"],
               "protocol": "Full-width R01 H32 actor objective with critic bootstrap, observation noise and TF32 as in R01. Batch32/128, one warmup then two measured trials each; no optimizer instantiated or called.",
               "new_physics_steps": 0, "optimizer_updates": 0, "trials": [],
               "limits": "Actor portion only; excludes replay allocation, WM/critic updates and optimizer state. Resource feasibility is not a learning result.",
               "storage_status": "LOCAL_ONLY"}
    started = time.monotonic()
    try:
        receipt["initial_health"] = health(args.device)
        ck = torch.load(evaluation["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
        assets = Q1Assets(args.root)
        if assets.manifest() != ck["manifest"]["assets"]:
            raise ValueError("R01 adapter changed")
        config = dict(ck["manifest"]["world_model"])
        config["state_spec"] = StateSpecCfg(**config["state_spec"])
        wm = WorldModel(WorldModelCfg(**config), assets.heads(args.device), args.device).eval()
        pcfg = PolicyCfg(**ck["manifest"]["policy"])
        policy = Policy(pcfg).to(args.device).train()
        critic = Critic(CriticCfg(**ck["manifest"]["critic"]), pcfg=pcfg).to(args.device).eval()
        for module, key in ((wm, "wm"), (policy, "policy"), (critic, "critic")):
            module.load_state_dict(ck[key], strict=True)
        del ck
        wm.requires_grad_(False)
        critic.requires_grad_(False)
        modules = {"wm": wm, "policy": policy, "critic": critic}
        receipt["parameters_before"] = digest_modules(modules)
        lib = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                            device=args.device, active_indices=assets.active_indices)
        with np.load(row["reference"]["path"], allow_pickle=False) as ref:
            initial = np.concatenate([ref["qpos"][0], ref["qvel"][0]])
        with np.load(row["trace"]["path"], allow_pickle=False) as trace:
            states = torch.as_tensor(np.concatenate([initial[None], trace["state"]]), device=args.device)
            actions = torch.as_tensor(trace["actions"].copy(), device=args.device).clamp(-20., 20.)
        rng = np.random.default_rng(20261023)
        indices = rng.integers(10, len(actions) - 32 + 1, size=128)
        receipt["sampled_frames"] = indices.tolist()
        for batch in (32, 128):
            frames = torch.as_tensor(indices[:batch], device=args.device)
            history = frames[None] + torch.arange(-9, 1, device=args.device)[:, None]
            context = torch.zeros(batch, XI_DIM, device=args.device)
            context[:, 0] = 1.
            start = StartBatch(states[frames], context, frames, actions[frames - 1],
                               states[history], actions[history - 1])
            for trial in range(3):
                current = health(args.device)
                if current["oom_kill"] != receipt["initial_health"]["oom_kill"]:
                    raise RuntimeError("New container OOM event")
                policy.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                tick = time.monotonic()
                ro = imagine(wm, policy, lib, start, 32, obs_noise=True)
                feat = wm.spec.canonical(ro.states.reshape(33 * batch, -1)).reshape(33, batch, -1)
                terminal = critic(feat[-1], ro.goals[-1], wm.spec.normalize_xi(start.xi), ro.e_root[-1])
                loss = -policy_objective(ro, terminal, .99) / 32
                loss.backward()
                torch.cuda.synchronize(args.device)
                seconds = time.monotonic() - tick
                grads = [p.grad for p in policy.parameters() if p.grad is not None]
                finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in grads)
                norm = float(sum(g.double().square().sum() for g in grads).sqrt())
                frozen = all(p.grad is None for module in (wm, critic) for p in module.parameters())
                if not finite or not frozen or not np.isfinite(norm) or norm <= 0:
                    raise AssertionError("Actor gradient contract failed")
                result = {"batch": batch, "trial": trial, "warmup": trial == 0,
                          "seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
                          "loss": float(loss.detach()), "gradient_norm": norm,
                          "gradients_finite": finite, "wm_critic_parameter_gradients_absent": frozen}
                receipt["trials"].append(result)
                print(json.dumps(result), flush=True)
                del ro, feat, terminal, loss, grads
            del start
        policy.zero_grad(set_to_none=True)
        receipt["parameters_after"] = digest_modules(modules)
        if receipt["parameters_before"] != receipt["parameters_after"]:
            raise AssertionError("Benchmark changed module state")
        receipt["final_health"] = health(args.device)
        if receipt["final_health"]["oom_kill"] != receipt["initial_health"]["oom_kill"]:
            raise RuntimeError("New container OOM event")
        receipt["status"] = "ACTOR_BATCH_RESOURCE_BENCHMARK_COMPLETED"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        with args.output.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in receipt.items() if k not in ("trials", "sampled_frames")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
