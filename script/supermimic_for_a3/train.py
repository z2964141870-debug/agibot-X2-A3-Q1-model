#!/usr/bin/env python3
"""Train an A3 SuperMimic tracker with real MuJoCo transitions and imagined BPTT."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.a3_sim import A3Sim, SimCfg
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg


HELDOUT = ("004_walk_frontleft_slow", "048_balance_right_singleleg", "056_lunge_side_alternating", "080_reach_front_both")


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def health():
    temp = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    temp = int(temp.splitlines()[0])
    if temp >= 86:
        raise RuntimeError(f"GPU temperature protection: {temp} C")
    memfile = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    mem = int(memfile.read_text()) if memfile.exists() else 0
    if mem >= 30 * 1024 ** 3:
        raise RuntimeError(f"Container memory protection: {mem} bytes")
    return {"gpu_temperature_c": temp, "cgroup_memory_bytes": mem}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend-receipt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--wm-batch", type=int, default=128)
    p.add_argument("--wm-batches", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--model-warmup-updates", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1404)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clips", nargs="+", default=None)
    args = p.parse_args()
    receipt = json.loads(args.backend_receipt.read_text())
    if receipt["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED":
        raise ValueError("A3 backend has not passed numerical and gradient verification")
    for source in receipt["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError(f"Verified backend source changed: {source['path']}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assets = A3Assets(args.root)
    files = sorted(args.motions.glob("*.npz"))
    files = [f for f in files if f.stem not in HELDOUT and (args.clips is None or f.stem in args.clips)]
    if not files:
        raise FileNotFoundError("No training clips")
    source_manifest = json.loads((args.motions / "manifest.json").read_text())
    source_by_path = {m["output"]["path"]: m for m in source_manifest["motions"]}
    for path in files:
        record = source_by_path[str(path.resolve())]
        if identity(path)["sha256"] != record["output"]["sha256"]:
            raise ValueError("Motion hash mismatch")
    heads = assets.heads(args.device)
    lib = MotionLibrary(files, TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
    scfg = SimCfg(num_envs=args.num_envs)
    sim = A3Sim(assets, lib, scfg, heads, args.seed)
    buffer = ReplayBuffer(args.num_envs, 1000, 83, 29, 33, args.device)
    wcfg = WorldModelCfg()
    pcfg = PolicyCfg()
    ccfg = CriticCfg()
    wtcfg = WMTrainerCfg(batch_size=args.wm_batch, batches_per_update=args.wm_batches,
                        compile="default" if args.compile else None)
    ptcfg = PolicyTrainerCfg(batch_size=args.batch, compile="default" if args.compile else None)
    wm = WorldModel(wcfg, heads, args.device)
    policy = Policy(pcfg).to(args.device)
    critic = Critic(ccfg, pcfg=pcfg).to(args.device)
    wt = WorldModelTrainer(wm, lib, wtcfg)
    pt = PolicyTrainer(wm, policy, critic, lib, ptcfg)
    manifest = {"status": "ADMITTED_NOT_TRAINED", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "assets": assets.manifest(), "sim": asdict(scfg), "world_model": asdict(wcfg),
                "policy": asdict(pcfg), "critic": asdict(ccfg), "wm_trainer": asdict(wtcfg), "policy_trainer": asdict(ptcfg),
                "train_clips": [identity(f) for f in files], "heldout_clips": list(HELDOUT),
                "backend_receipt": identity(args.backend_receipt), "training_entry": identity(__file__),
                "library_summary": lib.summary(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
                "paper_deviations": ["CPU MuJoCo backend; implicit PD with original A3 gains and contact geometry",
                    "Full A3 state including head and passive feet; active 29-joint actor",
                    "Smaller environment and update batch on shared RTX3090", "Pelvis reward anchor",
                    "No pose/velocity reset noise in this first adaptation", "Model-only warmup before BPTT"]}
    atomic_json(args.output / "admission.json", manifest)
    step = updates = policy_updates = 0
    started = time.monotonic()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save(name):
        destination = args.output / name
        tmp = destination.with_suffix(".tmp")
        torch.save({"step": step, "wm_updates": updates, "policy_updates": policy_updates,
                    "wm": wm.state_dict(), "policy": policy.state_dict(), "critic": critic.state_dict(),
                    "wm_opt": wt.opt.state_dict(), "policy_opt": pt.opt.state_dict(), "critic_opt": pt.critic_opt.state_dict(),
                    "bin_return": sim.bin_return, "bin_memory": sim.bin_memory,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "manifest": manifest}, tmp)
        os.replace(tmp, destination)

    save("step_000000.pt")
    atomic_json(args.output / "status.json", {"status": "RUNNING", "step": 0, "pid": os.getpid(), "health": health()})
    status = "FAILED"
    error = None
    accumulated = 0.
    last_metrics = {}
    try:
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                obs, goal = sim.policy_inputs()
                result = sim.step(policy(obs, goal), buffer)
            if step >= args.warmup_steps:
                accumulated += .4
                if accumulated >= 1.:
                    accumulated -= 1.
                    lr = 2e-4 * (.03 + .97 * .5 * (1 + math.cos(math.pi * min(step / 250000., 1.))))
                    wt.set_lr(lr)
                    pt.set_lr(lr)
                    wm_metrics = wt.update(buffer)
                    updates += 1
                    if updates > args.model_warmup_updates:
                        last_metrics = pt.update(buffer)
                        policy_updates += 1
                    if not all(np.isfinite(v) for v in [*wm_metrics.values(), *last_metrics.values()]):
                        raise RuntimeError("Nonfinite world-model, actor or critic metrics")
                    last_metrics = {**last_metrics, **{"wm/" + k: v for k, v in wm_metrics.items()}}
            if step % 25 == 0:
                row = {"status": "RUNNING", "step": step, "env_steps": step * args.num_envs,
                       "wm_updates": updates, "policy_updates": policy_updates,
                       "seconds": time.monotonic() - started, "pid": os.getpid(),
                       "reward_mean": float(result["reward"].mean()), "termination_fraction": float(result["terminated"].float().mean()),
                       "joint_rmse_rad": float(result["joint_rmse"].mean()), "health": health(), **last_metrics}
                atomic_json(args.output / "status.json", row)
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if step % args.checkpoint_every == 0:
                save(f"step_{step:06d}.pt")
            if stop_requested:
                status = "STOPPED_SIGNAL"
                break
        else:
            status = "TRAINING_FINISHED_PENDING_EVALUATION"
        save("final.pt")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sim.close()
        result = {"status": status, "step": step, "env_steps": step * args.num_envs,
                  "wm_updates": updates, "policy_updates": policy_updates,
                  "seconds": time.monotonic() - started, "error": error,
                  "artifacts": [identity(f) for f in sorted(args.output.glob("*.pt"))],
                  "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
        atomic_json(args.output / "receipt.json", result)
        atomic_json(args.output / "status.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
