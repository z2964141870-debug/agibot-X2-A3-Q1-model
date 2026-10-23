#!/usr/bin/env python3
"""Extend an audited Q1 batch-pair arm with a fixed additional interaction budget."""

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

import initialization

CORE = Path(__file__).resolve().parents[1] / "supermimic_for_q1"
sys.path.insert(0, str(CORE))

from contact_protocol import ContactCorrectedAssets
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.q1_assets import TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg
from run_batch_pair import reclaim_checkpoint_cache


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def health(root, baseline_oom=None):
    from run_batch_pair import health as sample

    reclaim_checkpoint_cache(root)
    return sample(root, baseline_oom)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.steps < 3000 or args.steps % 500 or len(args.code_commit) != 40:
        parser.error("Use at least 3000 additional rounds, a multiple of 500, and a frozen release commit")
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = initialization.validated_parent(args.pair, args.parent_run)
    parent = json.loads((args.parent_run / "admission.json").read_text())
    if args.seed == parent["args"]["seed"]:
        raise ValueError("Fresh replay continuation requires an explicit different seed")
    baseline = health(root)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assets = ContactCorrectedAssets(root)
    backend = json.loads(Path(parent["backend_receipt"]["path"]).read_text())
    if backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED" or backend["assets"] != assets.manifest():
        raise ValueError("Corrected backend contract changed")
    for item in backend["source_files"] + [parent["backend_receipt"], parent["reference_receipt"], parent["motion_manifest"]]:
        initialization.check_identity(item)
    files = [Path(item["path"]) for item in parent["train_clips"]]
    for item in parent["train_clips"]:
        initialization.check_identity(item)
    heads = assets.heads(args.device)
    lib = MotionLibrary(files, TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
    scfg = SimCfg(**parent["sim"])
    sim = Q1Sim(assets, lib, scfg, heads, args.seed)
    try:
        buffer = ReplayBuffer(scfg.num_envs, 1000, STATE_DIM, ACT_DIM, XI_DIM, args.device)
        wm_config = {**parent["world_model"], "state_spec": StateSpecCfg(**parent["world_model"]["state_spec"])}
        wcfg, pcfg, ccfg = WorldModelCfg(**wm_config), PolicyCfg(**parent["policy"]), CriticCfg(**parent["critic"])
        wtcfg, ptcfg = WMTrainerCfg(**parent["wm_trainer"]), PolicyTrainerCfg(**parent["policy_trainer"])
        wm = WorldModel(wcfg, heads, args.device)
        policy, critic = Policy(pcfg).to(args.device), Critic(ccfg, pcfg=pcfg).to(args.device)
        wt, pt = WorldModelTrainer(wm, lib, wtcfg), PolicyTrainer(wm, policy, critic, lib, ptcfg)
        manifest = {key: parent[key] for key in initialization.CONTRACT_KEYS}
        manifest.update({"status": "ADMITTED_NOT_TRAINED", "assets": assets.manifest(), "sim": asdict(scfg),
            "world_model": asdict(wcfg), "policy": asdict(pcfg), "critic": asdict(ccfg),
            "wm_trainer": asdict(wtcfg), "policy_trainer": asdict(ptcfg),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "code_commit": args.code_commit, "baseline_health": baseline,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "training_entry": identity(__file__),
            "source_files": [identity(p) for p in sorted([*CORE.rglob("*.py"), *Path(__file__).parent.rglob("*.py")])],
            "warmup_steps": parent["args"]["warmup_steps"],
            "model_warmup_updates": parent["args"]["model_warmup_updates"],
            "paper_deviations": parent["paper_deviations"] + ["Continuation uses a fresh replay and episodes; parent networks, Adam states and return bins retained."]})
        manifest["initialization"] = initialization.load_parent(evidence, manifest,
            (wm, policy, critic), (wt.opt, pt.opt, pt.critic_opt), sim)
        manifest["learning_rate_step_offset"] = manifest["initialization"]["learning_rate_step_offset"]
        gc.collect()
        health(root, baseline["oom_kill"])
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "admission.json", manifest)
        step = updates = policy_updates = 0
        started = time.monotonic()
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def save(name):
            destination = args.output / name
            temporary = destination.with_suffix(".tmp")
            torch.save({"step": step, "wm_updates": updates, "policy_updates": policy_updates,
                "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                "wm": wm.state_dict(), "policy": policy.state_dict(), "critic": critic.state_dict(),
                "wm_opt": wt.opt.state_dict(), "policy_opt": pt.opt.state_dict(), "critic_opt": pt.critic_opt.state_dict(),
                "bin_return": sim.bin_return, "bin_memory": sim.bin_memory,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "manifest": manifest}, temporary)
            os.replace(temporary, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

        status, error = "FAILED", None
        try:
            save("step_000000.pt")
            sim.reset(sim.ids)
            accumulated = 0.
            last_metrics = {}
            for step in range(1, args.steps + 1):
                with torch.no_grad():
                    obs, goal = sim.policy_inputs()
                    result = sim.step(policy(obs, goal), buffer)
                if step >= manifest["warmup_steps"]:
                    accumulated += .4
                    if accumulated >= 1.:
                        accumulated -= 1.
                        scheduled = manifest["learning_rate_step_offset"] + step
                        lr = 2e-4 * (.03 + .97 * .5 * (1 + math.cos(math.pi * min(scheduled / 250000., 1.))))
                        wt.set_lr(lr)
                        pt.set_lr(lr)
                        wm_metrics = wt.update(buffer)
                        updates += 1
                        if updates > manifest["model_warmup_updates"]:
                            last_metrics = pt.update(buffer)
                            policy_updates += 1
                        if not all(np.isfinite(v) for v in [*wm_metrics.values(), *last_metrics.values()]):
                            raise RuntimeError("Nonfinite continuation training metrics")
                        last_metrics = {**last_metrics, **{"wm/" + k: v for k, v in wm_metrics.items()}}
                if step % 25 == 0:
                    row = {"status": "RUNNING", "step": step, "env_steps": step * scfg.num_envs,
                        "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                        "wm_updates": updates, "policy_updates": policy_updates,
                        "seconds": time.monotonic() - started, "pid": os.getpid(),
                        "reward_mean": float(result["reward"].mean()),
                        "termination_fraction": float(result["terminated"].float().mean()),
                        "joint_rmse_rad": float(result["joint_rmse"].mean()),
                        "health": health(root, baseline["oom_kill"]), **last_metrics}
                    atomic_json(args.output / "status.json", row)
                    with (args.output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                if step % 500 == 0:
                    save(f"step_{step:06d}.pt")
                if stop:
                    status = "STOPPED_SIGNAL"
                    break
            else:
                status = "TRAINING_FINISHED_PENDING_EVALUATION"
            save("final.pt")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            receipt = {"status": status, "step": step, "env_steps": step * scfg.num_envs,
                "wm_updates": updates, "policy_updates": policy_updates,
                "total_collection_rounds": manifest["learning_rate_step_offset"] + step,
                "seconds": time.monotonic() - started, "error": error,
                "artifacts": [identity(p) for p in sorted(args.output.glob("*.pt"))],
                "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
            atomic_json(args.output / "receipt.json", receipt)
            atomic_json(args.output / "status.json", receipt)
            print(json.dumps(receipt, allow_nan=False), flush=True)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
