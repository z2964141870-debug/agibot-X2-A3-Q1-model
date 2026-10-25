#!/usr/bin/env python3
"""Train the Q1 SuperMimic adaptation without local physical reference gates."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from functools import partial

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "supermimic_for_q1"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_expanded"))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_stability"))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))

from supermimic.q1_assets import TRACKED, identity
from contact_protocol import ContactCorrectedAssets as Q1Assets
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg
from slotted_sim import SlottedQ1Sim
from bounded_policy import BoundedPolicy, GuardedPolicyTrainer, GuardedWorldModelTrainer
from relocated_io import JsonView, migration_view
from reference_admission import POLICY, validate_arrays
from collection_budget import CollectionBudget

Q1Sim = partial(SlottedQ1Sim, slots=4)
Policy, PolicyTrainer, WorldModelTrainer = BoundedPolicy, GuardedPolicyTrainer, GuardedWorldModelTrainer


HELDOUT = ()


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
    oom = Path("/sys/fs/cgroup/memory/memory.oom_control")
    counters = dict(line.split() for line in oom.read_text().splitlines()) if oom.exists() else {}
    return {"gpu_temperature_c": temp, "cgroup_memory_bytes": mem,
            "oom_kill": int(counters.get("oom_kill", 0))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--motions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend-receipt", type=Path, required=True)
    p.add_argument("--reference-receipt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--training-seconds", type=float)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--model-slots", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--wm-batch", type=int, default=64)
    p.add_argument("--wm-batches", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=128)
    p.add_argument("--model-warmup-updates", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clips", nargs="+", default=None)
    p.add_argument("--code-commit", required=True)
    args = p.parse_args()
    budget = CollectionBudget(args.steps, args.training_seconds)
    if args.steps < 300 or args.num_envs < 1 or len(args.code_commit) != 40:
        raise ValueError("An explicit training budget and published code identity are required")
    view = migration_view(Path(args.root))
    receipts = JsonView(view)
    receipt = receipts.loads(args.backend_receipt.read_text())
    if receipt["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED":
        raise ValueError("Q1 backend has not passed its own numerical and gradient verification")
    for source in receipt["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError(f"Verified backend source changed: {source['path']}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    baseline_health = health()
    assets = Q1Assets(args.root)
    files = sorted(args.motions.glob("*.npz"))
    files = [f for f in files if f.stem not in HELDOUT and (args.clips is None or f.stem in args.clips)]
    if not files:
        raise FileNotFoundError("No training clips")
    source_manifest = json.loads((args.motions / "manifest.json").read_text())
    reference_receipt = json.loads(args.reference_receipt.read_text())
    if (reference_receipt["status"] != "Q1_REFERENCE_IO_ADMISSION_COMPLETED"
            or reference_receipt["admission_policy"] != POLICY
            or source_manifest["admission_policy"] != POLICY):
        raise ValueError("The user-authorized reproduction reference policy is required")
    if identity(args.motions / "manifest.json")["sha256"] != reference_receipt["input_manifest"]["sha256"]:
        raise ValueError("Reference checks belong to a different dataset")
    reference_checks = {row["clip"]: row for row in reference_receipt["clips"]}
    source_by_path = {m["output"]["path"]: m for m in source_manifest["motions"]}
    action_coverage = {}
    for path in files:
        record = source_by_path[str(path.resolve())]
        if not record["admitted_for_training"] or not reference_checks[path.stem]["io_valid"]:
            raise ValueError("Q1 reference arrays could not be loaded")
        if identity(path)["sha256"] != record["output"]["sha256"]:
            raise ValueError("Motion hash mismatch")
        validate_arrays(path)
        with np.load(path, allow_pickle=False) as reference:
            required_action = np.abs((reference["joint_pos"] - assets.q0) / assets.scale)
        action_coverage[path.stem] = float(required_action.max())
    heads = assets.heads(args.device)
    lib = MotionLibrary(files, TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
    scfg = SimCfg(num_envs=args.num_envs, threads=args.threads)
    sim = Q1Sim(assets, lib, scfg, heads, args.seed, slots=args.model_slots)
    buffer = ReplayBuffer(args.num_envs, 1000, STATE_DIM, ACT_DIM, XI_DIM, args.device)
    wcfg = WorldModelCfg()
    pcfg = PolicyCfg()
    ccfg = CriticCfg()
    wtcfg = WMTrainerCfg(batch_size=args.wm_batch, batches_per_update=args.wm_batches,
                        compile="default" if args.compile else None)
    ptcfg = PolicyTrainerCfg(batch_size=args.batch, horizon=32,
                             compile="default" if args.compile else None)
    wm = WorldModel(wcfg, heads, args.device)
    policy = Policy(pcfg).to(args.device)
    critic = Critic(ccfg, pcfg=pcfg).to(args.device)
    wt = WorldModelTrainer(wm, lib, wtcfg)
    pt = PolicyTrainer(wm, policy, critic, lib, ptcfg)
    manifest = {"status": "ADMITTED_NOT_TRAINED", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "assets": assets.manifest(), "sim": asdict(scfg), "world_model": asdict(wcfg),
                "policy": asdict(pcfg), "critic": asdict(ccfg), "wm_trainer": asdict(wtcfg), "policy_trainer": asdict(ptcfg),
                "train_clips": [identity(f) for f in files], "heldout_clips": list(HELDOUT),
                "reference_max_required_normalized_actions": action_coverage,
                "backend_receipt": identity(args.backend_receipt), "training_entry": identity(__file__),
                "reference_receipt": identity(args.reference_receipt),
                "library_summary": lib.summary(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
                "code_commit": args.code_commit, "baseline_health": baseline_health,
                "motion_manifest": identity(args.motions / "manifest.json"),
                "source_files": [identity(path) for path in sorted({*CORE.rglob("*.py"),
                    *HERE.rglob("*.py"), HERE.parent / "supermimic_q1_expanded/slotted_sim.py",
                    HERE.parent / "supermimic_q1_stability/bounded_policy.py"})],
                "reference_admission_policy": POLICY,
                "local_geometry_gates_used_for_admission": False,
                "relocation": view.evidence(),
                "paper_deviations": ["CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds",
                    "Q1 free-root 57-state and 22-joint actor; no artificial support",
                    "Smaller environment and update batch on shared RTX3090", "Pelvis reward anchor",
                    "No pose/velocity reset noise in this first adaptation", "Model-only warmup before BPTT",
                    "BONES G1 references retargeted to Q1; first finite batch, not the paper full G1 corpus",
                    "Original dataset-wide filename and G1-specific geometric filtering is not yet replicated for the full archive",
                    "Local 90% support/30mm gap/3mm collision/2rad-s gates are not training requirements"]}
    manifest["initialization"] = "RANDOM_FROM_SCRATCH_NO_A3_WEIGHTS"
    manifest["learning_rate_step_offset"] = 0
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
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    save("step_000000.pt")
    collection_started = time.monotonic()
    collection_seconds = 0.
    status = "FAILED"
    error = None
    accumulated = 0.
    last_metrics = {}
    try:
        atomic_json(args.output / "status.json", {"status": "RUNNING", "step": 0, "pid": os.getpid(), "health": health()})
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                obs, goal = sim.policy_inputs()
                result = sim.step(policy(obs, goal), buffer)
            if step >= args.warmup_steps:
                accumulated += .4
                if accumulated >= 1.:
                    accumulated -= 1.
                    schedule_step = manifest["learning_rate_step_offset"] + step
                    lr = 2e-4 * (.03 + .97 * .5 * (1 + math.cos(math.pi * min(schedule_step / 250000., 1.))))
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
                sample_health = health()
                if sample_health["oom_kill"] > baseline_health["oom_kill"]:
                    raise RuntimeError("New container OOM-kill event since Q1 admission")
                row = {"status": "RUNNING", "step": step, "env_steps": step * args.num_envs,
                       "wm_updates": updates, "policy_updates": policy_updates,
                       "seconds": time.monotonic() - started, "pid": os.getpid(),
                       "reward_mean": float(result["reward"].mean()), "termination_fraction": float(result["terminated"].float().mean()),
                       "joint_rmse_rad": float(result["joint_rmse"].mean()), "health": sample_health, **last_metrics}
                atomic_json(args.output / "status.json", row)
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
            if step % args.checkpoint_every == 0:
                save(f"step_{step:06d}.pt")
            if stop_requested:
                status = "STOPPED_SIGNAL"
                break
            if budget.finished(step, time.monotonic() - collection_started):
                status = "TRAINING_FINISHED_PENDING_EVALUATION"
                break
        else:
            status = "TRAINING_FINISHED_PENDING_EVALUATION"
        collection_seconds = time.monotonic() - collection_started
        save("final.pt")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if not collection_seconds:
            collection_seconds = time.monotonic() - collection_started
        sim.close()
        result = {"status": status, "step": step, "env_steps": step * args.num_envs,
                  "wm_updates": updates, "policy_updates": policy_updates,
                  "training_seconds": collection_seconds,
                  "requested_training_seconds": args.training_seconds,
                  "seconds": time.monotonic() - started, "error": error,
                  "artifacts": [identity(f) for f in sorted(args.output.glob("*.pt"))],
                  "storage_status": "LOCAL_ONLY", "not_hardware_validated": True}
        atomic_json(args.output / "receipt.json", result)
        atomic_json(args.output / "status.json", result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
