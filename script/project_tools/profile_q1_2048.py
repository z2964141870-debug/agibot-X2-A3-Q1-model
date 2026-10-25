#!/usr/bin/env python3
"""Measure the declared 2048-environment Q1 collector and full training batches."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_reproduction"))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from bounded_archive_io import bounded_identity as identity
from guard_phuma_batch import MemoryBoundBudget
from runtime_guard import StorageBudget, guarded_run, save
from run_q1_single_motion import CLIP, REFERENCE_SHA


def worker(args):
    import train_reproduction as train
    torch, np = train.torch, train.np
    torch.set_num_threads(2)
    torch.manual_seed(2026102513)
    np.random.seed(2026102513)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assets = train.Q1Assets(args.root)
    reference = args.root / "data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references" / (CLIP + ".npz")
    if identity(reference)["sha256"] != REFERENCE_SHA:
        raise ValueError("Selected existing reference changed")
    lib = train.MotionLibrary([reference], train.TRACKED, assets.q0, device="cuda:0",
                              active_indices=assets.active_indices)
    heads = assets.heads("cuda:0")
    cfg = train.SimCfg(num_envs=2048, threads=args.threads)
    torch.cuda.reset_peak_memory_stats()
    sim = None
    result = {"status": "PROFILE_RUNNING", "source": identity(__file__), "reference": identity(reference),
              "sim": asdict(cfg), "model_slots": args.model_slots, "wm_batch": 1024,
              "policy_batch": 2048, "storage_status": "LOCAL_ONLY"}
    progress = args.output / "progress.json"
    try:
        sim = train.Q1Sim(assets, lib, cfg, heads, 2026102513, slots=args.model_slots)
        buffer = train.ReplayBuffer(2048, 1000, train.STATE_DIM, train.ACT_DIM, train.XI_DIM, "cuda:0")
        wm = train.WorldModel(train.WorldModelCfg(), heads, "cuda:0")
        policy = train.Policy(train.PolicyCfg()).to("cuda:0")
        critic = train.Critic(train.CriticCfg(), pcfg=train.PolicyCfg()).to("cuda:0")
        baseline = args.root / "data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt"
        result["profile_weights"] = identity(baseline)
        checkpoint = torch.load(baseline, map_location="cpu", mmap=True, weights_only=False)
        for name, network in (("wm", wm), ("policy", policy), ("critic", critic)):
            network.load_state_dict(checkpoint[name], strict=True)
        del checkpoint
        result["profile_weights_scope"] = "Existing RF02 weights only for timing; formal training starts from scratch"
        wt = train.WorldModelTrainer(wm, lib, train.WMTrainerCfg(
            batch_size=1024, batches_per_update=10, compile=None))
        pt = train.PolicyTrainer(wm, policy, critic, lib, train.PolicyTrainerCfg(
            batch_size=2048, horizon=32, compile=None))
        result["initial_health"] = train.health()
        result["initial_reference_frames"] = int(torch.unique(sim.frame).numel())
        result["friction_std"] = float(sim.context[:, 0].std())
        result["distinct_randomization_contexts"] = int(torch.unique(sim.context, dim=0).shape[0])
        timings = []
        for step in range(1, 17):
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.no_grad():
                obs, goal = sim.policy_inputs()
                transition = sim.step(policy(obs, goal), buffer)
            torch.cuda.synchronize()
            timings.append(time.monotonic() - started)
            if not bool(torch.isfinite(transition["reward"]).all()):
                raise FloatingPointError("Nonfinite physical reward")
            save(progress, {"step": step, "health": train.health()})
        result["collection_steps"] = 16
        result["real_transitions"] = 16 * 2048
        result["collection_seconds"] = sum(timings)
        result["median_collection_round_seconds"] = float(np.median(timings))
        result["transitions_per_second"] = 16 * 2048 / sum(timings)
        started = time.monotonic()
        result["world_model_metrics"] = wt.update(buffer)
        torch.cuda.synchronize()
        result["world_model_update_seconds"] = time.monotonic() - started
        save(progress, {"phase": "world_model_updated", "health": train.health()})
        started = time.monotonic()
        result["policy_metrics"] = pt.update(buffer)
        torch.cuda.synchronize()
        result["policy_update_seconds"] = time.monotonic() - started
        if not all(np.isfinite(value) for key in ("world_model_metrics", "policy_metrics")
                   for value in result[key].values()):
            raise FloatingPointError("Nonfinite full-batch update")
        result["replay_tensor_bytes"] = sum(value.numel() * value.element_size()
            for value in vars(buffer).values() if isinstance(value, torch.Tensor))
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["final_health"] = train.health()
        result["status"] = "PROFILE_COMPLETED"
        result["limits"] = "Short throughput and full-batch feasibility check, not a learned-policy evaluation."
    except BaseException as exc:
        result.update(status="PROFILE_FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if sim is not None:
            sim.close()
        save(args.output / "receipt.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model-slots", type=int, default=8)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        parser.error("Use a new profile under the unified Q1 root")
    if args.worker:
        return worker(args)
    args.output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget = MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/bones_seed/archives", 100 * 1024 ** 3,
        12 * 1024 ** 3, 250 * 1024 ** 3), 30 * 1024 ** 3, args.expected_oom_kill)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
        "--output", str(args.output), "--threads", str(args.threads), "--model-slots", str(args.model_slots),
        "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    receipt = guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.5,
        grace_s=10., stop_requested=lambda: requested, progress_path=args.output / "progress.json")
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
