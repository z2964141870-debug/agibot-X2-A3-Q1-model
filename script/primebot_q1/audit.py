"""CPU-only audit of native Primebot learner, rank shard and Adam counters."""

import argparse
import gc
import json
from pathlib import Path

from common import identity, require, save

OPTIMIZERS = ("model_optimizer", "actor_optimizer", "critic_optimizer")


def audit(run, expected_iteration):
    import torch
    require(not torch.cuda.is_initialized(), "Checkpoint audit must not initialize CUDA")
    path = run / f"checkpoint_{expected_iteration}.pt"
    learner = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    counts = dict(zip(OPTIMIZERS, (expected_iteration * 10, expected_iteration, expected_iteration * 20)))
    require(learner["iteration"] == expected_iteration and learner["actual_optimizer_counts"] == counts,
            "Checkpoint iteration or actual optimizer count differs")
    steps = {}
    finite = 0
    for name in OPTIMIZERS:
        states = learner[name]["state"].values()
        steps[name] = sorted({int(s["step"].item()) for s in states if "step" in s})
        require(steps[name] == ([counts[name]] if expected_iteration else []), "Saved Adam counter differs: " + name)
        for state in states:
            for tensor in state.values():
                if isinstance(tensor, torch.Tensor):
                    require(bool(torch.isfinite(tensor).all()), "Nonfinite Adam state")
    for name in ("model", "actor", "critic", "robot"):
        for tensor in learner[name].values():
            if isinstance(tensor, torch.Tensor):
                require(bool(torch.isfinite(tensor).all()), "Nonfinite saved model or robot buffer")
                finite += 1
    shard = path.parent / learner["rank_directory"] / "rank_0.pt"
    rank = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
    require(rank["snapshot"] == learner["snapshot"] and rank["rank"] == 0,
            "Checkpoint rank transaction identity differs")
    require(rank["local_env_steps"] == learner["global_env_steps"], "Actual transition counters differ")
    replay_bytes = full_replay_bytes = 0
    if rank["replay"] is not None:
        replay = rank["replay"]
        for value in replay.values():
            if isinstance(value, torch.Tensor):
                replay_bytes += value.numel() * value.element_size()
        require(replay["count"] > 0, "Final/profile replay is empty")
        full_replay_bytes = replay_bytes // replay["count"] * replay["capacity"]
    record = dict(checkpoint=identity(path), rank_shard=identity(shard), iteration=expected_iteration,
        global_env_steps=learner["global_env_steps"], actual_optimizer_counts=counts,
        saved_adam_steps=steps, finite_tensor_count=finite, initialized_replay_bytes=replay_bytes,
        full_capacity_replay_bytes=full_replay_bytes, cuda_initialized=False)
    del learner, rank, states, tensor
    gc.collect()
    require(not torch.cuda.is_initialized(), "CPU audit created a CUDA context")
    return record


def validate_profile(run):
    zero = json.loads((run / "zero_snapshot_audit.json").read_text())
    require(zero["rng_before"] == zero["rng_after"] and
            zero["parameters_before"] == zero["parameters_after"], "Zero snapshot altered state or RNG")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    require([r["iteration"] for r in rows] == list(range(1, 9)), "Eight complete consecutive updates required")
    for iteration, row in enumerate(rows, 1):
        require(all(row["actual_" + n] == iteration * k for n, k in zip(OPTIMIZERS, (10, 1, 20))),
                "Passive optimizer counts differ from the full recipe")
    import statistics
    return dict(status="PASSED", initial=audit(run, 0), final=audit(run, 8),
        steady_iteration_seconds=statistics.median(r["update_seconds"] for r in rows[-6:]),
        all_iteration_seconds=[r["update_seconds"] for r in rows], zero_snapshot_audit=zero)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    require(args.profile or args.iteration is not None, "Select an exact audit target")
    result = validate_profile(args.run) if args.profile else dict(status="PASSED", **audit(args.run, args.iteration))
    save(args.output, result)
    print(json.dumps({k: result[k] for k in result if k in ("status", "iteration", "global_env_steps", "steady_iteration_seconds")}))
