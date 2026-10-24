#!/usr/bin/env python3
"""Read a saved R02 checkpoint without simulation or optimizer updates."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, choices=(1000, 5000), default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root.resolve(strict=True)
    experiment = root / "data/supermimic_for_q1/PHUMA_R02_expanded64_H32_12000_20261024"
    run = experiment / "run"
    cgroup = Path("/sys/fs/cgroup/memory")
    samples = []
    started = time.monotonic()

    def sample_memory():
        current = int((cgroup / "memory.usage_in_bytes").read_text())
        counters = dict(line.split() for line in (cgroup / "memory.oom_control").read_text().splitlines())
        if current >= int(29.5 * 1024 ** 3) or int(counters["oom_kill"]) != 14:
            raise RuntimeError("Checkpoint audit reached its memory reserve or observed a new OOM")
        samples.append(current)
        return current

    before = sample_memory()
    if before >= int(28.75 * 1024 ** 3):
        raise RuntimeError("Insufficient shared-memory headroom for the CPU checkpoint audit")
    import torch

    torch.set_num_threads(1)
    sample_memory()
    paths = (run / "step_000000.pt", run / f"step_{args.checkpoint_step:06d}.pt")
    before_files = [identity(path) for path in paths]
    initial = torch.load(paths[0], map_location="cpu", mmap=True, weights_only=False)
    current = torch.load(paths[1], map_location="cpu", mmap=True, weights_only=False)
    sample_memory()
    manifest = current["manifest"]
    spec = json.loads((experiment / "spec.json").read_text())
    admission = json.loads((run / "admission.json").read_text())
    step = current["step"]
    wm_updates = max(0, step - manifest["args"]["warmup_steps"] + 1) * 2 // 5
    actor_updates = max(0, wm_updates - manifest["args"]["model_warmup_updates"])
    expected = {
        "wm_opt": wm_updates * manifest["wm_trainer"]["batches_per_update"],
        "policy_opt": actor_updates,
        "critic_opt": actor_updates * manifest["policy_trainer"]["critic_epochs"]
                      * manifest["policy_trainer"]["critic_minibatches"],
    }
    finite, networks, optimizers = True, {}, {}
    for name in ("wm", "policy", "critic"):
        changes = []
        same_keys = current[name].keys() == initial[name].keys()
        for key, value in current[name].items():
            finite = bool(torch.isfinite(value).all()) and finite
            if key in initial[name]:
                original = initial[name][key]
                finite = bool(torch.isfinite(original).all()) and finite
                if value.is_floating_point() and value.shape == original.shape:
                    changes.append(float((value - original).abs().max()))
                elif value.shape != original.shape:
                    same_keys = False
            sample_memory()
        networks[name] = {
            "same_parameter_keys_and_shapes": same_keys,
            "tensors": len(current[name]), "changed_tensors": sum(delta > 0 for delta in changes),
            "max_abs_change": max(changes, default=0.),
        }
    for name, count in expected.items():
        states = current[name]["state"]
        counters = sorted({int(state["step"].item()) for state in states.values()})
        for state in states.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    finite = bool(torch.isfinite(value).all()) and finite
            sample_memory()
        optimizers[name] = {
            "expected": count, "actual_distinct": counters, "parameter_states": len(states),
            "initial_parameter_states": len(initial[name]["state"]),
            "passed": bool(states) and counters == [count] and not initial[name]["state"],
        }
    # Ignore an in-progress final line from the live append-only metrics file.
    complete_lines = (run / "metrics.jsonl").read_bytes().split(b"\n")[:-1]
    prefix = [json.loads(line) for line in complete_lines if line]
    prefix = [row for row in prefix if row["step"] <= step]
    frozen_changes = []
    for record in spec["frozen_files"]:
        if identity(Path(record["path"])) != record:
            frozen_changes.append(record["path"])
        sample_memory()
    after_files = [identity(path) for path in paths]
    gates = {
        f"checkpoint_{args.checkpoint_step}_within_fixed_budget": step == args.checkpoint_step
            < manifest["args"]["steps"] == spec["steps"] == 12000,
        "initial_step_zero_empty_optimizers": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0
            and all(not initial[name]["state"] for name in expected),
        "saved_manifest_unchanged": initial["manifest"] == manifest and json.loads(json.dumps(manifest)) == admission,
        "phuma_train_split_only": sorted(Path(row["path"]).stem for row in manifest["train_clips"])
            == spec["split_clips"]["train"] and sorted(manifest["heldout_clips"])
            == sorted(spec["split_clips"]["development"] + spec["split_clips"]["test"]),
        "random_initialization": manifest["initialization"] == "RANDOM_FROM_SCRATCH_NO_A3_WEIGHTS"
            and spec["initialization"] == "RANDOM_NO_PARENT_CHECKPOINT",
        "published_code_identity": manifest["code_commit"] == spec["code_commit"]
            == "b6d310897b89d1b8664f6165867bf1256cf49208",
        "local_update_counts": current["wm_updates"] == wm_updates == {1000: 349, 5000: 1949}[args.checkpoint_step]
            and current["policy_updates"] == actor_updates == {1000: 299, 5000: 1899}[args.checkpoint_step],
        "all_optimizer_steps_match": all(row["passed"] for row in optimizers.values()),
        "all_networks_changed": all(row["same_parameter_keys_and_shapes"] and row["changed_tensors"] > 0
                                    for row in networks.values()),
        "saved_network_optimizer_tensors_finite": finite,
        "saved_files_stable": before_files == after_files,
        "all_469_frozen_files_unchanged": not frozen_changes and len(spec["frozen_files"]) == 469,
        "logging_complete_to_checkpoint": [row["step"] for row in prefix] == list(range(25, step + 1, 25))
            and prefix[-1]["wm_updates"] == wm_updates and prefix[-1]["policy_updates"] == actor_updates,
        "logged_metrics_finite": all(math.isfinite(value) for row in prefix for value in row.values()
                                     if isinstance(value, (int, float))),
        "no_new_oom": manifest["baseline_health"]["oom_kill"] == 14
            and all(row["health"]["oom_kill"] == 14 for row in prefix),
        "no_cuda_context": not torch.cuda.is_initialized(),
    }
    del initial, current
    gc.collect()
    for path in paths:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = sample_memory()
    record = {
        "status": "PARTIAL_TRAINING_CHECKPOINT_AUDIT_PASSED" if all(gates.values())
                  else "PARTIAL_TRAINING_CHECKPOINT_AUDIT_FAILED",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": step, "collection_transitions": step * manifest["args"]["num_envs"],
        "wm_updates": wm_updates, "policy_updates": actor_updates, "gates": gates,
        "networks": networks, "optimizer_counters": optimizers, "files": after_files,
        "frozen_changes": frozen_changes, "source": identity(Path(__file__).resolve()),
        "memory_before_bytes": before, "max_sampled_cgroup_bytes": max(samples),
        "memory_after_bytes": after, "memory_guard_bytes": int(29.5 * 1024 ** 3),
        "oom_kill": 14, "elapsed_seconds": time.monotonic() - started,
        "new_physics_steps": 0, "new_policy_calls": 0, "new_optimizer_updates": 0,
        "final_budget_completed": False, "tracking_effectiveness_proven": False,
        "storage_status": "LOCAL_ONLY",
    }
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
