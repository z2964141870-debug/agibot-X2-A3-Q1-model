#!/usr/bin/env python3
"""Inspect a saved Q1 checkpoint while collection continues independently."""

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
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    run = args.run.resolve(strict=True)
    output = args.output.resolve()
    checkpoint = (run / args.checkpoint).resolve(strict=True)
    data_root = root / "data/supermimic_for_q1"
    if (not run.is_relative_to(data_root) or not output.is_relative_to(data_root)
            or checkpoint.parent != run or output.exists()):
        raise ValueError("Use an existing Q1 checkpoint and a new Q1 evidence file")
    started = time.monotonic()
    cgroup = Path("/sys/fs/cgroup/memory")
    samples = []

    def health(start=False):
        memory = int((cgroup / "memory.usage_in_bytes").read_text())
        counters = dict(row.split() for row in (cgroup / "memory.oom_control").read_text().splitlines())
        limit = int((28.75 if start else 29.5) * 1024 ** 3)
        if memory >= limit or int(counters["oom_kill"]) != args.expected_oom_kill:
            raise RuntimeError("CPU inspection has insufficient shared memory or observed a new OOM")
        samples.append(memory)
        return {"cgroup_memory_bytes": memory, "oom_kill": int(counters["oom_kill"])}

    before = health(start=True)
    import torch

    torch.set_num_threads(1)
    paths = [run / "step_000000.pt", checkpoint, run / "admission.json"]
    originals = [identity(path) for path in paths]
    initial = torch.load(paths[0], map_location="cpu", mmap=True, weights_only=False)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    health()
    manifest = saved["manifest"]
    step = saved["step"]
    updates, accumulator = 0, 0.
    for number in range(1, step + 1):
        if number >= manifest["args"]["warmup_steps"]:
            accumulator += .4
            if accumulator >= 1.:
                accumulator -= 1.
                updates += 1
    actors = max(0, updates - manifest["args"]["model_warmup_updates"])
    expected = {"wm_opt": updates * manifest["wm_trainer"]["batches_per_update"],
                "policy_opt": actors,
                "critic_opt": actors * manifest["policy_trainer"]["critic_epochs"]
                              * manifest["policy_trainer"]["critic_minibatches"]}
    finite, networks, counters = True, {}, {}
    for name in ("wm", "policy", "critic"):
        first, last = initial[name], saved[name]
        matched = first.keys() == last.keys()
        changes = []
        for key, value in last.items():
            finite = bool(torch.isfinite(value).all()) and finite
            same = key in first and value.shape == first[key].shape
            matched = matched and same
            if same and value.is_floating_point():
                finite = bool(torch.isfinite(first[key]).all()) and finite
                changes.append(float((value - first[key]).abs().max()))
            health()
        networks[name] = {"same_keys_shapes": matched, "changed_tensors": sum(v > 0 for v in changes),
                          "max_abs_change": max(changes, default=0.)}
    for name, count in expected.items():
        states = saved[name]["state"]
        values = sorted({int(state["step"].item()) for state in states.values()})
        for state in states.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    finite = bool(torch.isfinite(value).all()) and finite
            health()
        counters[name] = {"expected": count, "actual_distinct": values,
                          "parameter_states": len(states),
                          "initial_parameter_states": len(initial[name]["state"]),
                          "passed": bool(states) and values == [count] and not initial[name]["state"]}
    finite = finite and all(bool(torch.isfinite(saved[key]).all()) for key in ("bin_return", "bin_memory"))
    # The live writer may have an incomplete trailing line; inspect the saved prefix only.
    lines = (run / "metrics.jsonl").read_bytes().split(b"\n")[:-1]
    prefix = [row for line in lines if line for row in [json.loads(line)] if row["step"] <= step]
    changed = {"source_files": [], "train_clips": []}
    for key, differences in changed.items():
        for record in manifest[key]:
            if identity(record["path"]) != record:
                differences.append(record["path"])
            health()
    checks = {
        "checkpoint_within_budget": 0 < step <= manifest["args"]["steps"],
        "initial_step_zero": initial["step"] == initial["wm_updates"] == initial["policy_updates"] == 0,
        "saved_manifest_matches_initial_and_admission": initial["manifest"] == manifest
            and json.loads(json.dumps(manifest)) == json.loads(paths[2].read_text()),
        "saved_update_counts": saved["wm_updates"] == updates and saved["policy_updates"] == actors,
        "three_networks_changed": all(r["same_keys_shapes"] and r["changed_tensors"] > 0 for r in networks.values()),
        "three_adam_counts": all(r["passed"] for r in counters.values()),
        "saved_tensors_finite": finite,
        "source_and_input_identities_unchanged": not any(changed.values()),
        "metrics_prefix_complete": [r["step"] for r in prefix] == list(range(25, step + 1, 25))
            and bool(prefix) and prefix[-1]["wm_updates"] == updates and prefix[-1]["policy_updates"] == actors,
        "metrics_finite": all(math.isfinite(v) for r in prefix for v in r.values() if isinstance(v, (int, float))),
        "no_new_oom": manifest["baseline_health"]["oom_kill"] == args.expected_oom_kill
            and all(r["health"]["oom_kill"] == args.expected_oom_kill for r in prefix),
        "files_stable": originals == [identity(path) for path in paths],
        "no_cuda_context": not torch.cuda.is_initialized(),
    }
    del initial, saved
    gc.collect()
    for path in paths[:2]:
        with path.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    result = {"status": "CHECKPOINT_EVIDENCE_PASSED" if all(checks.values()) else "CHECKPOINT_EVIDENCE_FAILED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_step": step,
              "collection_transitions": step * manifest["args"]["num_envs"],
              "training_budget": manifest["args"]["steps"], "code_commit": manifest["code_commit"],
              "training_clips": len(manifest["train_clips"]), "checks": checks,
              "wm_updates": updates, "policy_updates": actors, "networks": networks,
              "optimizer_counters": counters, "changed_files": changed, "files": originals,
              "health_before": before, "health_after": health(),
              "max_sampled_cgroup_bytes": max(samples), "seconds": time.monotonic() - started,
              "source": identity(Path(__file__).resolve()), "new_physics_steps": 0,
              "new_optimizer_updates": 0, "tracking_effectiveness_proven": False,
              "storage_status": "LOCAL_ONLY"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
