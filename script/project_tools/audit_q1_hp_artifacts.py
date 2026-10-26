"""Verify saved HP-Q1 training artifacts without policy or physics execution."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def finite_numbers(value):
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def audit(args):
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(1)
    trial = args.trial.resolve(strict=True)
    run = trial / "run"
    plan, receipt, guard = (read(trial / "plan.json"), read(run / "receipt.json"),
                            read(trial / "train_guard.json"))
    configuration = read(run / "args.json")
    contract = read(run / "contract.json")
    cfg = OmegaConf.load(run / "config.yaml")
    library = read(plan["library"]["path"])
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("worker_completed_and_reaped", receipt["status"] == "TRAINING_COMPLETED"
          and guard["status"] == "WORKER_COMPLETED"
          and guard["cleanup"]["worker_reaped"]
          and guard["cleanup"]["worker_return_code"] == 0)
    check("actual_training_within_frozen_budget",
          plan["soft_training_seconds"] <= receipt["training_seconds"]
          <= plan["maximum_training_seconds"], receipt["training_seconds"])
    check("published_source_commit", receipt["code_commit"] == plan["code_commit"]
          == args.expected_commit)
    changes = [entry["path"] for entry in plan["source"]
               if identity(entry["path"]) != entry]
    check("all_frozen_source_assets_unchanged", not changes,
          {"files": len(plan["source"]), "changed": changes})
    for name in ("library", "runtime_source", "config", "q1_adapter"):
        check("frozen_" + name, identity(plan[name]["path"]) == plan[name])
    splits = library["splits"]
    train, validation = splits["train"], splits["validation"]
    check("expected_split_sizes", (len(train), len(validation))
          == (args.train_clips, args.validation_clips))
    actors = {name: {row["actor_uid"] for row in values}
              for name, values in splits.items()}
    check("actors_disjoint", not actors["train"].intersection(actors["validation"]),
          {name: len(values) for name, values in actors.items()})
    rows = train + validation
    check("source_identities_unique", len({row["name"] for row in rows}) == len(rows))
    changed_inputs = [row["path"] for row in rows
                      if identity(row["path"])["sha256"] != row["sha256"]]
    check("all_reference_inputs_unchanged", not changed_inputs,
          {"clips": len(rows), "changed": changed_inputs})
    expected = {"wm_batch": 1024, "wm_batches": 10, "pi_batch": 1024,
                "horizon": 32, "utd": .4, "hist": 10, "goal_frames": 10,
                "wm_only_iters": 500, "warmup_steps": 64, "critic_epochs": 5,
                "critic_minibatches": 4, "amp": "bf16", "compile": True, "seed": 0}
    check("actual_4096_gpu_configuration", contract["num_envs"] == 4096
          and int(cfg.algo.num_envs) == 4096
          and all(configuration[key] == value for key, value in expected.items()),
          {"num_envs": contract["num_envs"], **{key: configuration[key]
                                                       for key in expected}})
    n = receipt["iteration"]
    check("actual_optimizer_group_counts", receipt["wm_updates"]
          == n * configuration["wm_batches"] and receipt["actor_updates"]
          == max(0, n - configuration["wm_only_iters"]))
    expected_transitions = (configuration["warmup_steps"]
                            + int(n / configuration["utd"])) * contract["num_envs"]
    check("actual_collection_count", receipt["sim_steps"] == expected_transitions,
          {"actual": receipt["sim_steps"], "expected": expected_transitions})
    saved = {name: torch.load(run / (name + ".pt"), map_location="cpu",
                             weights_only=True, mmap=True)
             for name in ("initial", "final", "last")}
    first, final, last = (saved[name] for name in ("initial", "final", "last"))
    check("from_zero_initialization", first["iteration"] == first["sim_steps"] == 0
          and not any(key.startswith("opt_") for key in first))
    check("checkpoint_counters_match", all(s["iteration"] == n
          and s["sim_steps"] == receipt["sim_steps"]
          and s["wall_time"] == receipt["training_seconds"] for s in (final, last)))
    for name, saved_value in saved.items():
        count, elements, finite = 0, 0, True
        for tensor in tensors(saved_value):
            count += 1
            elements += tensor.numel()
            finite = bool(torch.isfinite(tensor).all()) and finite
        check(name + "_tensors_finite", finite, {"tensors": count, "elements": elements})
        # JSON stores argparse tuple defaults as arrays; compare their saved values.
        checkpoint_args = json.loads(json.dumps(saved_value["args"], allow_nan=False))
        check(name + "_args_match", checkpoint_args == configuration,
              {"tuple_fields_serialized_as_arrays": sorted(
                  key for key, value in saved_value["args"].items()
                  if isinstance(value, tuple)),
               "different_fields": sorted(
                   key for key in checkpoint_args.keys() | configuration.keys()
                   if checkpoint_args.get(key) != configuration.get(key))})
    for name in ("wm", "pi"):
        initial_state, final_state = first[name], final[name]
        same = initial_state.keys() == final_state.keys() and all(
            initial_state[key].shape == value.shape for key, value in final_state.items())
        changed = sum(not torch.equal(initial_state[key], value)
                      for key, value in final_state.items()) if same else 0
        check(name + "_parameters_updated", same and changed > 0,
              {"state_tensors": len(final_state), "changed_tensors": changed})
    check("last_matches_final_models", all(final[name].keys() == last[name].keys()
          and all(torch.equal(value, last[name][key]) for key, value in final[name].items())
          for name in ("wm", "pi", "critic")))
    optimizer_counts = {"opt_wm": receipt["wm_updates"],
                        "opt_pi": receipt["actor_updates"],
                        "opt_v": receipt["actor_updates"] * configuration["critic_epochs"]
                        * configuration["critic_minibatches"]}
    for name, expected_count in optimizer_counts.items():
        actual = sorted({int(state["step"].item())
                         for state in final[name]["state"].values()})
        check(name + "_adam_step_count", actual == [expected_count],
              {"expected": expected_count, "actual_distinct": actual})
    for name in ("initial", "final"):
        check(name + "_file_size_sha", identity(run / (name + ".pt")) == receipt[name])
    evidence = [json.loads(line) for line in (run / "training_evidence.jsonl").read_text().splitlines()]
    check("complete_logged_iterations", [row["step"] for row in evidence]
          == list(range(1, n + 1)))
    check("logged_evidence_finite", finite_numbers(evidence))
    check("valid_completed_episode_counts", all(0 <= row["hard_terminations"]
          <= row["completed_episodes"] <= row["transitions"] for row in evidence))
    end = evidence[-1]
    check("last_logged_counters_match", end["step"] == n and all(
        end[key] == receipt[key] for key in ("sim_steps", "wm_updates", "actor_updates")))
    check("cpu_only_audit", not torch.cuda.is_initialized())
    result = {"status": "HP_Q1_TRAINING_ARTIFACT_AUDIT_PASSED" if all(
        row["passed"] for row in checks) else "HP_Q1_TRAINING_ARTIFACT_AUDIT_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "training_receipt": receipt, "artifacts": [identity(path) for path in (
            trial / "plan.json", trial / "train_guard.json", run / "receipt.json",
            run / "config.yaml", run / "training_evidence.jsonl", run / "initial.pt",
            run / "final.pt", run / "last.pt")], "source": identity(__file__),
        "physics_integration_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
        "scope": "Saved training validity and identity, not policy tracking quality.",
        "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [row for row in checks if not row["passed"]]}))
    return 0 if all(row["passed"] for row in checks) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--train-clips", type=int, required=True)
    parser.add_argument("--validation-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
