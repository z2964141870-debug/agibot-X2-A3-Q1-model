#!/usr/bin/env python3
"""Inspect PHUMA rewards and both world models on common saved trajectories."""

import argparse
from dataclasses import asdict
import datetime
import gc
import json
import math
from pathlib import Path
import sys

from diagnose_completed_run import verified_evaluation
from render_audited_trace import checked_file, identity


def diagnostic_rows(initial, final, splits):
    expected = {name: split for split, names in splits.items() for name in names}
    if (set(splits) != {"train", "development", "test"}
            or len(expected) != sum(map(len, splits.values()))):
        raise ValueError("Require three disjoint registered splits")
    indexed = []
    for receipt in (initial, final):
        rows = {row["clip"]: row for row in receipt["rows"]}
        if len(rows) != len(receipt["rows"]) or rows.keys() != expected.keys():
            raise ValueError("Paired receipt coverage differs from the specification")
        if any(row["split"] != ("training" if expected[name] == "train" else "heldout")
               for name, row in rows.items()):
            raise ValueError("A diagnostic row has a different registered split")
        indexed.append(rows)
    for name in expected:
        a, b = (rows[name] for rows in indexed)
        if a["reference"] != b["reference"] or a["expected_steps"] != b["expected_steps"]:
            raise ValueError("Initial/final references or full denominators differ")
    selected = splits["train"] + splits["development"]
    if not selected:
        raise ValueError("No train/development diagnostic clips")
    return [[dict(rows[name], split=expected[name], evaluation_split=rows[name]["split"])
             for name in selected] for rows in indexed]


def full_denominator_terms(row, maximum):
    count, expected = row["executed_steps"], row["expected_steps"]
    if (type(count) is not int or type(expected) is not int
            or not 0 < count <= expected or not math.isfinite(maximum) or maximum <= 0):
        raise ValueError("Invalid complete-reference reward denominator")
    means = row["mean_step_rewards"]
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError("Nonfinite reward component")
    terms = {key: value * count / (maximum * expected)
             for key, value in means.items() if key != "total"}
    if abs(sum(terms.values()) - row["normalized_return"]) > 2e-5:
        raise ValueError("Reward components do not sum to the full-denominator return")
    return terms


def window_starts(executed, horizon, stride=8):
    if (any(type(value) is not int for value in (executed, horizon, stride))
            or executed < 0 or horizon < 1 or stride < 1):
        raise ValueError("Invalid saved-transition window")
    return list(range(0, executed - horizon + 1, stride))


def run(args):
    root, experiment, output = (path.resolve() for path in
                                (args.root, args.experiment, args.output))
    if output.exists():
        raise FileExistsError(output)
    if not all(path.is_relative_to(root / "data/supermimic_for_q1")
               for path in (experiment, output)):
        raise ValueError("Use the Q1 artifact root")
    comparison_path = experiment / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    controller = json.loads((experiment / "controller_status.json").read_text())
    spec_path = checked_file(comparison["spec"])
    spec = json.loads(spec_path.read_text())
    if (controller["status"] != "PHUMA_TRAINING_AND_MATCHED_EVALUATION_COMPLETED"
            or len(controller["events"]) != 7
            or any(row["exit_code"] != 0 for row in controller["events"])
            or comparison["status"] != "PHUMA_MATCHED_PHYSICS_COMPARISON_COMPLETED"
            or spec_path != experiment / "spec.json"
            or spec["initialization"] != "RANDOM_NO_PARENT_CHECKPOINT"):
        raise ValueError("Require the original completed PHUMA random-initialization run")

    def frozen():
        for record in spec["frozen_files"]:
            checked_file(record)

    cg = Path("/sys/fs/cgroup/memory")
    memory_samples = []

    def health():
        usage = int((cg / "memory.usage_in_bytes").read_text())
        oom = dict(line.split() for line in (cg / "memory.oom_control").read_text().splitlines())
        memory_samples.append(usage)
        if (usage >= 30 * 1024**3 - 256 * 1024**2
                or int(oom["oom_kill"]) != args.expected_oom_kill
                or int(oom["under_oom"]) != 0):
            raise RuntimeError("PHUMA diagnosis resource guard refused execution")
        return {"memory_bytes": usage, "oom_kill": int(oom["oom_kill"])}

    initial_health = health()
    if 30 * 1024**3 - initial_health["memory_bytes"] < 2 * 1024**3:
        raise RuntimeError("Require 2 GiB initial headroom for CPU model diagnosis")
    frozen()
    for record in comparison["evidence"]:
        checked_file(record)
    evaluations = {}
    for label in ("initial", "final"):
        receipt = identity(experiment / ("evaluation_" + label) / "receipt.json")
        audit = identity(experiment / ("evaluation_audit_" + label + ".json"))
        if receipt not in comparison["evidence"] or audit not in comparison["evidence"]:
            raise ValueError("Paired comparison does not bind the original evaluation audits")
        evaluations[label] = verified_evaluation(receipt, audit)
    selected = diagnostic_rows(evaluations["initial"], evaluations["final"], spec["split_clips"])
    selected = dict(zip(("initial", "final"), selected))

    sys.path.insert(0, str(root / "script/supermimic_for_q1"))
    import numpy as np
    import torch
    import analyze_learning
    from contact_protocol import ContactCorrectedAssets
    from measure_world_model import metrics
    from supermimic.robot_spec import XI_DIM
    from supermimic.world_model.state_spec import DELTA_DIM, StateSpecCfg
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    assets = ContactCorrectedAssets(root)
    heads = assets.heads("cpu")
    health()
    output.mkdir(parents=True, exist_ok=False)
    rewards = {}
    for label in ("initial", "final"):
        view = dict(evaluations[label])
        view.update(status="DIAGNOSTIC_TRAIN_DEVELOPMENT_SUBSET_OF_AUDITED_EVALUATION",
                    rows=selected[label],
                    parent_receipt=identity(experiment / ("evaluation_" + label) / "receipt.json"))
        path = output / ("reward_input_" + label + ".json")
        with path.open("x") as stream:
            json.dump(view, stream, indent=2, allow_nan=False)
            stream.write("\n")
        rewards[label] = analyze_learning.describe_evaluation(path, assets, heads)
        for row in rewards[label]["rows"]:
            row["full_denominator_contributions"] = full_denominator_terms(
                row, heads.reward_cfg.max_step_reward)
        health()
    changes = []
    for a, b in zip(rewards["initial"]["rows"], rewards["final"]["rows"]):
        delta = {key: b["full_denominator_contributions"][key]
                 - a["full_denominator_contributions"][key]
                 for key in a["full_denominator_contributions"]}
        changes.append({"clip": b["clip"], "split": b["split"],
                        "initial_executed_steps": a["executed_steps"],
                        "final_executed_steps": b["executed_steps"],
                        "expected_steps": b["expected_steps"],
                        "normalized_return_delta": b["normalized_return"] - a["normalized_return"],
                        "full_denominator_term_deltas": delta})

    models, arrays = [], []
    for label in ("initial", "final"):
        health()
        checkpoint_record = evaluations[label]["checkpoint"]
        checkpoint = torch.load(checked_file(checkpoint_record), map_location="cpu",
                                weights_only=False, mmap=True)
        if checkpoint["manifest"]["assets"] != assets.manifest():
            raise ValueError("World model uses different contact assets")
        config = dict(checkpoint["manifest"]["world_model"])
        config["state_spec"] = StateSpecCfg(**config["state_spec"])
        model = WorldModel(WorldModelCfg(**config), heads, "cpu").eval()
        model.load_state_dict(checkpoint["wm"], strict=True)
        del checkpoint
        for row in selected["final"]:
            health()
            with np.load(checked_file(row["reference"]), allow_pickle=False) as reference:
                if float(reference["fps"]) != 50.:
                    raise ValueError("Expected the native-time 50 Hz PHUMA reference")
                initial = np.concatenate([reference["qpos"][0], reference["qvel"][0]])
            with np.load(checked_file(row["trace"]), allow_pickle=False) as trace:
                states = torch.as_tensor(np.concatenate([initial[None], trace["state"]], axis=0))
                actions = torch.as_tensor(trace["actions"]).clamp(-20., 20.)
            if len(actions) != row["executed_steps"] or len(states) != len(actions) + 1:
                raise ValueError("Saved transition counts differ")
            for horizon in (1, 8, 32):
                starts = window_starts(len(actions), horizon)
                result = {"model": label, "trajectory": "final", "clip": row["clip"],
                          "split": row["split"], "horizon": horizon, "windows": len(starts)}
                if not starts:
                    result["status"] = "INSUFFICIENT_SAVED_STEPS"
                    models.append(result)
                    continue
                indices = torch.tensor(starts)
                predicted = states[indices].clone()
                constant = predicted.clone()
                context = torch.zeros(len(indices), XI_DIM)
                context[:, 0] = 1.
                zero = torch.zeros(len(indices), DELTA_DIM)
                for offset in range(horizon):
                    predicted = model.step(predicted, actions[indices + offset], context)
                    constant = model.spec.integrate(constant, zero)
                actual = states[indices + horizon]
                learned, baseline = (metrics(model.spec, value, actual)
                                     for value in (predicted, constant))
                path = output / (label + "_" + row["clip"] + "_h" + str(horizon) + ".npz")
                np.savez_compressed(path, starts=np.asarray(starts), actual=actual.numpy(),
                                    predicted=predicted.numpy(), constant_velocity=constant.numpy())
                record = identity(path)
                arrays.append(record)
                result.update(status="COMMON_SAVED_TRAJECTORY_PREDICTION_COMPLETED",
                              learned=learned, constant_velocity=baseline, arrays=record,
                              learned_to_cv_ratio={key: learned[key] / baseline[key] if baseline[key] else None
                                                   for key in learned})
                models.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
                health()
        del model
        gc.collect()
        health()
    frozen()
    final_health = health()
    report = {"status": "PHUMA_COMMON_TRAJECTORY_REWARD_MODEL_DIAGNOSIS_COMPLETED",
              "code_commit": args.code_commit, "source": identity(__file__),
              "diagnostic_dependencies": [identity(Path(__file__).parent / name) for name in
                                          ("diagnose_completed_run.py", "render_audited_trace.py")],
              "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "spec": comparison["spec"], "comparison": identity(comparison_path),
              "training_code_commit": spec["code_commit"],
              "diagnostic_splits": ["train", "development"],
              "selected_clips": [row["clip"] for row in selected["final"]],
              "reward_config": asdict(heads.reward_cfg),
              "max_step_reward": heads.reward_cfg.max_step_reward,
              "rewards": rewards, "reward_changes": changes, "world_models": models,
              "arrays": arrays, "initial_health": initial_health, "final_health": final_health,
              "max_sampled_cgroup_bytes": max(memory_samples),
              "frozen_files_unchanged_before_and_after": len(spec["frozen_files"]),
              "new_physics_steps": 0, "new_policy_calls": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY",
              "limits": "Both models use identical final saved states, clamped physical actions and windows. "
                        "Train/development only; overlapping windows are not independent samples. "
                        "No targets after saved termination. Nominal context friction=1, other offsets=0. "
                        "Reward changes include completion differences and are arithmetic attribution, not ablation. "
                        "Prediction accuracy does not establish action-gradient accuracy or causal training failure."}
    with (output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "receipt": identity(output / "receipt.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-oom-kill", type=int, default=14)
    run(parser.parse_args())
