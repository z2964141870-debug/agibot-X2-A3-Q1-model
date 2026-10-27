"""Audit RF12 saved PPO evidence without policy inference or simulation."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from audit_q1_hp_artifacts import finite_numbers, identity, read, tensors


def audit(trial):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    checks = []

    def check(name, passed, detail=None):
        checks.append(dict(name=name, passed=bool(passed), detail=detail))

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    def close(a, b):
        return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=2e-6, abs_tol=2e-7)

    plan, status = read(trial / "plan.json"), read(trial / "status.json")
    run = trial / "run"
    receipt, clock = read(run / "receipt.json"), read(run / "training_clock.json")
    decisions = read(run / "stage_decisions.json")
    evidence = [json.loads(line) for line in (run / "training_evidence.jsonl").read_text().splitlines()]
    final = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    initial = torch.load(run / "initial.pt", map_location="cpu", weights_only=False)
    stage = torch.load(run / "stage_01.pt", map_location="cpu", weights_only=False)
    parent = torch.load(final["source_RF11"]["path"], map_location="cpu", weights_only=False)
    check("pipeline_completed", status["status"] == "COMPLETED")
    check("normal_training_exit_with_physical_regression", receipt["status"] == "TRAINING_COMPLETED"
          and receipt["stop_reason"] == "physical_regression" and not receipt["physical_gate_met"])
    for label in ("train", "delivery"):
        guard = read(trial / (label + "_guard.json"))
        check(label + "_worker_reaped_zero", guard["status"] == "WORKER_COMPLETED"
              and guard["cleanup"]["worker_return_code"] == 0, guard["cleanup"])
    mismatches = [item["path"] for item in plan["source"] if identity(item["path"]) != item]
    check("all_frozen_sources_unchanged", not mismatches,
          dict(files=len(plan["source"]), bytes=sum(x["bytes"] for x in plan["source"]), mismatches=mismatches))
    check("frozen_code_identity", plan["code_commit"] == final["code_commit"] == initial["code_commit"]
          == "684548c8e912802960a837924e5af7fd4c03248d")
    check("RF11_parent_identity", identity(final["source_RF11"]["path"]) == final["source_RF11"]
          and final["source_RF11"]["sha256"] == "6c4de3171bd45359691a63ffe1712729c14fdb2080f8066e4c25632f4bbc5146")
    check("initial_policy_exact_RF11", equal(initial["pi"], parent["pi"]))
    check("initial_zero_updates", initial["iteration"] == initial["sim_steps"] == initial["optimizer_steps"]
          == initial["actor_optimizer_steps"] == 0 and not initial["optimizer"]["state"])
    normalizers = [key for key in initial["pi"] if "_norm." in key]
    check("normalizers_copied_and_frozen", len(normalizers) == 6
          and all(torch.equal(initial["pi"][k], final["pi"][k]) for k in normalizers))
    check("fixed_exploration_std", torch.equal(initial["distribution"]["std_param"],
          final["distribution"]["std_param"]) and torch.allclose(final["distribution"]["std_param"],
          torch.full((22,), .1), rtol=0, atol=1e-8))
    check("finite_saved_weights_and_optimizer", all(torch.isfinite(t).all().item()
          for payload in (initial, stage, final) for key in ("pi", "critic", "optimizer", "distribution")
          for t in tensors(payload[key])))
    check("stage_to_final_no_updates", all(equal(stage[k], final[k]) for k in
          ("pi", "critic", "optimizer", "distribution", "torch_rng", "cuda_rng", "python_rng", "numpy_rng")))
    check("actor_actually_changed", any(not torch.equal(initial["pi"][k], final["pi"][k])
          for k in initial["pi"] if k not in normalizers))
    n, seconds = final["iteration"], final["training_seconds"]
    check("A_training_budget", 3500 <= seconds <= 3600 and seconds == receipt["training_seconds"]
          == clock["accumulated_training_seconds"] and clock["active_started_monotonic"] is None)
    check("counters_match_every_completed_iteration", len(evidence) == n and all(
          row["step"] == i and row["sim_steps"] == i * 98304 and row["transitions"] == 98304
          and row["optimizer_steps"] == i * 20 and row["actor_optimizer_steps"] == max(i - 10, 0) * 20
          for i, row in enumerate(evidence, 1)))
    optimizer_steps = sorted(set(int(x["step"]) for x in final["optimizer"]["state"].values()))
    check("actual_optimizer_state_steps", optimizer_steps == sorted({n * 20, (n - 10) * 20}), optimizer_steps)
    check("finite_monotonic_training_evidence", all(finite_numbers(row) for row in evidence)
          and all(b["training_seconds"] > a["training_seconds"] for a, b in zip(evidence, evidence[1:]))
          and evidence[-1]["training_seconds"] == seconds)
    check("A_only_no_reward_or_data_ramp", all(row["recipe_phase"] == "A" and row["reward_alpha"] == 0
          and row["new_motion_fraction"] == 0 and row["bonus_reward_sum"] == 0 for row in evidence)
          and all(final[k] is None for k in ("stage_b_started", "stage_c_started", "extension_deadline")))
    check("world_model_and_imagination_NA", all(row["wm_err8"] is None and row["imag_survival_H"] is None for row in evidence))
    check("one_stop_decision", len(decisions) == 1 and decisions[0]["phase"] == "A"
          and decisions[0]["decision"] == {"action": "stop", "reason": "physical_regression"}
          and not decisions[0]["gate"]["extension_qualified"])

    rows = read(trial / "evaluation/evaluation_library.json")["splits"]["train"]
    validation = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    check("fixed_300_30_evaluation", len(rows) == 330 and len(validation) == 30
          and len({r["name"] for r in rows}) == 330)
    trace_results, saved_identities = [], []
    for label, identities in (("RF11_baseline", receipt["baseline_traces"]), ("stage_01", receipt["final_traces"])):
        for repeat, declared in enumerate(identities):
            path = Path(declared["path"])
            actual = identity(path)
            saved_identities.append(actual)
            summary = read(path.with_suffix(".json"))
            with np.load(path, allow_pickle=False) as stored:
                steps, alive = stored["steps"], stored["alive"]
                finite = all(np.isfinite(stored[key]).all() for key in stored.files if stored[key].dtype.kind in "fc")
                valid_steps = np.array_equal(steps, alive.sum(0)) and np.all(steps > 0) and np.all(steps <= stored["reference_steps"])
                aligned = all(np.all(np.diff(stored["frames"][:int(count), i]) == 1) for i, count in enumerate(steps))
                root_values, upper_values, leg_values, completion, failures = [], [], [], [], []
                for i in validation:
                    count = int(steps[i])
                    state = stored["states"][:count, i]
                    root_values.append(float(np.sqrt(np.mean((state[:, :2] - stored["reference_p"][:count, i, :2]) ** 2))))
                    upper_values.append(float(np.sqrt(np.mean((state[:, 27:35] - stored["reference_q"][:count, i, 14:]) ** 2))))
                    leg_values.append(float(np.sqrt(np.mean((state[:, 13:25] - stored["reference_q"][:count, i, :12]) ** 2))))
                    completion.append(float(count / stored["reference_steps"][i]))
                    failures.append(bool(stored["failed"][i]))
                metrics = summary["summary"]["validation"]
                verified = all(close(float(np.mean(values)), metrics[key]) for values, key in
                               ((root_values, "root_xy_rmse"), (upper_values, "upper_rmse"),
                                (leg_values, "leg_rmse"), (completion, "completion"))) and sum(failures) == metrics["failures"]
                trace_results.append(dict(label=label, repeat=repeat, failures=sum(failures),
                                          completion=float(np.mean(completion)), validation_values_verified=verified))
            check(label + f"_{repeat}_saved_trace_and_summary", actual == declared and finite and valid_steps and aligned and verified)

    for repeat, comparison in enumerate(decisions[0]["comparisons"]):
        saved = read(trial / "evaluation" / f"stage_01_repeat{repeat}.json")["RF11_common_prefix_comparison"]
        ratios = comparison["ratios"]
        derived = all(close(ratios[k], v["candidate"] / v["baseline"])
                      for k, v in comparison["measurements"].items())
        severe = comparison["failures"] >= 6 or ratios["root_xy_rmse"] >= 1.25 or ratios["upper_rmse"] >= 1.25
        check(f"paired_gate_arithmetic_repeat{repeat}", saved == comparison and derived and severe
              and decisions[0]["gate"]["per_repeat"][repeat]["severe"])

    with (trial / "dashboard/training_dashboard.csv").open(newline="") as stream:
        plotted = list(csv.DictReader(stream))
    series = {
        "episode_normalized_return": [r["normalized_return_sum"] / r["completed_episodes"] for r in evidence],
        "hard_termination_per_completed_episode": [r["hard_terminations"] / r["completed_episodes"] for r in evidence],
        "completed_episode_steps": [r["length_steps_sum"] / r["completed_episodes"] for r in evidence],
        "step_reward_over_current_max": [r["normalized_step_reward_sum"] / r["transitions"] for r in evidence],
    }
    for metric, values in series.items():
        check("csv_raw_and_500_iteration_mean_" + metric, len(plotted) == len(values) and all(
              close(row[metric], values[i]) and close(row[metric + "_mean500"], float(np.mean(values[max(0, i - 499):i + 1])))
              for i, row in enumerate(plotted)))
    for relative in ("delivery_status.json", "stages/stage_01/delivery_status.json"):
        delivery = read(trial / relative)
        check(relative + "_no_new_physics_or_updates", delivery["status"] == "COMPLETED"
              and delivery["new_physical_steps"] == delivery["new_optimizer_steps"] == 0)
    return dict(status="RF12_SAVED_EVIDENCE_AUDIT_PASSED" if all(c["passed"] for c in checks) else "RF12_SAVED_EVIDENCE_AUDIT_FAILED",
                created_utc=datetime.now(timezone.utc).isoformat(), trial=str(trial), checks=checks,
                actual_training_seconds=seconds, iterations=n, sim_steps=final["sim_steps"],
                physical_outcome="physical_regression", gate_met=False, trace_results=trace_results,
                artifacts=saved_identities + [identity(run / name) for name in ("initial.pt", "stage_01.pt", "final.pt", "receipt.json", "stage_decisions.json", "training_evidence.jsonl")],
                source=identity(__file__), physics_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
                limits="Integrity checks passing do not mean physical improvement. Whole-trace metrics are independently recomputed; paired comparison ratios are checked from saved measurements. No new rollout or PPO update is executed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.trial.resolve(strict=True))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    return 0 if result["status"].endswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
