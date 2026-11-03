"""Read a Primebot log snapshot without importing its learner or touching CUDA."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


COUNTERS = ("global_env_steps", "actual_model_optimizer", "actual_actor_optimizer",
            "actual_critic_optimizer")
METRICS = ("joint_rmse_rad", "body_relative_rmse_m")


def identity(path):
    content = path.read_bytes()
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def read_metrics(path):
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    incomplete = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete:
        lines.pop()
    rows = [json.loads(line) for line in lines if line.strip()]
    if not rows:
        raise ValueError("No complete iteration records")
    for i, row in enumerate(rows):
        for name in ("iteration", *COUNTERS):
            value = row[name]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid actual counter: " + name)
            if i and value <= rows[i - 1][name]:
                raise ValueError("Nonincreasing actual counter: " + name)
        if row["iteration"] < 1 or any(
                row[name] != row["iteration"] * factor
                for name, factor in zip(COUNTERS[1:], (10, 1, 20))):
            raise ValueError("Actual optimizer totals differ from the Primebot recipe")
        if not math.isfinite(row["update_seconds"]) or row["update_seconds"] <= 0:
            raise ValueError("Invalid measured update duration")
    return rows, incomplete


def observation_rows(history):
    result = []
    previous = -1
    for item in history["observations"]:
        iteration = item["iteration"]
        if type(iteration) is not int or iteration <= previous:
            raise ValueError("Duplicate or nonincreasing observation")
        previous = iteration
        gate = item.get("gate") or {}
        pairs = gate.get("pairs", [])
        row = dict(iteration=iteration, status=item["status"],
                   gate_passed=bool(gate.get("passed", False)),
                   complete_fixed_cases=sum(bool(p.get("final_complete")) for p in pairs)
                   if pairs else None,
                   comparison_cases=len(pairs), observer_seconds=item.get("seconds"),
                   parameters_unchanged=item.get("parameters_unchanged"),
                   rng_unchanged=item.get("rng_unchanged"),
                   gate_reason=";".join(gate.get("gate_reason", [])))
        for metric in METRICS:
            scores = gate.get("paired_metrics", {}).get(metric, {})
            before, after = scores.get("initial"), scores.get("final")
            for value in (before, after):
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError("Invalid paired tracking metric")
            row["initial_" + metric], row["current_" + metric] = before, after
            row["ratio_" + metric] = after / before if before and after is not None else None
        result.append(row)
    return result


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, observations, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 150})
    x = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for axis in axes.flat:
        axis.grid(alpha=.18)
        axis.set_xlabel("Completed iterations")
    axes[0, 0].plot(x, [row["global_env_steps"] / 1e6 for row in rows], color="#16795c")
    axes[0, 0].set(title="Actual environment transitions", ylabel="Million transitions")
    for name, color in (("state_loss", "#177a9b"), ("termination_loss", "#bd534d")):
        axes[0, 1].plot(x, [row.get(name, math.nan) for row in rows],
                        label=name, linewidth=.8, color=color)
    axes[0, 1].set(title="World model fit", ylabel="Loss", yscale="symlog")
    axes[0, 1].legend()
    axes[1, 0].plot(x, [row.get("collection_reward", math.nan) for row in rows],
                    color="#ae6d20", linewidth=.8)
    axes[1, 0].set(title="Collected training reward", ylabel="Reward")
    for name, color in (("actor_loss", "#6e6093"), ("critic_loss", "#177a9b")):
        axes[1, 1].plot(x, [row.get(name, math.nan) for row in rows],
                        label=name, linewidth=.8, color=color)
    axes[1, 1].set(title="Actor and critic updates", ylabel="Loss", yscale="symlog")
    axes[1, 1].legend()
    figure.suptitle("Primebot Q1: training signals do not establish gait quality", fontsize=14)
    figure.savefig(output / "training_signals.png")
    plt.close(figure)

    measured = [row for row in observations if row["iteration"] > 0]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis in axes:
        axis.grid(alpha=.18)
        axis.set_xlabel("Checkpoint iteration")
    present = [row for row in measured if row["complete_fixed_cases"] is not None]
    axes[0].plot([row["iteration"] for row in present],
                 [row["complete_fixed_cases"] for row in present], "o-", color="#16795c")
    axes[0].axhline(9, linestyle="--", color="#666666", linewidth=.8)
    axes[0].set(title="Fixed walking completion", ylim=(-.5, 9.5), ylabel="Complete / 9")
    for axis, metric, title in zip(axes[1:], METRICS, ("Joint tracking", "Body-relative tracking")):
        present = [row for row in measured if row["ratio_" + metric] is not None]
        axis.plot([row["iteration"] for row in present],
                  [row["ratio_" + metric] for row in present], "o-", color="#177a9b")
        axis.axhline(1, linestyle="--", color="#bd534d", linewidth=.8)
        axis.set(title=title, ylabel="RMSE / paired zero-update RMSE")
    figure.suptitle("Physical evaluation: common valid prefix, equal clip weight", fontsize=14)
    figure.savefig(output / "physical_progress.png")
    plt.close(figure)


def summarize(metrics_path, history_path, output, make_plots=False):
    rows, incomplete = read_metrics(metrics_path)
    history = json.loads(history_path.read_text())
    observations = observation_rows(history)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "iterations.csv", rows)
    write_csv(output / "physical_observations.csv", observations)
    times = [row["update_seconds"] for row in rows[-100:] if not row.get("startup_update")]
    estimate = statistics.median(times) if times else None
    record = dict(status="COMPLETED", inputs=[identity(metrics_path), identity(history_path)],
                  latest_iteration=rows[-1]["iteration"],
                  actual_counters={key: rows[-1][key] for key in COUNTERS},
                  median_last_100_update_seconds=estimate,
                  remaining_iteration_hours_to_250000=(250000 - rows[-1]["iteration"]) * estimate / 3600
                  if estimate else None,
                  estimate_includes_collection=True,
                  estimate_excludes_compilation_evaluation_saving_and_load_changes=True,
                  incomplete_last_metric_line_ignored=incomplete,
                  observations=observations, early_stop=history.get("early_stop", False),
                  new_training_updates=0, new_physics_steps=0,
                  user_accepted_checkpoint=None,
                  limitations=["Training signals are not gait qualification",
                               "Nominal seed repeats do not establish perturbation robustness",
                               "Common-prefix RMSE is not full-duration tracking RMSE"])
    if make_plots:
        plot(rows, observations, output)
    record["artifacts"] = [identity(p) for p in sorted(output.iterdir())]
    (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = summarize(args.metrics, args.history, args.output, args.plot)
    print(json.dumps({key: result[key] for key in ("status", "latest_iteration", "actual_counters")}))
