#!/usr/bin/env python3
"""Render a six-panel dashboard from recorded training and episode metrics."""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time


PANELS = (
    ("episode_return", "Episode normalized return", "Mean over completed episodes"),
    ("episode_termination", "Hard terminations / completed episodes", "Fraction"),
    ("episode_length", "Completed episode length (steps)", "Control steps"),
    ("step_reward", "Simulator step reward / maximum", "Normalized reward"),
    ("world_model", "World model error at step 8", "Normalized squared error"),
    ("survival", "Imagined survival", "Survival at horizon H"),
)
COLORS = ("#237c6a", "#bc4661", "#444950", "#327eaf", "#a36a26", "#775ba8")


def identity(path, content=None):
    path = Path(path)
    content = path.read_bytes() if content is None else content
    return {"path": str(path.resolve()), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def read_jsonl(path, allow_partial=False):
    content = path.read_bytes()
    rows = []
    lines = content.splitlines(keepends=True)
    ignored = 0
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if allow_partial and index == len(lines) - 1 and not line.endswith(b"\n"):
                ignored = len(line)
                break
            raise ValueError(f"Invalid recorded JSON at {path}:{index + 1}") from None
        if not isinstance(row, dict) or not isinstance(row.get("step"), int) or row["step"] < 0:
            raise ValueError(f"Missing integer training iteration at {path}:{index + 1}")
        if rows and row["step"] <= rows[-1]["step"]:
            raise ValueError(f"Non-increasing training iterations at {path}:{index + 1}")
        rows.append(row)
    return rows, content, ignored


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rolling_ratio(points, width):
    """Pool numerator/denominator in (iteration-width, iteration], not N rows."""
    left = 0
    numerator = denominator = 0.
    result = []
    for right, (step, num, den) in enumerate(points):
        numerator += num
        denominator += den
        while left <= right and points[left][0] <= step - width:
            numerator -= points[left][1]
            denominator -= points[left][2]
            left += 1
        result.append(numerator / denominator if denominator > 0 else None)
    return result


def episode_points(rows, key):
    points = []
    for row in rows:
        count = row.get("completed_episodes")
        value = row.get(key)
        if not isinstance(count, int) or count < 0 or not finite(value):
            raise ValueError("Episode windows need a nonnegative count and finite sums")
        if key == "hard_terminations" and not 0 <= value <= count:
            raise ValueError("Hard terminations must be a subset of completed episodes")
        if count == 0 and value != 0:
            raise ValueError("Empty episode windows must have zero sums")
        points.append((row["step"], float(value), float(count)))
    return points


def load_run(path, label, reward_max=None, reward_source=None, allow_partial=False):
    path = path.resolve(strict=True)
    metrics, content, partial = read_jsonl(path / "metrics.jsonl", allow_partial)
    if not metrics:
        raise ValueError(f"No training metrics in {path}")
    admission_path = path / "admission.json"
    admission = json.loads(admission_path.read_text())
    status_path = path / "status.json"
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes)
    args = admission["args"]
    sources = [identity(path / "metrics.jsonl", content), identity(admission_path),
               identity(status_path, status_bytes)]
    metadata_path = path / "reporting_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        recorded_max = metadata["reward_max"]
        if reward_max is not None and not math.isclose(recorded_max, reward_max):
            raise ValueError("Explicit reward maximum differs from the run's reporting metadata")
        reward_max = recorded_max
        reward_source = metadata["reward_max_source"]
        sources.append(identity(metadata_path))
    if reward_max is not None and (not finite(reward_max) or reward_max <= 0 or not reward_source):
        raise ValueError("Reward normalization requires a positive maximum and its explicit source")
    if reward_max is not None and isinstance(reward_source, str) and Path(reward_source).is_file():
        reward_source = identity(reward_source)
    points = {}
    episode_path = path / "episodes.jsonl"
    if episode_path.exists():
        episodes, episode_bytes, ep_partial = read_jsonl(episode_path, allow_partial)
        partial += ep_partial
        last_step = metrics[-1]["step"] if status["status"] == "RUNNING" else status["step"]
        episodes = [row for row in episodes if row["step"] <= last_step]
        sources.append(identity(episode_path, episode_bytes))
        for key, total in (("episode_return", "normalized_return_sum"),
                           ("episode_termination", "hard_terminations"),
                           ("episode_length", "length_steps_sum")):
            points[key] = episode_points(episodes, total)
    for key, metric, scale in (("step_reward", "reward_mean", reward_max),
                               ("world_model", "wm/nse_kstep", 1.),
                               ("survival", "imag_survival_H", 1.)):
        if key == "world_model" and admission.get("wm_trainer", {}).get("window") != 8:
            continue
        if scale is not None:
            points[key] = [(row["step"], float(row[metric]) / scale, 1.)
                           for row in metrics if finite(row.get(metric))]
    return {"label": label, "path": str(path), "points": points, "sources": sources,
            "last_logged_step": metrics[-1]["step"], "requested_steps": args["steps"],
            "num_envs": args["num_envs"], "actor_batch": args["batch"],
            "wm_batch": args["wm_batch"], "wm_window": admission["wm_trainer"]["window"],
            "horizon": admission["policy_trainer"]["horizon"],
            "status": status["status"], "ignored_incomplete_tail_bytes": partial,
            "reward_max": reward_max, "reward_max_source": reward_source}


def create_dashboard(runs, output, window=500, title="Q1 SuperMimic training", provisional=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if window < 1:
        raise ValueError("Moving-average window must be positive training iterations")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    figure.subplots_adjust(top=.79, bottom=.12, left=.065, right=.985, wspace=.25, hspace=.42)
    budget = "; ".join(f"{r['label']}: {r['num_envs']} env, actor B{r['actor_batch']}, "
                       f"WM B{r['wm_batch']}, {r['last_logged_step']}/{r['requested_steps']} iterations"
                       for r in runs)
    figure.suptitle(title + (" [in progress]" if provisional else ""), fontsize=17, y=.98)
    figure.text(.5, .935, f"Thin: recorded values; thick: trailing {window}-iteration moving average",
                ha="center", fontsize=10)
    figure.text(.5, .895, budget, ha="center", fontsize=9, wrap=True)
    exported = []
    missing = {}
    max_step = max(r["last_logged_step"] for r in runs)
    for axis, (key, caption, ylabel) in zip(axes.flat, PANELS):
        missing[key] = []
        drawn = False
        for index, run in enumerate(runs):
            values = run["points"].get(key, [])
            if not any(den > 0 for _, _, den in values):
                missing[key].append(run["label"])
                continue
            color = COLORS[index % len(COLORS)]
            x = [point[0] for point in values]
            y = [num / den if den > 0 else float("nan") for _, num, den in values]
            average = rolling_ratio(values, window)
            axis.plot(x, y, color=color, alpha=.23, linewidth=.7)
            axis.plot(x, [v if v is not None else float("nan") for v in average],
                      color=color, linewidth=2., label=run["label"])
            for (step, num, den), smoothed in zip(values, average):
                exported.append({"experiment": run["label"], "iteration": step,
                    "env_steps": step * run["num_envs"], "metric": key,
                    "raw": num / den if den else "", "smoothed": smoothed if smoothed is not None else "",
                    "numerator": num, "denominator": den})
            drawn = True
        axis.set_title(caption, fontsize=11, pad=10)
        axis.set_xlabel("Training iterations", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        axis.set_xlim(0, max(1, max_step))
        axis.grid(color="#e3e6e9", linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
        if drawn:
            axis.legend(frameon=False, fontsize=8, loc="best")
        if missing[key]:
            text = "Not recorded" if not drawn else "Missing: " + ", ".join(missing[key])
            axis.text(.5, .5 if not drawn else .03, text, ha="center", va="center",
                      transform=axis.transAxes, fontsize=11 if not drawn else 8, color="#656b73")
    figure.text(.5, .025, "Episode averages use completed episodes; WM errors are training minibatches. "
                "Imagined survival is a model prediction.\nTraining diagnostics are separate from fixed-reference physics evaluation.",
                ha="center", fontsize=9, color="#555b63")
    png, pdf = output / "training_dashboard.png", output / "training_dashboard.pdf"
    figure.savefig(png, dpi=160)
    figure.savefig(pdf)
    plt.close(figure)
    csv_path = output / "training_dashboard.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("experiment", "iteration", "env_steps", "metric",
                                                    "raw", "smoothed", "numerator", "denominator"))
        writer.writeheader()
        writer.writerows(exported)
    serial_runs = [{key: value for key, value in run.items() if key != "points"} for run in runs]
    receipt = {"status": "PROVISIONAL_TRAINING_DASHBOARD" if provisional else "TRAINING_DASHBOARD_COMPLETED",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs": serial_runs,
        "layout": "2x3", "smoothing": {"window_training_iterations": window,
            "interval": "(iteration-window, iteration]", "episode_weight": "completed_episodes",
            "other_weight": "one per available logged row", "partial_initial_window": True},
        "missing_panels": missing, "artifacts": [identity(p) for p in (png, pdf, csv_path)],
        "source": identity(__file__), "physics_steps": 0, "optimizer_updates": 0,
        "storage_status": "LOCAL_ONLY"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def wait_for_training(path):
    """A one-run reporting callback; never starts or restarts training."""
    status_path = path / "status.json"
    first = json.loads(status_path.read_text())
    if first["status"] != "RUNNING":
        return
    pid = first["pid"]
    proc = Path(f"/proc/{pid}/stat")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    manifest = json.loads((path / "admission.json").read_text())
    if boot != manifest["boot_id"]:
        raise RuntimeError("Training boot identity differs; no callback was attached")
    ticks = proc.read_text().rsplit(") ", 1)[1].split()[19]
    deadline = time.monotonic() + 24 * 3600
    print(json.dumps({"status": "WAITING_FOR_ONE_TRAINING_RUN", "run": str(path),
        "training_pid": pid, "training_start_ticks": ticks, "boot_id": boot,
        "reporting_pid": os.getpid()}), flush=True)
    previous = None
    while True:
        status = json.loads(status_path.read_text())
        if status["status"] != "RUNNING":
            if not (path / "receipt.json").exists():
                raise RuntimeError("Terminal training has no receipt")
            return
        if status["pid"] != pid or not proc.exists():
            raise RuntimeError("Training disappeared or changed identity before a terminal receipt")
        if proc.read_text().rsplit(") ", 1)[1].split()[19] != ticks:
            raise RuntimeError("Training PID was reused")
        if time.monotonic() >= deadline:
            raise TimeoutError("Reporting callback expired; training was not changed")
        if status.get("step") != previous:
            previous = status.get("step")
            print(json.dumps({"status": "WAITING_FOR_ONE_TRAINING_RUN", "step": previous}), flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Q1 SuperMimic training")
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--reward-max", type=float)
    parser.add_argument("--reward-max-source")
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--after-training", action="store_true",
                        help="Wait for one identified existing run, then render its terminal dashboard")
    args = parser.parse_args()
    labels = args.labels or [path.parent.name for path in args.runs]
    if len(labels) != len(args.runs) or len(set(labels)) != len(labels):
        parser.error("Use one distinct label per run")
    if args.after_training:
        if len(args.runs) != 1 or args.provisional or args.output.exists():
            parser.error("The callback requires one run, a new output and no --provisional")
        wait_for_training(args.runs[0])
    runs = [load_run(path, label, args.reward_max, args.reward_max_source, args.provisional)
            for path, label in zip(args.runs, labels)]
    if not args.provisional and any(r["status"] == "RUNNING" for r in runs):
        parser.error("Running training requires --provisional")
    print(json.dumps(create_dashboard(runs, args.output, args.window, args.title, args.provisional),
                     indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
