"""Summarize the saved five-alpha sweep without more simulation or learning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from worker import identity, save

METRICS = ("root_xy_rmse", "upper_rmse", "leg_rmse", "wrist_relative_rmse",
           "target_oscillation", "loaded_slip")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    root = args.trial
    if (root / "delivery.json").exists():
        raise FileExistsError("Saved UR03 delivery already exists")
    source = json.loads((root / "results.json").read_text())
    names = list(source["alphas"])
    results = source["results"]
    records, points = [], []
    for name in names:
        repeats = results[name]
        point = dict(name=name, alpha=source["alphas"][name], common_mean={}, common_ratios={})
        for metric in METRICS:
            values, ratios = [], []
            for repeat, summary in enumerate(repeats):
                value = summary["all_alpha_common_prefix"]["measurements"][metric]
                base = results[names[0]][repeat]["all_alpha_common_prefix"]["measurements"][metric]
                ratio = value / base if value is not None and base is not None and base > 0 else None
                values.append(value)
                ratios.append(ratio)
                records.append(dict(condition=name, alpha=point["alpha"], repeat=repeat,
                    protocol="all_alpha_common_prefix", clip="validation_mean", metric=metric,
                    value=value, baseline=base, ratio=ratio))
                pair = summary["RF11_common_prefix_comparison"]["measurements"][metric]
                records.append(dict(condition=name, alpha=point["alpha"], repeat=repeat,
                    protocol="paired_RF11_common_prefix", clip="validation_mean", metric=metric,
                    value=pair["candidate"], baseline=pair["baseline"],
                    ratio=summary["RF11_common_prefix_comparison"]["ratios"][metric]))
            point["common_mean"][metric] = float(np.mean(values)) if all(v is not None for v in values) else None
            point["common_ratios"][metric] = ratios
        point["validation_failures"] = [s["summary"]["validation"]["failures"] for s in repeats]
        point["validation_completion"] = [s["summary"]["validation"]["completion"] for s in repeats]
        point["fixed_demos"] = [[{key: r[key] for key in
            ("name", "steps", "reference_steps", "failed", "raised_peak_error",
             "actual_wrist_peak_height_m", "reference_wrist_peak_height_m",
             "actual_wrist_peak_time_seconds", "reference_wrist_peak_time_seconds")}
            for r in s["fixed_demos"]] for s in repeats]
        for repeat, summary in enumerate(repeats):
            for key in ("failures", "completion"):
                records.append(dict(condition=name, alpha=point["alpha"], repeat=repeat,
                    protocol="full_episode", clip="validation_mean", metric=key,
                    value=summary["summary"]["validation"][key], baseline="", ratio=""))
            for demo in summary["fixed_demos"]:
                for key in ("steps", "reference_steps", "failed", "raised_peak_error"):
                    records.append(dict(condition=name, alpha=point["alpha"], repeat=repeat,
                        protocol="full_episode", clip=demo["name"], metric=key,
                        value=demo[key], baseline="", ratio=""))
        point["pareto_objectives"] = [point["common_mean"][m] for m in METRICS] + [
            max(point["validation_failures"]), -min(point["validation_completion"]),
            max(sum(d["failed"] for d in demos) for demos in point["fixed_demos"])]
        points.append(point)
    for point in points:
        own = point["pareto_objectives"]
        point["dominated_by"] = []
        if all(v is not None and np.isfinite(v) for v in own):
            for other in points:
                values = other["pareto_objectives"]
                if all(v is not None and np.isfinite(v) for v in values) and (
                    all(a <= b for a, b in zip(values, own)) and
                    any(a < b for a, b in zip(values, own))):
                    point["dominated_by"].append(other["name"])
        else:
            point["dominated_by"] = ["undefined_nonfinite_or_missing"]
    save(root / "tradeoff_summary.json", dict(points=points, optimizer_steps=0,
        common_prefix_mean_seconds=[s["all_alpha_common_prefix"]["mean_seconds"]
                                    for s in results[names[0]]],
        pareto_scope="Only the five fixed alpha settings and recorded objectives, using common-prefix error means, worst-repeat validation failures/completion and demo failures. Not a global optimum, independent test, statistical significance or deployment safety certificate.",
        source=identity(root / "results.json")))
    with (root / "tradeoff_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([p["alpha"] for p in points])
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.6), constrained_layout=True)
    axis = axes[0, 0]
    colors = ("#5d6268", "#248675", "#c18c24", "#b24a68", "#4475af")
    for point, color in zip(points, colors):
        upper, root_error = (np.array(point["common_ratios"][m]) for m in
                             ("upper_rmse", "root_xy_rmse"))
        axis.scatter(root_error, upper, color=color, s=20)
        axis.annotate(f'a={point["alpha"]:g}', (root_error.mean(), upper.mean()),
                      xytext=(3, 5), textcoords="offset points", fontsize=9)
    axis.axvline(1, color="#777777", lw=.8, ls="--")
    axis.axhline(1, color="#777777", lw=.8, ls="--")
    axis.set_xlabel("Root XY error / RF11 (lower is better)")
    axis.set_ylabel("Upper error / RF11 (lower is better)")
    axis.set_title("Same all-alpha shared prefix; three raw repeats", fontsize=10)
    axis.grid(alpha=.2)
    for axis, metrics, title in ((axes[0, 1], ("upper_rmse", "wrist_relative_rmse"), "Arm and local-wrist error ratios"),
                                (axes[0, 2], ("root_xy_rmse", "leg_rmse"), "Root and leg error ratios"),
                                (axes[1, 2], ("loaded_slip", "target_oscillation"), "Loaded slip and target oscillation ratios")):
        for metric, color in zip(metrics, ("#248675", "#b24a68")):
            values = np.array([p["common_ratios"][metric] for p in points], dtype=float)
            axis.plot(x, values.mean(1), marker="o", color=color, label=metric.replace("_", " "))
            axis.scatter(np.repeat(x, 3), values.reshape(-1), color=color, s=12, alpha=.55)
        axis.axhline(1, color="#777777", ls="--", lw=.8)
        axis.set_title(title + " (shared prefix)", fontsize=10)
        axis.set_xlabel("Reference blend alpha")
        axis.set_xticks(x)
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
    axis = axes[1, 0]
    for repeat, offset in enumerate((-.025, 0, .025)):
        axis.scatter(x + offset, [p["validation_failures"][repeat] for p in points],
                     label=f"repeat{repeat}", s=22)
    axis.set_ylim(-.3, max(3, max(max(p["validation_failures"]) for p in points) + 1))
    axis.set_xlabel("Reference blend alpha")
    axis.set_xticks(x)
    axis.set_title("Full episodes: validation early terminations /30", fontsize=10)
    axis.grid(alpha=.2)
    axis.legend(fontsize=7)
    axis = axes[1, 1]
    for demo_index, label, color in zip(range(3), ("Walk-stop", "Walk-start", "Raised hands"), colors[1:4]):
        completion = np.array([[d[demo_index]["steps"] / d[demo_index]["reference_steps"]
                                for d in p["fixed_demos"]] for p in points])
        axis.plot(x, completion.mean(1), color=color, marker="o", label=label)
        axis.scatter(np.repeat(x, 3), completion.reshape(-1), color=color, s=12, alpha=.55)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Reference blend alpha")
    axis.set_xticks(x)
    axis.set_title("Fixed demos: full-episode completion", fontsize=10)
    axis.legend(fontsize=7)
    axis.grid(alpha=.2)
    figure.suptitle("UR03 | Frozen RF11 | 33 closed-loop environments | 0 optimizer | No independent test", fontsize=12)
    for suffix in ("png", "pdf"):
        figure.savefig(root / ("upper_tradeoff." + suffix), dpi=160)
    plt.close(figure)
    save(root / "delivery.json", dict(status="COMPLETED", new_physics_steps=0,
        optimizer_steps=0, source=identity(root / "results.json"),
        artifacts=[identity(root / name) for name in
                   ("tradeoff_summary.json", "tradeoff_metrics.csv", "upper_tradeoff.png", "upper_tradeoff.pdf")],
        protocol="Diagnostic sweep, not a training dashboard; no training iteration smoothing, WM or imagined metrics. Complete failures/demos accompany all shared-prefix errors."))


if __name__ == "__main__":
    main()
