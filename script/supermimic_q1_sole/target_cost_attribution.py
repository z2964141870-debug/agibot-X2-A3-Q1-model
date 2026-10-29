"""Partition saved target penalties between residual-controlled and upper joints."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


def cost_parts(targets, scale, action_weight, dt):
    targets, scale = np.asarray(targets, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (targets.ndim != 2 or targets.shape[1] != 22 or len(targets) < 2 or scale.shape != (22,)
            or not np.isfinite(targets).all() or not np.isfinite(scale).all() or (scale <= 0).any()
            or not np.isfinite(action_weight) or not action_weight < 0 or not 0 < dt < 1):
        raise ValueError("Expected finite native22 targets, positive action scales and a negative rate weight")
    delta = np.diff(targets, axis=0)
    base = -action_weight * dt * np.square(delta / scale)
    smooth = .002 * np.square(delta / .1) / 22
    # Apply the original total clamp before assigning each group's share.
    smooth *= np.minimum(1., .05 / np.maximum(smooth.sum(-1), 1e-300))[:, None]
    return dict(base_controlled14=base[:, :14].sum(-1), base_upper8=base[:, 14:].sum(-1),
                ppo_controlled14=smooth[:, :14].sum(-1), ppo_upper8=smooth[:, 14:].sum(-1))


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project_tools"))
    from prepare_q1_gait_review import identity, write_csv

    destination = args.review / "target_cost_attribution.json"
    if destination.exists():
        raise FileExistsError(destination)
    summary_path = args.review / "summary.json"
    rows_path = args.review / "fixed_reward_signal.csv"
    summary = json.loads(summary_path.read_text())
    library_source = next(s for s in summary["sources"] if s["path"].endswith("/evaluation_library.json"))
    if identity(Path(library_source["path"])) != library_source:
        raise ValueError("Original evaluation library changed")
    library = json.loads(Path(library_source["path"]).read_text())["splits"]["train"]
    indices = {row["name"]:i for i, row in enumerate(library)}
    with rows_path.open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    groups = {}
    for row in source_rows:
        groups.setdefault((row["label"], int(row["repeat"])), []).append(row)
    trace_sources = {Path(row["path"]).name: row for row in summary["trace_sources"]
                     if row["path"].endswith(".npz")}
    rows = []
    for (label, repeat), clips in groups.items():
        source = trace_sources[f"{label}_repeat{repeat}.npz"]
        if identity(Path(source["path"])) != source:
            raise ValueError("Saved trace changed since the state audit")
        with np.load(source["path"], allow_pickle=False) as archive:
            targets = archive["targets"]
        for clip in clips:
            index = indices[clip["clip"]]
            values = cost_parts(targets[:int(clip["steps"]), index], summary["action_scales"],
                                summary["action_rate_weight"], summary["control_dt"])
            parts = {key:float(value.mean()) for key, value in values.items()}
            base_total = parts["base_controlled14"] + parts["base_upper8"]
            ppo_total = parts["ppo_controlled14"] + parts["ppo_upper8"]
            np.testing.assert_allclose(base_total, float(clip["base_saved_action_rate_cost_mean"]), atol=1e-9, rtol=2e-6)
            np.testing.assert_allclose(ppo_total, float(clip["ppo_saved_smooth_cost_mean"]), atol=1e-9, rtol=2e-6)
            rows.append(dict(label=label, repeat=repeat, clip=clip["clip"], **parts,
                base_upper_cost_fraction=parts["base_upper8"]/base_total if base_total else None,
                ppo_upper_cost_fraction=parts["ppo_upper8"]/ppo_total if ppo_total else None))
        if identity(Path(source["path"])) != source:
            raise ValueError("Saved trace changed during attribution")
    write_csv(args.review / "fixed_target_cost_attribution.csv", rows)
    value = dict(status="SAVED_TARGET_COST_ATTRIBUTION_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        sources=[identity(p) for p in (Path(__file__), summary_path, rows_path)], trace_sources=list(trace_sources.values()),
        rows=len(rows), reconstruction_of_prior_total_costs_passed=True,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        limits=["First transition excluded, matching the preceding saved-state audit.",
                "14 residual-controlled joints include 12 legs and 2 waist; 8 upper joints have no direct residual output.",
                "Upper targets still depend on future states; this partition does not prove zero indirect controllability.",
                "Absolute penalty magnitudes are not the marginal cost of an improved step or proof of a training cause."])
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    plot(args.review, source_rows, rows)
    print(json.dumps(dict(status=value["status"], rows=len(rows), plot=str(args.review / "reward_signal_review.png"))))


def plot(review, signals, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walking = [r for r in signals if r["clip"].startswith("walk_")]
    labels = list(dict.fromkeys(r["label"] for r in walking))
    x = np.arange(len(labels))
    def stats(rows, key):
        values = [[float(r[key]) for r in rows if r["label"] == label] for label in labels]
        means = np.array([np.mean(v) for v in values])
        return means, np.array([means - np.array([min(v) for v in values]),
                                np.array([max(v) for v in values]) - means])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    colors = ("#1b8a72", "#bc4737")
    for ax, keys, title in ((axes[0, 0], ("base_perfect_sole_z_gain_mean", "base_saved_action_rate_cost_mean"),
                             "SuperMimic: Foot-Z Signal And Action-Rate Cost"),
                            (axes[0, 1], ("ppo_perfect_swing_z_gain_mean", "ppo_saved_smooth_cost_mean"),
                             "PPO: Swing-Z Signal And Smoothness Cost")):
        for offset, key, color, text in zip((-.18, .18), keys, colors,
                                           ("Perfect-height gain (analytic)", "Observed target cost")):
            means, error = stats(walking, key)
            ax.bar(x+offset, means, .34, yerr=error, color=color, capsize=3, label=text)
        ax.set_yscale("log")
        ax.set_ylabel("Reward per 20ms step (log scale)")
        ax.set_title(title, fontsize=11, pad=31)
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    ax = axes[1, 0]
    for offset, key, color, text in zip((-.18, .18),
            ("reference_swing_loaded_fraction", "reference_swing_below5mm_fraction"), colors,
            ("Native force >5N", "Sole <=5mm")):
        means, error = stats(walking, key)
        ax.bar(x+offset, means*100, .34, yerr=error*100, color=color, capsize=3, label=text)
    ax.set_title("During Declared Reference Swing", fontsize=11, pad=31)
    ax.set_ylabel("Foot-frame fraction (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    ax = axes[1, 1]
    walking_costs = [r for r in costs if r["clip"].startswith("walk_")]
    for offset, key, color, text in zip((-.18, .18),
            ("base_upper_cost_fraction", "ppo_upper_cost_fraction"), colors,
            ("Native action-rate penalty", "PPO smoothness penalty")):
        means, error = stats(walking_costs, key)
        ax.bar(x+offset, means*100, .34, yerr=error*100, color=color, capsize=3, label=text)
    ax.set_title("Penalty Share From 8 Upper Joints", fontsize=11, pad=31)
    ax.set_ylabel("Penalty share (%)")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
    for ax in axes.flat:
        ax.set_xticks(x, ["Initial", "301s", "601s", "1201s"])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Q1 Saved Walking States: Reward Sensitivity, Not A New Training Result", fontsize=13)
    fig.supxlabel("Means across two fixed walks and three repeats; whiskers show range. Absolute cost is not marginal cost.", fontsize=9)
    fig.savefig(review / "reward_signal_review.png", dpi=170)
    fig.savefig(review / "reward_signal_review.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--review", type=Path, required=True)
    run(parser.parse_args())
