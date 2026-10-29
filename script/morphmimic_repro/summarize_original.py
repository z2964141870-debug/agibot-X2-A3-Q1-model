#!/usr/bin/env python3
"""Export upstream TensorBoard scalars and the six-panel training figure."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(args.run), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows = {}
    tags = sorted(accumulator.Tags()["scalars"])
    for tag in tags:
        for event in accumulator.Scalars(tag):
            rows.setdefault(event.step, {"iteration": event.step})[tag] = event.value
    if not rows:
        raise ValueError("No upstream scalar events found")
    ordered = [rows[key] for key in sorted(rows)]
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", *tags])
        writer.writeheader()
        writer.writerows(ordered)

    x = np.array([row["iteration"] for row in ordered])

    def series(tag):
        return np.array([row.get(tag, np.nan) for row in ordered], dtype=float)

    failures = series("collect/failures")
    finished = series("collect/episodes_finished")
    ratio = np.full_like(failures, np.nan)
    np.divide(failures, finished, out=ratio, where=finished > 0)
    panels = [
        ("Normalized Episode Return", series("collect/mean_norm_return")),
        ("Hard Failures / Finished Episodes", ratio),
        ("Finished Episode Length", series("collect/mean_episode_length")),
        ("Mean Step Reward (Upstream Units)", series("collect/mean_reward")),
        ("World Model 8-Step Error", series("wm/pred_err_8step")),
        ("Imagined Survival at H32", series("imag/survival_H")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (title, y) in zip(axes.flat, panels):
        smooth = np.full_like(y, np.nan)
        for index, step in enumerate(x):
            start = np.searchsorted(x, step - 499)
            window = y[start:index + 1]
            finite = window[np.isfinite(window)]
            if len(finite):
                smooth[index] = finite.mean()
        ax.plot(x, y, color="#5691b9", alpha=0.5, linewidth=0.8)
        ax.plot(x, smooth, color="#1e624c", linewidth=2)
        ax.axvline(500, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training Iteration")
        ax.grid(alpha=0.2)
        if not np.isfinite(y).any():
            ax.text(0.5, 0.5, "Metric Not Recorded", ha="center", transform=ax.transAxes)
    fig.suptitle("Original MorphMimic G1 | 64 envs | WM/imag batch 128 | H32\n"
                 "Thin: logged windows; thick: trailing 500 iterations; WM-only through iteration 500",
                 fontsize=12)
    fig.savefig(args.output / "training.png", dpi=160)
    fig.savefig(args.output / "training.pdf")
    plt.close(fig)

    event_sources = []
    for path in sorted(args.run.glob("events.out.tfevents.*")):
        event_sources.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    receipt = {
        "event_sources": event_sources,
        "log_rows": len(ordered), "tags": tags,
        "first_iteration": int(x[0]), "last_iteration": int(x[-1]),
        "smoothing": "Mean of available logged windows whose iteration is in [current-499,current]; not fabricated per-iteration observations.",
        "failure_ratio": "Upstream failures divided by episodes_finished in the same log window; missing when denominator is zero.",
        "reward_scale": "Upstream raw units; maximum reward was not separately logged.",
        "policy_quality_evaluation": False,
    }
    (args.output / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"rows": len(ordered), "tags": len(tags), "output": str(args.output)}))


if __name__ == "__main__":
    main()
