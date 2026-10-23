#!/usr/bin/env python3
"""Plot recorded Q1 collection and model-fit diagnostics without loading weights."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from supermimic.q1_assets import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = args.run / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    if not rows:
        raise ValueError("No recorded training metrics")
    x = np.array([row["env_steps"] / 1000 for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    figure.suptitle("Q1 R01: Training Diagnostics", fontsize=15)
    axes[0].plot(x, [row["reward_mean"] for row in rows], color="#137f70", linewidth=1.5)
    axes[0].set_ylabel("Collection reward / step")
    axes[0].set_title("Random reference starts and domain randomization; not full-clip evaluation", fontsize=9)
    axes[1].plot(x, [row["joint_rmse_rad"] for row in rows], color="#b8424e", linewidth=1.5)
    axes[1].set_ylabel("Collection joint RMSE (rad)")
    for key, label, color in (("wm/nse_1step", "One-step model error", "#187ca6"),
                              ("wm/nse_kstep", "Eight-step model error", "#9b6542")):
        selected = [row for row in rows if key in row]
        axes[2].plot([row["env_steps"] / 1000 for row in selected], [row[key] for row in selected],
                     label=label, color=color, linewidth=1.5)
    axes[2].set_ylabel("Normalized squared error")
    axes[2].set_title("Online training minibatches; the data distribution changes with the policy", fontsize=9)
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].set_xlabel("New physics control steps (thousands)")
    for axis in axes:
        axis.grid(axis="y", color="#e1e5e8", linewidth=.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
    destination = args.output / "training_diagnostics.png"
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    receipt = {"status": "RECORDED_TRAINING_CURVES_PLOTTED", "metrics": identity(metrics),
               "rows": len(rows), "last_recorded_step": rows[-1]["step"],
               "artifacts": [identity(destination)], "new_physics_steps": 0, "new_optimizer_updates": 0,
               "source": identity(__file__), "storage_status": "LOCAL_ONLY"}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
