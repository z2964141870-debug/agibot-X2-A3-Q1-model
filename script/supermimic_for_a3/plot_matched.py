#!/usr/bin/env python3
"""Render matched physics results only after the run audit has passed."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from supermimic.a3_assets import identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "TRAINING_AND_MATCHED_EVALUATION_VERIFIED":
        raise ValueError("A verified matched evaluation is required")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = audit["rows"]
    y = np.arange(len(rows))
    labels = [r["clip"].replace("_", " ") + (" [train]" if r["split"] == "training" else " [held-out]") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, key, title in zip(axes, ("normalized_return", "full_clip_completion"),
                              ("Full-clip normalized return", "Full-clip completion")):
        for offset, model, label, color in ((-.18, "initial", "Initialization", "#8c959f"),
                                            (.18, "trained", "Trained", "#16836b")):
            ax.barh(y + offset, [r[key][model] for r in rows], height=.32, color=color, label=label)
        ax.set_title(title, fontsize=11, pad=12)
        ax.xaxis.set_major_formatter(PercentFormatter(1., decimals=1))
        ax.tick_params(labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.2)
        ax.axvline(0., color="#535b63", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("SuperMimic for A3: matched physics evaluation", fontsize=13)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.67, .93), ncol=2, frameon=False)
    counts = []
    for split in ("training", "heldout"):
        g = audit["groups"][split]
        counts.append(f"{split}: {g['initial']['failures']} -> {g['trained']['failures']} / {g['clips']}")
    fig.text(.02, .055, "Terminations (initial -> trained): " + "; ".join(counts), fontsize=9)
    fig.text(.02, .02, "Full reference duration remains in the denominator after early failure. Reserved clips share the training data source.", fontsize=8)
    fig.tight_layout(rect=(0., .10, 1., .88))
    paths = [args.output / "matched_evaluation.png", args.output / "matched_evaluation.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    result = {"audit": identity(args.audit), "plot_entry": identity(__file__),
              "figures": [identity(path) for path in paths], "storage_status": "LOCAL_ONLY"}
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
