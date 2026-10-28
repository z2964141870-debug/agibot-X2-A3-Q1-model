"""Publish saved UR02 attribution/replay measurements without more physics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    args = parser.parse_args()
    root = args.trial
    if (root / "delivery.json").exists():
        raise FileExistsError("UR02 saved delivery already exists")
    attribution = json.loads((root / "attribution.json").read_text())
    physical = json.loads((root / "physical_replay.json").read_text())
    rows = json.loads((root / "evaluation_library.json").read_text())["splits"]["train"]
    names = ("replay_RF11", "replay_reference_arms")
    labels = ("Replay RF11 targets", "Replay lower + reference arms")
    colors = ("#5d6268", "#008675")
    summaries = {name: [json.loads((root / f"{name}_repeat{i}.json").read_text())
                        for i in range(3)] for name in names}
    records = []
    for repeat in attribution["repeats"]:
        for direction, measurements in repeat["validation"].items():
            records.extend(dict(protocol="saved_history", condition=direction, repeat=repeat["repeat"],
                                metric=key, candidate=value, baseline="", ratio="")
                           for key, value in measurements.items())
    for repeat in physical["repeats"]:
        for protocol in ("arm_intervention", "playback_vs_saved_closed_loop"):
            comparison = repeat[protocol]
            records.extend(dict(protocol=protocol, condition="physical_replay", repeat=repeat["repeat"],
                                metric=key, **pair, ratio=comparison["ratios"][key])
                           for key, pair in comparison["measurements"].items())
    with (root / "coupling_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    axis = axes[0, 0]
    fields = ("position", "velocity", "previous_action", "all_upper")
    for offset, direction, color in ((-.13, "inject_into_RF11", colors[0]),
                                     (.13, "remove_from_reference", colors[1])):
        values = np.array([[r["validation"][direction][f] for f in fields]
                           for r in attribution["repeats"]])
        x = np.arange(len(fields)) + offset
        axis.bar(x, values.mean(0), width=.25, color=color, alpha=.8,
                 label=direction.replace("_", " "))
        for i in range(3):
            axis.scatter(x, values[i], color=color, s=10)
    axis.set_xticks(range(4), ("Upper q", "Upper qd", "Upper prev.a", "All upper"), fontsize=9)
    axis.set_title("Synthetic upper-input swap: lower target RMS (rad)", fontsize=10)
    axis.legend(fontsize=7)
    axis.grid(axis="y", alpha=.2)
    for axis, metric, title in zip(axes[0, 1:], ("root_xy_rmse", "upper_rmse"),
            ("Physical playback: paired root XY RMSE (m)", "Physical playback: paired upper RMSE (rad)")):
        for x, name, key, color in zip(range(2), names, ("baseline", "candidate"), colors):
            values = [r["arm_intervention"]["measurements"][metric][key] for r in physical["repeats"]]
            axis.scatter(np.full(3, x) + np.linspace(-.05, .05, 3), values, color=color)
            axis.plot([x - .12, x + .12], [np.mean(values)] * 2, color=color)
        axis.set_xticks(range(2), ("RF11 target replay", "Reference arms"), fontsize=9)
        axis.set_title(title, fontsize=10)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axis = axes[1, 0]
    for x, name, color in zip(range(2), names, colors):
        failures = [v["summary"]["validation"]["failures"] for v in summaries[name]]
        axis.scatter(np.full(3, x) + np.linspace(-.05, .05, 3), failures, color=color)
    axis.set_xticks(range(2), ("RF11 target replay", "Reference arms"), fontsize=9)
    axis.set_ylim(-.3, 30.5)
    axis.set_title("Physical playback: early terminations /30", fontsize=10)
    axis.grid(alpha=.2)
    index = next(i for i, r in enumerate(rows) if r["name"] == "triumph_two_handed_001__A058")
    axis = axes[1, 1]
    for name, label, color in zip(names, labels, colors):
        with np.load(root / f"{name}_repeat0.npz", allow_pickle=False) as trace:
            count = int(trace["steps"][index])
            trajectory = trace["states"][:count, index, :2]
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, label=label)
            if bool(trace["failed"][index]):
                axis.scatter(*trajectory[-1], color=color, marker="x")
            if name == names[0]:
                ref = trace["reference_p"][:int(trace["reference_steps"][index]), index, :2]
                axis.plot(ref[:, 0], ref[:, 1], "k--", label="Reference")
    axis.set_title("Fixed raised hands: root XY, repeat0", fontsize=10)
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=.2)
    axis.legend(fontsize=7)
    axis = axes[1, 2]
    for x, name, color in zip(range(2), names, colors):
        values = [r["arm_intervention"]["measurements"]["loaded_slip"][key]
                  for r in physical["repeats"] for key in ["baseline" if x == 0 else "candidate"]]
        axis.scatter(np.full(3, x) + np.linspace(-.05, .05, 3), values, color=color)
    axis.set_xticks(range(2), ("RF11 target replay", "Reference arms"), fontsize=9)
    axis.set_title("Physical playback: force-loaded slip (m/s)", fontsize=10)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=.2)
    figure.suptitle("UR02 | Frozen RF11 | 0 optimizer | Synthetic attribution and open-loop lower-command intervention", fontsize=12)
    for suffix in ("png", "pdf"):
        figure.savefig(root / ("balance_coupling." + suffix), dpi=160)
    plt.close(figure)
    save(root / "delivery.json", dict(status="COMPLETED", new_physics_steps=0, optimizer_updates=0,
         sources=[identity(root / n) for n in ("attribution.json", "physical_replay.json")],
         artifacts=[identity(root / n) for n in ("balance_coupling.png", "balance_coupling.pdf", "coupling_metrics.csv")],
         boundary="This is a nontraining causal diagnostic, not a training dashboard; no500-iteration smoothing, WM, imagined survival, or new54-second training video is applicable."))


if __name__ == "__main__":
    main()
