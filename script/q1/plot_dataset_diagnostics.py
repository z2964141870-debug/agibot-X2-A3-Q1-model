#!/usr/bin/env python3
"""Plot two measured retargeting failures from the verified result manifest."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_input import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Preserve existing plot evidence")
    rows = {row["id"]: row for row in json.loads(args.manifest.read_text())["cases"]}
    kick, jog = rows["amass_kick"], rows["bones_jog"]
    knee = [kick["conversion"]["source_leg_geometry"]["right"]["knee_flexion_range_deg"],
            kick["metrics"]["q1_leg_geometry"]["right"]["knee_flexion_range_deg"]]
    vertical = [100 * jog["root_gravity_diagnostic"]["source_world_z_range_scaled_m"],
                100 * jog["root_gravity_diagnostic"]["q1_kinematic_root_z_range_m"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=150, layout="constrained")
    for ax, values, title, unit, labels in [
        (axes[0], knee, "AMASS kick: right knee motion", "Flexion range (deg)", ["SMPL-X source", "Q1 IK"]),
        (axes[1], vertical, "BONES-SEED jog: root height", "Vertical range (cm)", ["Source, size-scaled", "Q1 IK"])]:
        bars = ax.bar([0, 1], values, width=0.52, color=["#25766d", "#b44142"])
        ax.set_xticks([0, 1], labels)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values) * 1.24)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=11)
    fig.suptitle("Current Q1 retargeting: kinematics only", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    evidence = {"usage": "TEST_ONLY", "training_allowed": False,
                "source_manifest": identity(args.manifest), "plotter": identity(__file__),
                "image": identity(args.output), "knee_range_deg": knee,
                "root_vertical_range_cm": vertical, "hardware_control": False,
                "scope": "DIAGNOSTIC_GEOMETRY_NOT_DYNAMIC_BALANCE"}
    args.output.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
