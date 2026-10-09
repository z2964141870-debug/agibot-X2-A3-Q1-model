#!/usr/bin/env python3
"""Collect verified GMR results and plot amplitude/root diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_input import identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unbounded", type=Path, required=True)
    parser.add_argument("--bounded", type=Path, required=True)
    parser.add_argument("--validated", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists() or args.plot.exists():
        parser.error("Preserve existing summary and plot")
    report = {"usage": "TEST_ONLY", "training_allowed": False, "collector": identity(__file__),
              "scope": "KINEMATIC_DIAGNOSTICS_NOT_DYNAMIC_BALANCE", "sources": {}, "results": {}}
    for name in ("unbounded", "bounded", "validated", "calibration"):
        path = getattr(args, name)
        result = json.loads(path.read_text())
        for case in result["cases"]:
            if case["status"] != "COMPLETE":
                raise ValueError("Incomplete comparison")
            if name in ("unbounded", "bounded"):
                for artifact in case["artifacts"]:
                    if identity(artifact["path"])["sha256"] != artifact["sha256"]:
                        raise ValueError("GMR artifact changed")
            if name == "validated":
                for artifact in (case["reference"], case["supported_replay"]):
                    if identity(artifact["path"])["sha256"] != artifact["sha256"]:
                        raise ValueError("Validated reference or replay changed")
        report["sources"][name] = identity(path)
        report["results"][name] = result
    calibration = {(r["case"], r["mode"]): r for r in report["results"]["calibration"]["cases"]}
    bounded = {r["id"]: r for r in report["results"]["bounded"]["cases"]}
    kick = calibration[("amass_kick", "gravity_fixed")]
    standing = calibration[("amass_kick", "standing_pair")]
    values = [kick["source_leg_geometry"]["right"]["knee_flexion_range_deg"],
              kick["q1_leg_geometry"]["right"]["knee_flexion_range_deg"],
              standing["q1_leg_geometry"]["right"]["knee_flexion_range_deg"],
              bounded["amass_kick"]["q1_leg_geometry"]["right"]["knee_flexion_range_deg"]]
    jog_old = calibration[("bones_jog", "legacy")]
    jog_new = calibration[("bones_jog", "gravity_fixed")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150, layout="constrained")
    groups = [(axes[0], values, ["Human", "First frame", "Standing pair", "GMR Q1"],
               "AMASS kick: right knee range", "Geometric range (deg)"),
              (axes[1], [100 * jog_old["source_scaled_z_range_m"], 100 * jog_old["q1_root_z_range_m"], 100 * jog_new["q1_root_z_range_m"]],
               ["Scaled source", "Old alignment", "Gravity fixed"], "Jog: world root vertical range", "Vertical range (cm)")]
    for ax, numbers, labels, title, ylabel in groups:
        bars = ax.bar(range(len(numbers)), numbers, width=.55, color=["#25766d", "#b44142", "#bf9232", "#65568f"][:len(numbers)])
        ax.set_xticks(range(len(numbers)), labels, fontsize=9)
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(numbers) * 1.22)
        ax.yaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, number in zip(bars, numbers):
            ax.annotate(f"{number:.1f}", (bar.get_x() + bar.get_width() / 2, number),
                        xytext=(0, 5), textcoords="offset points", ha="center")
    fig.suptitle("Q1 calibration and GMR diagnostics", fontsize=13)
    fig.supxlabel("Robot knee outputs limited to 2 rad/s; targets and initialization differ. Fixed support does not verify walking.", fontsize=9)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot)
    plt.close(fig)
    report["plot"] = identity(args.plot)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified GMR 8 cases and supported PD 4 cases; manifest {args.manifest}")


if __name__ == "__main__":
    main()
