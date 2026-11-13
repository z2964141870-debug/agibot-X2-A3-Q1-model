"""Read saved CPU/FP32/FP16 evidence and make a six-panel diagnostic."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from script.x2_fgp_realtime.assets import file_digest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--acceleration", type=Path, required=True)
    parser.add_argument("--mixed", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing analysis")
    folders = [args.baseline, args.acceleration / "probe_fp32_r1", args.acceleration / "probe_fp16_r1"]
    names, colors = ["CPU ONNX", "TRT FP32", "TRT FP16"], ["#306c9b", "#a9473b", "#22866b"]
    if args.mixed is not None:
        folders.append(args.mixed)
        names.append("TRT mixed")
        colors.append("#7b4a9e")
    reports = [json.loads((p / "report.json").read_text()) for p in folders]
    for report in reports:
        if report["output_frames"] != 32 or report["fixture_sha256"] != reports[0]["fixture_sha256"]:
            raise ValueError("Evidence identity mismatch")
    args.output.mkdir(parents=True)
    values = [np.asarray([row["total_ms"] for row in report["rows"]]) for report in reports]
    diagnostics, inputs = [], []
    for folder, report in zip(folders, reports):
        for filename in ("report.json", "outputs.npz"):
            path = folder / filename
            inputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=file_digest(path)))
        if "precision" in report:
            path = folder / "cano_comparison.npz"
            inputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=file_digest(path)))
            with np.load(path, allow_pickle=False) as comparison:
                residual = np.abs(comparison["gpu"]-comparison["cpu"]).reshape(32, -1).max(axis=1)
            with np.load(folder / "outputs.npz", allow_pickle=False) as saved, np.load(args.baseline / "outputs.npz", allow_pickle=False) as original:
                relative = saved["pose"].astype(np.float64) @ np.swapaxes(original["pose"].astype(np.float64), -1, -2)
                pose = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))).max(axis=1)
                root = np.linalg.norm(saved["translation"]-original["translation"], axis=1)*1000
            diagnostics.append((residual, pose, root))
    summary = dict(status="SAVED_ACCELERATION_ANALYSIS_PASSED", inputs=inputs,
        cases=[dict(name=name, timing_all_ms=r["timing_all_ms"],
                    timing_after_first2_ms=r["timing_after_first2_ms"],
                    agreement_status=r.get("agreement_status", "REFERENCE"),
                    agreement_metrics=r.get("agreement_metrics")) for name, r in zip(names, reports)],
        limitation="32-frame unpaced recording, no BLE/browser/network/robot timing or human GT. CPU baseline predates reboot; not a randomized concurrent performance trial.",
        new_model_calls=0, hardware_commands=0, training_updates=0)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (args.output / "timing.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_frame", *[name+" ms" for name in names]])
        for i in range(32):
            writer.writerow([29+i, *(series[i] for series in values)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    frame = np.arange(29, 61)
    for label, color, series in zip(names, colors, values):
        axes[0, 0].plot(frame, series, label=label, color=color, lw=1.8)
    axes[0, 0].axhline(1000/30, color="#555555", ls="--", label="30 Hz budget")
    axes[0, 0].set(title="Full pipeline per output", xlabel="Source frame", ylabel="ms")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].boxplot([v[2:] for v in values], tick_labels=names, showfliers=True)
    axes[0, 1].axhline(1000/30, color="#555555", ls="--")
    axes[0, 1].set(title="After first two outputs", ylabel="Full pipeline ms")
    stages = ("cano", "pose", "physics")
    width = .8/len(reports)
    for i, (report, name, color) in enumerate(zip(reports, names, colors)):
        medians = [np.median([row["author_exclusive_stage_ms"][key] for row in report["rows"][2:]]) for key in stages]
        axes[0, 2].bar(np.arange(3)+(i-(len(reports)-1)/2)*width, medians, width=width, label=name, color=color)
    axes[0, 2].set_xticks(np.arange(3), ["CANO", "LIP", "GP + physics"])
    axes[0, 2].set(title="Exclusive stage medians", ylabel="ms")
    axes[0, 2].legend(fontsize=8)
    for (residual, pose, root), label, color in zip(diagnostics, names[1:], colors[1:]):
        axes[1, 0].semilogy(frame, np.maximum(residual, 1e-12), label=label, color=color)
        axes[1, 1].plot(frame, pose, label=label, color=color)
        axes[1, 2].plot(frame, root, label=label, color=color)
    for ax, title, unit in zip(axes[1], ["CANO residual max difference", "Full-chain pose vs CPU", "Full-chain root vs CPU"], ["absolute", "max joint angle (deg)", "Euclidean difference (mm)"]):
        ax.set(title=title, xlabel="Source frame", ylabel=unit)
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("X2 Orin: recorded-input CANO acceleration | 32 outputs | no hardware control", fontsize=14)
    fig.savefig(args.output / "six_panel.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
