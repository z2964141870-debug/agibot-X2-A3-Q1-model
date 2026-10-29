"""Plot the paired saved RF16 sole windows and final-substep force snapshots."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analyze_q1_lift_windows import DT, FORCE_N, LIFT_M, read_series, windows
from prepare_q1_gait_review import FIXED, identity


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if args.output.exists():
        raise FileExistsError(args.output)
    saved = json.loads(args.summary.read_text())
    if (saved["status"] != "SAVED_PAIRED_LIFT_WINDOWS_ANALYZED"
            or saved["baseline_label"] != "initial_static075"
            or saved["diagnostic_lift_threshold_m"] != LIFT_M
            or saved["diagnostic_force_threshold_N"] != FORCE_N):
        raise ValueError("Expected the paired RF16 saved-window analysis")
    paths = [Path(row["path"]) for row in saved["sources"]]
    if [identity(path) for path in paths] != saved["sources"]:
        raise ValueError("Saved analysis input identities changed")
    before = identity(args.summary)
    actual = read_series(paths[0], ("label", "repeat", "clip", "side"),
                         ("sole_height_m", "native_force_N"), first_tick=1)
    reference = read_series(paths[1], ("clip", "side"), ("sole_height_m",),
                            first_tick=1, reference_only=True)
    labels = (("initial_static075", "Static arms 0.50 / no learned head", "#62676d"),
              ("stage_02", "Contact head: 601 s", "#008878"),
              ("stage_03", "Contact head: 1201 s", "#bc4040"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, (clip, title) in enumerate(zip(FIXED[:2], ("Forward walk: stop", "Forward walk: start"))):
        count = min(len(series) for key, series in actual.items() if key[2] == clip)
        ticks = range(1, count+1)
        seconds = np.arange(1, count+1) * DT
        for column, side in enumerate(("left", "right")):
            axis = axes[row, column]
            target = np.array([reference[clip, side][t]["sole_height_m"] for t in ticks])
            for start, end in windows(target > LIFT_M):
                axis.axvspan(seconds[start]-DT/2, seconds[end-1]+DT/2, color="#bbc5d1", alpha=.45)
                if side == "right":
                    axes[row, 2].axvspan(seconds[start]-DT/2, seconds[end-1]+DT/2,
                                        color="#bbc5d1", alpha=.45)
            axis.plot(seconds, target*1000, color="#111111", linestyle="--", linewidth=1.4,
                      label="Kinematic reference")
            for label, display, color in labels:
                heights, forces = [], []
                for repeat in range(3):
                    series = actual[label, str(repeat), clip, side]
                    heights.append([series[t]["sole_height_m"]*1000 for t in ticks])
                    forces.append([series[t]["native_force_N"] > FORCE_N for t in ticks])
                values = np.array(heights)
                axis.fill_between(seconds, values.min(0), values.max(0), color=color, alpha=.10)
                axis.plot(seconds, np.median(values, axis=0), color=color, linewidth=1.1, label=display)
                if side == "right":
                    axes[row, 2].plot(seconds, np.mean(forces, axis=0), color=color,
                                      linewidth=.9, alpha=.85)
            axis.axhline(0, color="#222222", linewidth=.6)
            axis.set_title(title + " / " + side + " sole", fontsize=11)
            axis.set_ylabel("Lowest collision sole above floor (mm)")
        force_axis = axes[row, 2]
        force_axis.set_title(title + " / right force snapshot", fontsize=11)
        force_axis.set_ylabel("Fraction of 3 repeats with force >5 N")
        force_axis.set_ylim(-.04, 1.04)
        force_axis.set_yticks((0, 1/3, 2/3, 1), ("0/3", "1/3", "2/3", "3/3"))
        for axis in axes[row]:
            axis.set_xlabel("Seconds from saved clip start")
            axis.grid(alpha=.16)
            axis.tick_params(labelsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#bbc5d1", alpha=.45))
    names.append("Reference sole >20 mm")
    fig.legend(handles, names, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("RF16: sole clearance at the reference lift times\n"
                 "Height: 3-repeat median + range, no time smoothing. Force: final 5 ms snapshot only.", fontsize=13)
    if before != identity(args.summary) or [identity(path) for path in paths] != saved["sources"]:
        raise ValueError("Saved evidence changed while plotting")
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / ("paired_lift_windows." + suffix), dpi=160)
    plt.close(fig)
    result = dict(status="SAVED_PAIRED_LIFT_WINDOWS_PLOTTED",
                  captured_utc=datetime.now(timezone.utc).isoformat(),
                  analysis=before, tool=identity(Path(__file__)),
                  figures=[identity(args.output / ("paired_lift_windows." + suffix))
                           for suffix in ("png", "pdf")],
                  new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
                  user_accepted_checkpoint=None, limits=saved["limits"])
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "figures": len(result["figures"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    plot(parser.parse_args())
