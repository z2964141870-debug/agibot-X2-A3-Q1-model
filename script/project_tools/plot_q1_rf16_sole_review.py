"""Plot saved collision-sole clearances without policy or physics execution."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_q1_gait_review import FIXED, identity


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if args.output.exists():
        raise FileExistsError(args.output)
    with args.actual_csv.open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    with args.reference_csv.open(newline="") as stream:
        reference = [row for row in csv.DictReader(stream) if row["case"] == "reference"]
    labels = (("initial_static075", "Static arms, no learned head", "#5d6268"),
              ("stage_02", "Contact head: 601 s", "#087f73"),
              ("stage_03", "Contact head: 1201 s", "#b44747"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    titles = ("Forward walk: stop", "Forward walk: start", "Raised hands")
    for column, (clip, title) in enumerate(zip(FIXED, titles)):
        for foot, side in enumerate(("left", "right")):
            axis = axes[foot, column]
            rows = [row for row in reference if row["clip"] == clip and row["side"] == side]
            seconds = np.array([float(row["seconds"]) for row in rows])
            height = np.array([float(row["sole_height_m"]) * 1000 for row in rows])
            if len(rows) < 2 or not np.isfinite(height).all():
                raise ValueError("Missing finite reference curve")
            axis.plot(seconds, height, color="#235ca3", linewidth=1.3, label="Kinematic reference")
            for label, display, color in labels:
                repeats = []
                for repeat in range(3):
                    measured = [row for row in actual if row["label"] == label
                                and int(row["repeat"]) == repeat and row["clip"] == clip
                                and row["side"] == side]
                    times = np.array([float(row["seconds"]) for row in measured])
                    values = np.array([float(row["sole_height_m"]) * 1000 for row in measured])
                    if not np.array_equal(times, seconds) or not np.isfinite(values).all():
                        raise ValueError("Physical and reference full-clip times differ")
                    repeats.append(values)
                values = np.stack(repeats)
                axis.fill_between(seconds, values.min(0), values.max(0), color=color, alpha=.10)
                axis.plot(seconds, values[0], color=color, linewidth=1., label=display)
            axis.axhline(0, color="#222222", linewidth=.7)
            axis.set_title(title + " / " + side + " foot", fontsize=11)
            axis.set_xlabel("Seconds")
            axis.set_ylabel("Lowest collision sole above floor (mm)")
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=9)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("RF16: actual collision-sole clearance\n"
                 "Raw 20 ms saved poses: line = repeat 0, shading = range of all 3 repeats", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / ("collision_sole_clearance." + suffix), dpi=160)
    plt.close(fig)
    receipt = dict(status="RF16_SOLE_CURVES_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
                   sources=[identity(path) for path in (args.actual_csv, args.reference_csv, Path(__file__))],
                   figures=[identity(args.output / ("collision_sole_clearance." + suffix))
                            for suffix in ("png", "pdf")],
                   repeats=3, fixed_clips=3, smoothing=False,
                   new_physical_steps=0, policy_inference_calls=0, optimizer_updates=0,
                   user_accepted_checkpoint=None,
                   limits=["Exact active collision geometry, not ankle height.",
                           "Small negative clearances can occur in soft-contact simulation.",
                           "Clearance curves alone do not establish natural gait or isolate learning failure."])
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: receipt[key] for key in ("status", "repeats", "fixed_clips")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("actual-csv", "reference-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    plot(parser.parse_args())
