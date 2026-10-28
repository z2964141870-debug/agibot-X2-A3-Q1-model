"""Tabulate completed zero-optimizer probes without simulation or policy selection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from prepare_q1_gait_review import ARRAYS, FIXED, GAIT, RAISE, VALIDATION, identity, span, write_csv


def prepare(trial, output, plot=False):
    if output.exists():
        raise FileExistsError(output)
    status = json.loads((trial / "status.json").read_text())
    result = json.loads((trial / "results.json").read_text())
    plan = json.loads((trial / "plan.json").read_text())
    if status["status"] != "COMPLETED" or result["status"] != "COMPLETED" or result["optimizer_steps"] != 0:
        raise ValueError("Require completed zero-optimizer results")
    rows = result["rows"]
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation")}
    if len({row["name"] for row in rows}) != len(rows) or counts["validation"] != 30:
        raise ValueError("Unexpected fixed library identities")
    if (result["scope"] == "full330" and counts != {"train": 300, "validation": 30}
            or result["scope"] == "quick33" and counts != {"train": 3, "validation": 30}):
        raise ValueError("Unexpected original-library scope")
    split_rows, fixed_rows, per_clip, tracking_rows, walk_rows, stages, sources = [], [], [], [], [], [], []
    for candidate in result["candidates"]:
        label, repeats = candidate["candidate"], []
        for repeat in range(3):
            path = trial / f"{label}_repeat{repeat}.json"
            item = json.loads(path.read_text())
            clips = {clip["name"]: clip for clip in item["per_clip"]}
            if (item["candidate"] != label or item["repeat"] != repeat or item["seed"] != 32000 + repeat
                    or set(clips) != {row["name"] for row in rows}
                    or any(item["summary"][split]["clips"] != count for split, count in counts.items())):
                raise ValueError(f"Unexpected repeat or clip contract: {path}")
            sources.append(identity(path))
            prefix = {"candidate": label, "repeat": repeat, "configured_seed": item["seed"], "optimizer_updates": 0}
            for split in counts:
                for metric, unit in VALIDATION.items():
                    split_rows.append({**prefix, "split": split, "metric": metric,
                                       "unit": unit, "value": item["summary"][split][metric]})
            per_clip.extend({**prefix, **clip} for clip in clips.values())
            for name in FIXED:
                gait, clip = item["gait"][name], clips[name]
                if (gait["steps"] != clip["steps"] or gait["failed"] != clip["failed"]
                        or not 0 < clip["steps"] <= clip["reference_steps"]):
                    raise ValueError(f"Inconsistent fixed demonstration: {name}")
                row = {**prefix, "clip": name, "reference_steps": clip["reference_steps"],
                       **{key: gait[key] for key in GAIT}, **{key: clip[key] for key in RAISE}}
                for key in ARRAYS:
                    values = gait.get(key, clip.get(key))
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError(f"Missing bilateral measurement: {name}/{key}")
                    row.update({key + "_" + side: value for side, value in zip(("left", "right"), values)})
                fixed_rows.append(row)
            paired = item.get("initial_static075")
            if paired is not None:
                # This legacy key contains the actual first candidate, static0.50 in ACTIVITY.
                tracking = paired["validation_tracking_diagnostic"]
                for metric, measured in tracking["measurements"].items():
                    tracking_rows.append({**prefix, "baseline": result["candidates"][0]["candidate"],
                                          "metric": metric, **measured,
                                          "candidate_over_baseline": tracking["ratios"][metric],
                                          "common_prefix_mean_seconds": tracking["common_prefix_mean_seconds"]})
                for index, ratios in enumerate(paired["walking_ratios"]):
                    for metric, key in (("walking_slip", "loaded_slip_mps"),
                                        ("swing_error", "swing_local_foot_rmse_m"),
                                        ("walking_tilt", "torso_tilt_rms_rad"),
                                        ("target_change", "target_delta_rms_rad")):
                        walk_rows.append({**prefix, "clip": FIXED[index], "metric": metric,
                                          "candidate_value": paired["walking_common_prefix"][index][key],
                                          "baseline_value": paired["baseline_walking_common_prefix"][index][key],
                                          "candidate_over_baseline": ratios[metric]})
            repeats.append(item)
        stages.append({"candidate": label, "validation_failures": [x["summary"]["validation"]["failures"] for x in repeats],
                       "training_failures": [x["summary"]["train"]["failures"] for x in repeats],
                       "validation_values": {key: [x["summary"]["validation"][key] for x in repeats] for key in VALIDATION},
                       "legacy_foot_continuation_diagnostic": candidate["physical_continuation_diagnostic"]})
    sources.extend(identity(trial / name) for name in ("status.json", "plan.json", "results.json"))
    value = {"status": "ZERO_OPTIMIZER_REVIEW_READY", "captured_utc": datetime.now(timezone.utc).isoformat(),
             "trial": str(trial), "scope": result["scope"], "code_commit": plan["code_commit"],
             "stages": stages, "sources": sources, "generator": identity(Path(__file__)),
             "new_physical_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
             "user_accepted_checkpoint": None, "training_authorized_by_this_report": False,
             "limits": ["All repeated raw values retained; ranges are descriptive, not confidence intervals.",
                        "Validation was searched and is not an independent held-out test.",
                        "Three configured evaluation seeds share one frozen policy, not independent training seeds.",
                        "Whole available trajectories may have different lengths; ratios use saved common prefixes.",
                        "Ankle lift is not sole clearance; support alternations are diagnostic counts.",
                        "Contact is the last solved5ms substep, not integrated20ms impulse.",
                        "Completion or reduced upper error does not certify natural walking.",
                        "Zero-optimizer figures have no training iteration axis;500iteration mean and WM/imagination metrics are N/A.",
                        "Root XY stays diagnostic; policy adoption remains the user's decision."]}
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("split_metrics.csv", split_rows), ("fixed_demos.csv", fixed_rows),
                          ("per_clip.csv", per_clip), ("tracking_common_prefix.csv", tracking_rows),
                          ("walking_common_prefix.csv", walk_rows)):
        write_csv(output / name, records)
    with (output / "summary.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# Zero-Optimizer Control Review", "", "Frozen teacher; no new training; user adoption pending.", "",
             "| Candidate | Train failures | Val failures | Completion % | Upper rad | Wrist m | Leg rad | Root m diagnostic |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in stages:
        cells = [stage["candidate"], "/".join(map(str, stage["training_failures"])),
                 "/".join(map(str, stage["validation_failures"]))]
        cells.extend(span(stage["validation_values"][key], 100. if key == "completion" else 1.)
                     for key in ("completion", "upper_rmse", "wrist_relative_rmse", "leg_rmse", "root_xy_rmse"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Fixed Demonstrations", "",
                  "| Candidate | Clip | Complete | Forward m | Slip m/s | Swing error m | Tilt rad | Drift m | Peak error m |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for stage in stages:
        for name in FIXED:
            records = [row for row in fixed_rows if row["candidate"] == stage["candidate"] and row["clip"] == name]
            complete = sum(not row["failed"] and row["steps"] == row["reference_steps"] for row in records)
            cells = [stage["candidate"], name, f"{complete}/3"]
            cells.extend(span([row[key] for row in records]) for key in
                         ("forward_displacement_m", "loaded_slip_mps", "swing_local_foot_rmse_m", "torso_tilt_rms_rad"))
            cells.extend((span([row["root_displacement_m"] for row in records]),
                          span([row["raised_peak_error"] for row in records])) if name == FIXED[2] else ("N/A", "N/A"))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Limits", ""] + ["- " + item for item in value["limits"]])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    if plot:
        draw(output, stages, fixed_rows)
    print(json.dumps({"status": value["status"], "split_rows": len(split_rows), "fixed_rows": len(fixed_rows),
                      "per_clip_rows": len(per_clip), "tracking_rows": len(tracking_rows), "walking_rows": len(walk_rows)}))


def draw(output, stages, fixed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["candidate"] for row in stages]
    colors = ["#424242", "#bf5b19", "#168378", "#6f5799", "#386cb0"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [("Validation upper RMSE", "rad", "upper_rmse"),
              ("Validation wrist relative RMSE", "m", "wrist_relative_rmse"),
              ("Validation early terminations", "clips /30", "failures")]
    for ax, (title, unit, metric) in zip(axes[0], panels):
        for i, stage in enumerate(stages):
            values = stage["validation_values"][metric]
            ax.scatter([i - .09, i, i + .09], values, color=colors[i % len(colors)], s=24)
            ax.plot([i - .18, i + .18], [statistics.mean(values)] * 2, color=colors[i % len(colors)], lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
    for ax, (title, unit, key, names) in zip(axes[1], [
            ("Loaded foot slip: stop /start walk", "m/s", "loaded_slip_mps", FIXED[:2]),
            ("Swing tracking: stop /start walk", "m", "swing_local_foot_rmse_m", FIXED[:2]),
            ("Raised-hands peak height error", "m", "raised_peak_error", FIXED[2:])]):
        for i, label in enumerate(labels):
            for j, name in enumerate(names):
                values = [row[key] for row in fixed if row["candidate"] == label and row["clip"] == name]
                offset = (j - (len(names)-1)/2) * .24
                ax.scatter([i + offset - .035, i + offset, i + offset + .035], values,
                           color=colors[i % len(colors)], marker=("o", "^")[j], s=24)
                ax.plot([i + offset - .09, i + offset + .09], [statistics.mean(values)] * 2,
                        color=colors[i % len(colors)], lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
    for ax in axes.flat:
        ax.set_xticks(range(len(labels)), [x.replace("activity", "act.").replace("static", "fixed") for x in labels], rotation=18)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(bottom=0)
    fig.suptitle("Q1 frozen-policy control comparison | 3 configured seeds | 0 optimizer updates", fontsize=13)
    fig.savefig(output / "control_comparison.png", dpi=150)
    fig.savefig(output / "control_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    prepare(args.trial, args.output, args.plot)
