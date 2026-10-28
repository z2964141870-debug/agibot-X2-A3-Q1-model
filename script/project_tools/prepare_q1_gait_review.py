"""Prepare reviewer tables from saved RF14 JSON, without selecting a policy."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


FIXED = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071",
         "triumph_two_handed_001__A058")
VALIDATION = {"failures": "clips", "completion": "fraction",
              "upper_rmse": "rad", "wrist_relative_rmse": "m",
              "leg_rmse": "rad", "root_xy_rmse": "m"}
GAIT = {"steps": "20ms steps", "completion": "fraction", "failed": "bool",
        "forward_displacement_m": "m", "reference_displacement_m": "m",
        "root_displacement_m": "m", "loaded_slip_mps": "m/s",
        "foot_lift_tracking_rmse_m": "m", "swing_local_foot_rmse_m": "m",
        "torso_tilt_rms_rad": "rad", "target_delta_rms_rad": "rad/20ms",
        "sustained_single_support_alternations": "diagnostic count"}
RAISE = {"upper_rmse": "rad", "wrist_relative_rmse": "m",
         "raised_peak_error": "m"}
ARRAYS = ("foot_lift_p95_m", "reference_foot_lift_p95_m",
          "actual_wrist_peak_height_m", "reference_wrist_peak_height_m",
          "actual_wrist_peak_time_seconds", "reference_wrist_peak_time_seconds")


def identity(path):
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def span(values, scale=1.):
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return "N/A"
    low, high = min(values) * scale, max(values) * scale
    return f"{low:.4f}" if low == high else f"{low:.4f} to {high:.4f}"


def prepare(trial, labels, output):
    records, validation, fixed, common, stages = [], [], [], [], []
    for label in labels:
        repeats = []
        for repeat in range(3):
            path = trial / "evaluation" / f"{label}_repeat{repeat}.json"
            item = json.loads(path.read_text())
            clips = {row["name"]: row for row in item["per_clip"]}
            if (len(clips) != 330 or item["repeat"] != repeat
                    or item["summary"]["train"]["clips"] != 300
                    or item["summary"]["validation"]["clips"] != 30
                    or not all(name in clips and name in item["gait"] for name in FIXED)):
                raise ValueError(f"Unexpected evaluation identity: {path}")
            records.append(identity(path))
            prefix = {"label": label, "repeat": repeat,
                      "training_seconds": item["training_seconds"]}
            for metric, unit in VALIDATION.items():
                validation.append({**prefix, "metric": metric, "unit": unit,
                                   "value": item["summary"]["validation"][metric]})
            for name in FIXED:
                gait, clip = item["gait"][name], clips[name]
                if (gait["steps"] != clip["steps"] or gait["failed"] != clip["failed"]
                        or not 0 < clip["steps"] <= clip["reference_steps"]):
                    raise ValueError(f"Inconsistent fixed demonstration: {name}")
                row = {**prefix, "clip": name, "reference_steps": clip["reference_steps"],
                       **{metric: gait[metric] for metric in GAIT},
                       **{metric: clip[metric] for metric in RAISE}}
                for key in ARRAYS:
                    values = gait.get(key, clip.get(key))
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError(f"Missing bilateral metric: {name}/{key}")
                    for side, value in zip(("left", "right"), values):
                        row[key + "_" + side] = value
                fixed.append(row)
            comparison = item.get("RF11_common_prefix")
            if comparison is not None:
                for metric, measured in comparison["measurements"].items():
                    common.append({**prefix, "metric": metric, **measured,
                                   "candidate_over_baseline": comparison["ratios"][metric],
                                   "common_prefix_mean_seconds": comparison["common_prefix_mean_seconds"],
                                   "scope": "fixed raise available prefix" if metric == "raised_peak_error"
                                       else "30 validation clips matched per-clip prefix"})
            repeats.append(item)
        if len({item["training_seconds"] for item in repeats}) != 1:
            raise ValueError(f"Repeated evaluations have different budgets: {label}")
        stages.append({"label": label, "training_seconds": repeats[0]["training_seconds"],
                       "validation_failures": [item["summary"]["validation"]["failures"]
                                               for item in repeats],
                       "validation_ranges": {key: [item["summary"]["validation"][key]
                                                   for item in repeats] for key in VALIDATION}})
    selection_path = trial / "run/checkpoint_selection.json"
    selection = json.loads(selection_path.read_text())
    records.append(identity(selection_path))
    status_path = trial / "status.json"
    status = json.loads(status_path.read_text())
    records.append(identity(status_path))
    value = {"status": "REVIEW_TABLES_READY_USER_DECISION_PENDING",
             "captured_utc": datetime.now(timezone.utc).isoformat(),
             "pipeline_status_at_capture": status["status"], "trial": str(trial),
             "labels": labels, "stages": stages,
             "numerical_recommendation_only": selection["selected"]["label"],
             "user_accepted_checkpoint": None, "next_training_authorized_by_this_report": False,
             "sources": records, "generator": identity(Path(__file__)),
             "optimizer_updates": 0, "policy_inference_calls": 0, "new_physical_steps": 0,
             "limits": ["All repeat values retained; ranges are descriptive, not confidence intervals.",
                        "Raw full available trajectories may differ in duration; failed steps stay explicit.",
                        "Validation comparisons use saved common prefixes; raise peaks may be censored.",
                        "Ankle lift is not sole clearance; support changes are not validated step counts.",
                        "Contact is the last solved5ms substep, not integrated20ms impulse.",
                        "Root exact error is diagnostic; user reviews simulation before any decision."]}
    # New directories keep intermediate snapshots from being overwritten by final results.
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "validation.csv", validation)
    write_csv(output / "fixed_demos.csv", fixed)
    write_csv(output / "common_prefix.csv", common)
    with (output / "summary.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# RF14 Data For User Review", "", "No checkpoint has been accepted by the user.",
             "", f"Pipeline at capture: {status['status']}.",
             "", "Three repetitions are shown as ranges; raw values are in CSV.", "",
             "| Checkpoint | Cumulative seconds | Validation failures | Completion % | Upper rad | Wrist m | Leg rad | Root m (diagnostic) |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in stages:
        ranges = stage["validation_ranges"]
        cells = [stage["label"], f"{stage['training_seconds']:.6f}",
                 "/".join(map(str, stage["validation_failures"]))]
        cells.extend(span(ranges[key], 100. if key == "completion" else 1.)
                     for key in ("completion", "upper_rmse", "wrist_relative_rmse", "leg_rmse", "root_xy_rmse"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Fixed Demonstrations", "",
                  "| Checkpoint | Clip | Completed repeats | Forward m | Slip m/s | Ankle error m | Tilt rad | Raise drift m | Wrist peak error m |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for label in labels:
        for name in FIXED:
            rows = [row for row in fixed if row["label"] == label and row["clip"] == name]
            complete = sum(not row["failed"] and row["steps"] == row["reference_steps"] for row in rows)
            cells = [label, name, f"{complete}/3"]
            cells.extend(span([row[key] for row in rows]) for key in
                         ("forward_displacement_m", "loaded_slip_mps", "foot_lift_tracking_rmse_m", "torso_tilt_rms_rad"))
            cells.extend((span([row["root_displacement_m"] for row in rows]),
                          span([row["raised_peak_error"] for row in rows]))
                         if name == FIXED[2] else ("N/A", "N/A"))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation Limits", ""] + ["- " + item for item in value["limits"]])
    lines.extend(["", "## Simulation Videos", ""])
    for label in ("initial", "final", "selected", "stage_01", "stage_02"):
        media = trial / "evaluation" / (label + ".mp4")
        lines.append(f"- {label}: {media}" if media.exists() else f"- {label}: not generated at capture.")
    with (output / "README.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"status": value["status"], "labels": labels,
                      "validation_rows": len(validation), "fixed_demo_rows": len(fixed),
                      "common_prefix_rows": len(common), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True,
                        choices=("RF11_initial", "stage_01", "stage_02", "stage_03"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.labels)) != len(args.labels):
        parser.error("Checkpoint labels must be distinct")
    prepare(args.trial, args.labels, args.output)
