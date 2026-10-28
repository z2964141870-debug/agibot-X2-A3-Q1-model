"""Build RF16 review tables from saved evidence, without simulation or optimization."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_q1_gait_review import (
    ARRAYS, FIXED, GAIT, RAISE, VALIDATION, identity, span, write_csv,
)


def prepare(trial, output):
    if output.exists():
        raise FileExistsError(output)
    stage_path = trial / "run/stage_decisions.json"
    decisions = json.loads(stage_path.read_text()) if stage_path.exists() else []
    labels = ["selected_base", "initial_static075"]
    labels.extend(f"stage_{index:02d}" for index in range(1, len(decisions) + 1))
    sources, split_rows, fixed_rows, common_rows, walk_rows, clip_rows, stages = (
        [] for _ in range(7)
    )
    for label in labels:
        repeats = []
        for repeat in range(3):
            path = trial / "evaluation" / f"{label}_repeat{repeat}.json"
            item = json.loads(path.read_text())
            clips = {row["name"]: row for row in item["per_clip"]}
            if (len(clips) != 330 or len(item["per_clip"]) != 330
                    or item["repeat"] != repeat
                    or item["summary"]["train"]["clips"] != 300
                    or item["summary"]["validation"]["clips"] != 30
                    or not all(name in clips and name in item["gait"] for name in FIXED)):
                raise ValueError(f"Unexpected fixed330 evaluation: {path}")
            sources.append(identity(path))
            prefix = {"label": label, "repeat": repeat,
                      "training_seconds": item["training_seconds"]}
            for split in ("train", "validation"):
                for metric, unit in VALIDATION.items():
                    split_rows.append({**prefix, "split": split, "metric": metric,
                                       "unit": unit, "value": item["summary"][split][metric]})
            for clip in item["per_clip"]:
                if not 0 < clip["steps"] <= clip["reference_steps"]:
                    raise ValueError(f"Invalid saved duration: {label}/{clip['name']}")
                clip_rows.append({**prefix, **{
                    key: json.dumps(value, separators=(",", ":"), allow_nan=False)
                    if isinstance(value, (list, dict)) else value
                    for key, value in clip.items()
                }})
            for name in FIXED:
                gait, clip = item["gait"][name], clips[name]
                if gait["steps"] != clip["steps"] or gait["failed"] != clip["failed"]:
                    raise ValueError(f"Inconsistent fixed demonstration: {label}/{name}")
                row = {**prefix, "clip": name, "reference_steps": clip["reference_steps"],
                       **{key: gait[key] for key in GAIT},
                       **{key: clip[key] for key in RAISE}}
                for key in ARRAYS:
                    values = gait.get(key, clip.get(key))
                    if not isinstance(values, list) or len(values) != 2:
                        raise ValueError(f"Missing bilateral measurement: {name}/{key}")
                    row.update({key + "_" + side: value
                                for side, value in zip(("left", "right"), values)})
                fixed_rows.append(row)
            for reference in ("selected_base", "initial_static075"):
                comparison = item.get(reference)
                if comparison is None:
                    continue
                tracking = comparison["validation_tracking_diagnostic"]
                for metric, measured in tracking["measurements"].items():
                    common_rows.append({**prefix, "baseline": reference, "metric": metric,
                                        **measured,
                                        "candidate_over_baseline": tracking["ratios"][metric],
                                        "common_prefix_mean_seconds": tracking["common_prefix_mean_seconds"]})
                for index, ratios in enumerate(comparison["walking_ratios"]):
                    current = comparison["walking_common_prefix"][index]
                    previous = comparison["baseline_walking_common_prefix"][index]
                    for metric, key in (("walking_slip", "loaded_slip_mps"),
                                        ("swing_error", "swing_local_foot_rmse_m"),
                                        ("walking_tilt", "torso_tilt_rms_rad"),
                                        ("target_change", "target_delta_rms_rad")):
                        walk_rows.append({**prefix, "baseline": reference, "clip": FIXED[index],
                                          "metric": metric, "candidate": current[key],
                                          "baseline_value": previous[key],
                                          "candidate_over_baseline": ratios[metric]})
            repeats.append(item)
        if len({item["training_seconds"] for item in repeats}) != 1:
            raise ValueError(f"Inconsistent repeated budget: {label}")
        stages.append({"label": label, "training_seconds": repeats[0]["training_seconds"],
                       "training_failures": [item["summary"]["train"]["failures"] for item in repeats],
                       "validation_failures": [item["summary"]["validation"]["failures"] for item in repeats],
                       "validation_ranges": {key: [item["summary"]["validation"][key]
                                                   for item in repeats] for key in VALIDATION}})
    status_path, plan_path = trial / "status.json", trial / "plan.json"
    plan, status = json.loads(plan_path.read_text()), json.loads(status_path.read_text())
    recommendation_path = trial / "run/checkpoint_recommendation.json"
    recommendation = json.loads(recommendation_path.read_text()) if recommendation_path.exists() else None
    sources.extend(identity(path) for path in (status_path, plan_path))
    sources.extend(identity(path) for path in (stage_path, recommendation_path) if path.exists())
    value = {
        "status": "RF16_REVIEW_READY_USER_DECISION_PENDING",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "trial": str(trial),
        "pipeline_at_capture": status["status"], "training_code_commit": plan["code_commit"],
        "upper_reference_alpha": plan["upper_reference_alpha"],
        "labels": labels, "stages": stages, "stage_decisions": decisions,
        "numerical_recommendation": recommendation, "user_accepted_checkpoint": None,
        "next_training_authorized_by_this_report": False,
        "sources": sources, "generator": identity(Path(__file__)),
        "new_physical_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
        "limits": [
            "Static arm composition is separate from improvement learned by the lower residual head.",
            "Legacy static075/trained075 labels use the actual alpha recorded in this RF16 plan.",
            "Reference contact is retarget intent, not native loaded contact truth.",
            "Phase features and phase-conditioned rewards changed together; their effects are not isolated.",
            "Repetitions share one checkpoint and are not independent training seeds.",
            "Validation was used during candidate search and is not an independent final test.",
            "Available trajectory lengths can differ; ratios use matched prefixes.",
            "Ankle lift is not sole clearance; support alternations are diagnostic counts.",
            "Native contact is the last solved5ms substep, not integrated20ms impulse.",
            "Completion does not certify natural walking; fixed simulation videos need user review.",
            "RootXY is diagnostic; checkpoint adoption remains the user's decision.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("per_clip.csv", clip_rows), ("split_metrics.csv", split_rows),
                           ("fixed_demos.csv", fixed_rows), ("tracking_common_prefix.csv", common_rows),
                           ("walking_common_prefix.csv", walk_rows)):
        write_csv(output / filename, rows)
    with (output / "summary.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# RF16 Data For User Review", "", "No checkpoint has been accepted by the user.", "",
             f"Pipeline at capture: {status['status']}; actual upper blend: {plan['upper_reference_alpha']}.", "",
             "Base is frozen HP; initial adds static arms; later stages learn a lower residual with source contact phases.", "",
             "| Candidate | Actual seconds | Train failures/300 | Val failures/30 | Completion % | Upper rad | Wrist m | Leg rad | Root m (diagnostic) |",
             "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in stages:
        ranges = stage["validation_ranges"]
        cells = [stage["label"], f"{stage['training_seconds']:.6f}",
                 "/".join(map(str, stage["training_failures"])),
                 "/".join(map(str, stage["validation_failures"]))]
        cells.extend(span(ranges[key], 100. if key == "completion" else 1.) for key in
                     ("completion", "upper_rmse", "wrist_relative_rmse", "leg_rmse", "root_xy_rmse"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Fixed Demonstrations", "",
                  "| Candidate | Clip | Complete | Forward m | Slip m/s | Swing error m | Tilt rad | Raise drift m | Wrist peak error m |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for label in labels:
        for name in FIXED:
            rows = [row for row in fixed_rows if row["label"] == label and row["clip"] == name]
            complete = sum(not row["failed"] and row["steps"] == row["reference_steps"] for row in rows)
            cells = [label, name, f"{complete}/3"]
            cells.extend(span([row[key] for row in rows]) for key in
                         ("forward_displacement_m", "loaded_slip_mps", "swing_local_foot_rmse_m", "torso_tilt_rms_rad"))
            cells.extend((span([row["root_displacement_m"] for row in rows]),
                          span([row["raised_peak_error"] for row in rows])) if name == FIXED[2] else ("N/A", "N/A"))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation Limits", ""] + ["- " + item for item in value["limits"]])
    lines.extend(["", "## Simulation Videos", ""])
    for label in ("base", "initial", "final", "recommended"):
        path = trial / "evaluation" / (label + ".mp4")
        lines.append(f"- {label}: {path}" if path.exists() else f"- {label}: pending at capture.")
    with (output / "README.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"status": value["status"], "labels": labels,
                      "per_clip_rows": len(clip_rows), "split_rows": len(split_rows),
                      "fixed_rows": len(fixed_rows), "tracking_common_rows": len(common_rows),
                      "walk_common_rows": len(walk_rows), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.trial, args.output)
