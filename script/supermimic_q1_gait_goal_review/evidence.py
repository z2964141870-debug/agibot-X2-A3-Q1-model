"""Training-only evidence fixes; the successful native preflight stays frozen."""

import csv
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_gait_goal"))
from physical_gate import evaluate


def forward_metrics(full_reference_xy, actual_endpoint_xy, actual_origin_xy):
    direction = [float(end) - float(start)
                 for start, end in zip(full_reference_xy[0], full_reference_xy[-1])]
    length = math.hypot(*direction)
    forward = sum((float(end) - float(start)) * axis / max(length, 1e-8)
                  for end, start, axis in zip(actual_endpoint_xy, actual_origin_xy, direction))
    return dict(reference_displacement_m=length, forward_displacement_m=forward,
                reference_direction_scope="Full immutable reference; actual pre-reset endpoint")


def physical_decision(baseline, candidate):
    try:
        result = evaluate(baseline, candidate)
    except ValueError as error:
        return dict(status="PHYSICAL_EVIDENCE_UNAVAILABLE_OR_INVALID", metric_error=str(error),
                    fixed_numeric_passed=False, physical_regression_stop=True,
                    per_repeat=[], per_walk=[], overall_goal_achieved=False,
                    user_accepted_checkpoint=None)
    severe = any(not row["complete"] or not row.get("progress_preserved", True)
                 for row in result["per_repeat"])
    for row in result["per_repeat"]:
        severe |= any(value["candidate"] > 1.25 * value["baseline"] + 1e-12
                      for value in row["guards"].values())
    for row in result["per_walk"]:
        severe |= any(value["candidate_median"] > 1.5 * value["baseline_median"] + 1e-12
                      for value in row["metrics"].values())
    result["physical_regression_stop"] = bool(severe)
    return result


def candidate_rank(candidate):
    decision = candidate["decision"]
    checks = decision["per_repeat"]
    admissible = len(checks) == 9 and all(row["passed"] for row in checks)
    fallback = candidate["label"] == "initial"
    ratios = [value["ratio"] for row in decision["per_walk"] for value in row["metrics"].values()]
    return [int(not (fallback or admissible)), int(not decision["fixed_numeric_passed"]),
            sum(not row["passed"] for row in checks) if checks else 999,
            max((value if value is not None else 1e6 for value in ratios), default=1e6),
            candidate["training_seconds"]]


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)


def review_tables(trial):
    selection = json.loads((trial / "run/checkpoint_selection.json").read_text())
    labels = list(dict.fromkeys(row["label"] for row in selection["candidates"]))
    per_clip, fixed, grouped = [], [], []
    for label in labels:
        for repeat in range(3):
            summary = json.loads((trial / f"evaluation/{label}_repeat{repeat}.json").read_text())
            rows = [dict(label=label, repeat=repeat, training_seconds=summary["training_seconds"], **row)
                    for row in summary["per_clip"]]
            if len(rows) != 330 or len({row["name"] for row in rows}) != 330:
                raise ValueError("Full330 unique-clip table required")
            per_clip.extend(rows)
            fixed.extend(summary["goal_fixed"])
            for split, group in sorted({(row["split"], row["group"]) for row in rows}):
                chosen = [row for row in rows if row["split"] == split and row["group"] == group]
                values = dict(label=label, repeat=repeat, split=split, group=group,
                              clips=len(chosen), failures=sum(row["failed"] for row in chosen))
                for key in ("completion", "root_xy_rmse", "upper_rmse", "leg_rmse",
                            "wrist_relative_rmse", "loaded_slip", "target_oscillation"):
                    observed = [row[key] for row in chosen if row[key] is not None]
                    values[key] = statistics.fmean(observed) if observed else None
                    values[key + "_observations"] = len(observed)
                grouped.append(values)
    output = trial / "review"
    output.mkdir(exist_ok=False)
    for name, rows in (("per_clip", per_clip), ("fixed", fixed), ("split_group", grouped)):
        write_csv(output / (name + ".csv"), rows)
    return dict(labels=labels, per_clip_rows=len(per_clip), fixed_rows=len(fixed),
                split_group_rows=len(grouped), missing_values="Blank CSV cells; never zero-filled")
