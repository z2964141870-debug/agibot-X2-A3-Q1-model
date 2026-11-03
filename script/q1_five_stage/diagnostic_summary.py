"""Aggregate saved complete-motion diagnostics without running a simulator."""

import argparse
import json
import math
from pathlib import Path
import statistics

from common import FIXED, identity, require, save


def number(value, nonnegative=True):
    return type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0)


def values(row):
    diagnosis = row.get("diagnostics", {})
    result = {}
    for key in ("torso_tilt_rms_rad", "tilt_error_rms_rad", "upper_joint_rmse_rad"):
        require(number(diagnosis.get(key)), "Missing complete-motion diagnostic: " + key)
        result[key] = diagnosis[key]
    for key in ("sole_p95_m", "reference_sole_p95_m"):
        vector = diagnosis.get(key)
        require(isinstance(vector, list) and len(vector) == 2
                and all(number(value, False) for value in vector), "Invalid " + key)
        for side, value in zip(("left", "right"), vector):
            result[key.removesuffix("_m") + "_" + side + "_m"] = value
    for side in ("left", "right"):
        result["sole_p95_shortfall_" + side + "_m"] = (
            result["reference_sole_p95_" + side + "_m"] - result["sole_p95_" + side + "_m"])
    samples, slip = diagnosis.get("loaded_samples"), diagnosis.get("loaded_slip_mean_m_s")
    require(isinstance(samples, list) and isinstance(slip, list) and len(samples) == len(slip) == 2,
            "Missing loaded foot diagnostics")
    for side, count, speed in zip(("left", "right"), samples, slip):
        require(type(count) is int and count >= 0 and ((count == 0 and speed is None)
                or (count > 0 and number(speed))), "Loaded slip must identify absent load samples")
        result["loaded_slip_" + side + "_m_s"] = speed
    for key in ("lift_events_above_10mm", "alternating_lift_transitions"):
        require(type(diagnosis.get(key)) is int and diagnosis[key] >= 0, "Invalid lift event count")
        result[key] = diagnosis[key]
    require(result["alternating_lift_transitions"] <= max(0, result["lift_events_above_10mm"] - 1),
            "Alternating events exceed lift transitions")
    actual, target = (diagnosis.get(key) for key in
                      ("upper_joint_peak_to_peak_rad", "upper_reference_peak_to_peak_rad"))
    require(isinstance(actual, list) and isinstance(target, list) and len(actual) == len(target) > 0
            and all(number(value) for value in actual + target), "Missing upper-body amplitudes")
    result["upper_peak_to_peak_max_abs_error_rad"] = max(abs(a - b) for a, b in zip(actual, target))
    return result


def quantile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_complete(rows):
    rows = list(rows)
    keys = [(row["clip"], row["seed"]) for row in rows]
    require(rows and len(set(keys)) == len(keys), "Missing or duplicate diagnostic cases")
    require(all(type(row.get("complete")) is bool for row in rows), "Missing completion flag")
    selected = [row for row in rows if row["complete"]]
    grouped = {}
    for row in selected:
        require(row.get("failure_frame") is None
                and row["valid_steps"] == row["attempted_steps"] == row["reference_steps"],
                "Truncated diagnostics cannot be complete-motion evidence")
        grouped.setdefault(row["clip"], []).append(values(row))
    result = dict(complete_cases=len(selected), excluded_failed_cases=len(rows) - len(selected),
        complete_clips=len(grouped), total_clips=len({row["clip"] for row in rows}),
        statistics={}, counts={},
        weighting="mean repeated seeds within each clip, then equal clip weights; not frame weighted",
        interval="complete-motion summaries only; early-failure diagnostics are excluded",
        load_note="endpoint load >=5N; no continuous-support claim; missing load is not zero slip",
        event_note="10mm upward threshold crossings without duration filtering; events are not verified steps",
        natural_gait_qualified=False)
    if not grouped:
        return result
    fields = next(iter(grouped.values()))[0]
    for field in fields:
        means = [statistics.mean(item[field] for item in repeats)
                 for repeats in grouped.values() if all(item[field] is not None for item in repeats)]
        ordered = sorted(means)
        result["statistics"][field] = dict(included_clips=len(means), missing_clips=len(grouped) - len(means),
            mean=statistics.mean(means) if means else None,
            median=statistics.median(means) if means else None,
            p95=quantile(ordered, .95) if means else None,
            minimum=min(means) if means else None, maximum=max(means) if means else None)
    result["counts"] = dict(
        clips_with_no_lift_events=sum(all(item["lift_events_above_10mm"] == 0 for item in repeats)
                                     for repeats in grouped.values()),
        clips_with_no_alternating_events=sum(all(item["alternating_lift_transitions"] == 0 for item in repeats)
                                            for repeats in grouped.values()),
        clips_with_both_sole_p95_at_most_5mm=sum(all(max(item["sole_p95_left_m"], item["sole_p95_right_m"]) <= .005
                                                  for item in repeats) for repeats in grouped.values()),
        clips_with_reference_both_p95_above_10mm_but_actual_both_at_most_5mm=sum(all(
            min(item["reference_sole_p95_left_m"], item["reference_sole_p95_right_m"]) > .01
            and max(item["sole_p95_left_m"], item["sole_p95_right_m"]) <= .005 for item in repeats)
            for repeats in grouped.values()))
    return result


def execute(args):
    from comparison import case_index
    require(not args.output.exists(), "Preserve the previous diagnostic summary")
    manifest = json.loads(args.manifest.read_text())
    artifacts = manifest["artifacts"]
    records = {row["relative"]: row for row in artifacts}
    require(len(records) == len(artifacts), "Duplicate delivery artifact")
    verified = []

    def read(relative):
        expected = records.get(relative)
        require(expected is not None, "Diagnostic input is absent from the original delivery")
        source = args.review_root / relative
        actual = identity(source)
        require(all(actual[key] == expected[key] for key in ("bytes", "sha256")),
                "Original diagnostic input changed: " + relative)
        verified.append(actual)
        return json.loads(source.read_text())

    stages = {}
    for stage in ("initial", "final"):
        record = read(stage + "_library/result.json")
        names = [row["clip"] for row in record["cases"]]
        require(len(names) == 393 and len(set(names)) == 393 and set(FIXED) <= set(names),
                "Expected the complete frozen393 training library")
        index = case_index(record, names, (0,))
        for row in index.values():
            case = read(f"{stage}_library/{row['clip']}_s0/result.json")
            require(case == {key: value for key, value in row.items() if key != "relative"},
                    "Per-case and library diagnostic records differ")
        stages[stage] = record
    before, after = stages["initial"], stages["final"]
    require(before["iteration"] == 0 and after["iteration"] > 0
            and before["asset_sha256"] == after["asset_sha256"], "Baseline model identity changed")
    left, right = ({row["clip"]: row for row in stages[stage]["cases"]} for stage in ("initial", "final"))
    require(set(left) == set(right) and all(left[name]["reference_sha256"] == right[name]["reference_sha256"]
            and left[name]["reference_steps"] == right[name]["reference_steps"] for name in left),
            "Initial and final references differ")
    groups = {}
    for label, names in (("all_frozen_training_clips", set(right)),
                         ("literal_clip_name_prefix_walk_forward_", {name for name in right if name.startswith("walk_forward_")}),
                         ("fixed_three", set(FIXED))):
        groups[label] = {stage: summarize_complete(row for row in stages[stage]["cases"] if row["clip"] in names)
                         for stage in stages}
        groups[label]["clip_count"] = len(names)
    result = dict(status="VERIFIED_SAVED_DIAGNOSTICS", sources=dict(delivery_manifest=identity(args.manifest),
        library_results={stage: identity(args.review_root / (stage + "_library/result.json")) for stage in stages}),
        original_delivery_inputs_verified=len(verified),
        checkpoint={stage: stages[stage]["checkpoint"] for stage in stages}, groups=groups,
        grouping_note="walk_forward_ is a literal case-sensitive name prefix, not a semantic gait classifier",
        comparison_note="Initial failures have no full-motion diagnostic distribution; no full-length initial/final improvement claim",
        training_set_only=True, physical_transitions=0, optimizer_updates=0,
        original_evidence_modified=False, user_accepted_checkpoint=None)
    save(args.output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for field in ("manifest", "review-root", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    execute(parser.parse_args())
