"""Fixed-clip physical gate for GAIT_GOAL01; never settles the overall goal."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

WALKS = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071")
RAISE = "triumph_two_handed_001__A058"
FIXED = (*WALKS, RAISE)
REPEATS = (0, 1, 2)
LIMITS = {"swing_sole_z_rmse_m": .80, "loaded_slip_mps": .85}
GUARDS = ("torso_tilt_rms_rad", "upper_rmse", "wrist_relative_rmse")


def number(value, name, *, nonnegative=True):
    if isinstance(value, bool) or value is None:
        raise ValueError("Missing or boolean numeric field: " + name)
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid numeric field: " + name) from error
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError("Nonfinite or negative field: " + name)
    return result


def integer(value, name):
    result = number(value, name)
    if not result.is_integer():
        raise ValueError("Expected integer field: " + name)
    return int(result)


def boolean(value):
    if type(value) is bool:
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1"):
        return value.strip().lower() in ("true", "1")
    raise ValueError("failed must be a boolean or an explicit CSV boolean")


def validate(rows):
    lookup = {}
    for raw in rows:
        clip = raw.get("clip")
        repeat = integer(raw.get("repeat"), "repeat")
        key = (clip, repeat)
        if clip not in FIXED or repeat not in REPEATS or key in lookup:
            raise ValueError("Unexpected or duplicate fixed clip/repeat: " + str(key))
        row = {"clip": clip, "repeat": repeat, "failed": boolean(raw.get("failed"))}
        for name in ("steps", "reference_steps"):
            row[name] = integer(raw.get(name), name)
        if not 0 < row["steps"] <= row["reference_steps"]:
            raise ValueError("Invalid pre-reset physical duration")
        for name in GUARDS:
            row[name] = number(raw.get(name), name)
        if clip in WALKS:
            for name in LIMITS:
                row[name] = number(raw.get(name), name)
            row["swing_sole_foot_frames"] = integer(raw.get("swing_sole_foot_frames"), "swing_sole_foot_frames")
            phase_sha = raw.get("swing_sole_phase_sha256")
            if not isinstance(phase_sha, str) or not re.fullmatch("[0-9a-f]{64}", phase_sha):
                raise ValueError("Missing exact declared-swing mask SHA256")
            row["swing_sole_phase_sha256"] = phase_sha
            row["forward_displacement_m"] = number(raw.get("forward_displacement_m"), "forward_displacement_m", nonnegative=False)
            row["reference_displacement_m"] = number(raw.get("reference_displacement_m"), "reference_displacement_m")
            row["sustained_single_support_alternations"] = integer(
                raw.get("sustained_single_support_alternations"), "sustained_single_support_alternations")
            if row["swing_sole_foot_frames"] < 1 or row["reference_displacement_m"] <= 0:
                raise ValueError("Walking requires nonempty declared swing and forward reference")
        else:
            row["raised_peak_error"] = number(raw.get("raised_peak_error"), "raised_peak_error")
        lookup[key] = row
    if set(lookup) != {(clip, repeat) for clip in FIXED for repeat in REPEATS}:
        raise ValueError("Exactly three fixed clips times three repeats are required")
    return lookup


def ratio(current, baseline):
    return current / baseline if baseline > 0 else None


def guarded(current, baseline, limit=1.10):
    return current <= baseline * limit + 1e-12


def evaluate(baseline_rows, candidate_rows):
    baseline, candidate = validate(baseline_rows), validate(candidate_rows)
    checks, comparisons = [], []
    for key in baseline:
        old, new = baseline[key], candidate[key]
        if old["reference_steps"] != new["reference_steps"]:
            raise ValueError("Baseline/candidate reference duration differs")
        complete = all(not row["failed"] and row["steps"] == row["reference_steps"]
                       for row in (old, new))
        guard_values = {name: dict(baseline=old[name], candidate=new[name],
                                  ratio=ratio(new[name], old[name]),
                                  passed=guarded(new[name], old[name])) for name in GUARDS}
        if key[0] == RAISE:
            name = "raised_peak_error"
            guard_values[name] = dict(baseline=old[name], candidate=new[name],
                                     ratio=ratio(new[name], old[name]), passed=guarded(new[name], old[name]))
        check = dict(clip=key[0], repeat=key[1], complete=complete, guards=guard_values)
        if key[0] in WALKS:
            if any(old[name] != new[name] for name in ("swing_sole_foot_frames", "swing_sole_phase_sha256")):
                raise ValueError("Baseline/candidate declared swing masks differ")
            if not math.isclose(old["reference_displacement_m"], new["reference_displacement_m"], rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError("Baseline/candidate reference displacement differs")
            check["progress_preserved"] = (old["forward_displacement_m"] > .05 and
                new["forward_displacement_m"] >= max(.05, .90 * old["forward_displacement_m"]))
            check["sustained_alternations_present"] = new["sustained_single_support_alternations"] >= 2
        check["passed"] = (complete and all(value["passed"] for value in guard_values.values())
                           and check.get("progress_preserved", True)
                           and check.get("sustained_alternations_present", True))
        checks.append(check)
    for clip in WALKS:
        values = {}
        for name, limit in LIMITS.items():
            old = statistics.median(baseline[clip, repeat][name] for repeat in REPEATS)
            new = statistics.median(candidate[clip, repeat][name] for repeat in REPEATS)
            value = ratio(new, old)
            values[name] = dict(baseline_median=old, candidate_median=new, ratio=value,
                                limit=limit, passed=value is not None and value <= limit + 1e-12)
        comparisons.append(dict(clip=clip, metrics=values,
                                passed=all(value["passed"] for value in values.values())))
    passed = all(row["passed"] for row in checks + comparisons)
    return dict(status="FIXED_NUMERIC_GATE_PASSED" if passed else "FIXED_NUMERIC_GATE_NOT_PASSED",
                fixed_numeric_passed=passed, per_repeat=checks, per_walk=comparisons,
                full330_evaluation_reviewed=False, original_speed_video_reviewed=False,
                actual_clearance_and_switching_reviewed=False,
                overall_goal_achieved=False, user_accepted_checkpoint=None,
                protocol=dict(repeats=list(REPEATS), relative_error_guard=1.10,
                              minimum_forward_m=.05, minimum_baseline_forward_fraction=.90,
                              minimum_sustained_alternations=2,
                              sole_metric="RMSE over both feet at exact declared swing frames; neutral excluded",
                              zero_baseline="Cannot claim a relative gain from a zero baseline",
                              force_scope="Native final 5ms substep >5N; clearance/load transfer still require trajectory review"))


def selected_csv(path, label):
    with path.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("label") == label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()
    result = evaluate(selected_csv(args.baseline, args.baseline_label),
                      selected_csv(args.candidate, args.candidate_label))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"])
