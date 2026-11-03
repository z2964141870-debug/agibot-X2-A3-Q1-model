"""Complete Q1 comparisons with equal clip weights and explicit missing exposure."""

import argparse
import math
from pathlib import Path
import sys

from contracts import validate_frozen
from delivery import read
from common import FIXED, TARGET_TRANSITIONS, matched_checkpoint, require, save
from diagnostic_summary import summarize_complete

METRICS = ("joint_rmse_rad", "body_relative_rmse_m")


def case_index(value, names, seeds, interface=None):
    require(value.get("status") == "PASSED", "Evaluation did not finish")
    rows = value.get("cases", [])
    keys = [(row.get("clip"), row.get("seed")) for row in rows]
    expected = {(name, seed) for name in names for seed in seeds}
    require(len(keys) == len(expected) and set(keys) == expected,
            "Missing, duplicate or foreign evaluation case")
    for row in rows:
        valid, attempted, length = (row.get(k) for k in ("valid_steps", "attempted_steps", "reference_steps"))
        require(all(type(n) is int and n >= 0 for n in (valid, attempted, length))
                and valid <= attempted <= length and length > 0,
                "Invalid physical episode length")
        require(type(row.get("complete")) is bool and row.get("evaluation_randomization") is False,
                "Evaluation must identify its completion and nominal reset")
        if interface is not None:
            require(row.get("interface") == interface, "Motion and human interfaces must be separate")
        if row["complete"]:
            require(row.get("failure_frame") is None and attempted == valid == length,
                    "A truncated or failed rollout cannot count as complete")
            require(all(finite_metric(row.get("full_clip_metrics", {}).get(metric)) for metric in METRICS),
                    "Missing complete-motion errors")
        else:
            require(type(row.get("failure_frame")) is int and 0 < row["failure_frame"] <= length
                    and "full_clip_metrics" not in row,
                    "Failed rollouts cannot publish complete-motion errors")
        require(row.get("reference_sha256") and row.get("trace"), "Missing physical evidence identity")
    return dict(zip(keys, rows))


def finite_metric(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def clip_equal_mean(rows, key):
    grouped = {}
    for row in rows:
        value = row.get(key)
        require(finite_metric(value), "Missing comparable " + key)
        grouped.setdefault(row["clip"], []).append(value)
    require(grouped, "No comparable clips")
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def summary(left, right, pairs, names):
    require(len(pairs) == len(left) == len(right), "Paired results are missing")
    for key, a in left.items():
        b = right[key]
        require(a["reference_sha256"] == b["reference_sha256"]
                and a["reference_steps"] == b["reference_steps"], "The physical reference changed")
    keys = [(row.get("clip"), row.get("seed")) for row in pairs]
    require(len(set(keys)) == len(keys) and set(keys) == set(left), "Paired comparison identities differ")
    missing = []
    for row in pairs:
        key = row["clip"], row["seed"]
        require(row.get("common_steps") == min(left[key]["valid_steps"], right[key]["valid_steps"]),
                "Common prefix does not match the shorter valid rollout")
        if row["common_steps"] <= 0 or any(not finite_metric(row.get(stage + "_" + metric))
                for stage in ("initial", "final") for metric in METRICS):
            missing.append(dict(clip=key[0], seed=key[1], reason="no_comparable_noninitial_prefix"))
    common = {}
    if not missing:
        for metric in METRICS:
            initial, final = (clip_equal_mean(pairs, stage + "_" + metric) for stage in ("initial", "final"))
            common[metric] = dict(left=initial, right=final, right_minus_left=final - initial,
                relative_reduction=None if initial == 0 else (initial - final) / initial)
    full = {}
    for label, index in (("left", left), ("right", right)):
        complete = [dict(clip=row["clip"], **row["full_clip_metrics"])
                    for row in index.values() if row["complete"]]
        full[label] = dict(complete_cases=len(complete), total_cases=len(index),
            complete_clips=len({row["clip"] for row in complete}), total_clips=len(names),
            metrics={metric: clip_equal_mean(complete, metric) for metric in METRICS} if complete else {},
            diagnostics=summarize_complete(index.values()),
            aggregation="equal weight per clip among complete cases; failed cases are counted separately")
    return dict(status="COMPARABLE" if not missing else "MISSING_COMPARABLE_PREFIX",
        common_prefix_metrics=common, missing_common_prefix=missing, complete_motion=full,
        pairing="same physical clip and seed; exclude initialization and failed terminal frame",
        weighting="equal clip weight; equal mean of repeated seeds within each clip",
        natural_gait_qualified=False, user_accepted_checkpoint=None)


def exposure_comparison(rows):
    require(rows and all(type(row.get("global_env_steps")) is int and row["global_env_steps"] > 0
                        for row in rows), "Use audited positive checkpoint exposure records")
    if max(row["global_env_steps"] for row in rows) < .99 * TARGET_TRANSITIONS:
        return dict(status="UNAVAILABLE_EARLY_STOP_BEFORE_TARGET", target_q1_transitions=TARGET_TRANSITIONS,
            largest_primebot_checkpoint_transitions=max(row["global_env_steps"] for row in rows),
            additional_training_authorized=False, comparison_claim="No equal-exposure result is available")
    try:
        selected = matched_checkpoint(rows)
    except RuntimeError:
        return dict(status="UNAVAILABLE_NO_CHECKPOINT_WITHIN_ONE_PERCENT", target_q1_transitions=TARGET_TRANSITIONS,
                    additional_training_authorized=False)
    return dict(status="MATCHED_AUDITED_CHECKPOINT", target_q1_transitions=TARGET_TRANSITIONS, checkpoint=selected)


def execute(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1_review"))
    from metrics import compare
    require(not args.output.exists(), "Preserve previous comparison evidence")
    frozen = read(args.frozen)
    names = validate_frozen(frozen)
    names, seeds = (names, (0,)) if args.library else (FIXED, (0, 1, 2))
    left = case_index(read(args.left), names, seeds, args.left_interface)
    right = case_index(read(args.right), names, seeds, args.right_interface)
    pairs = [compare(left[key], right[key]) for key in left]
    result = summary(left, right, pairs, names)
    from common import identity
    result.update(left_label=args.left_label, right_label=args.right_label,
        sources=[identity(path) for path in (args.frozen, args.left, args.right)], pairs=pairs,
        comparison_scope="Q1 in the common Primebot CPU physical plant; not native G1 evaluation",
        training_set_only=True, unseen_morphology_zero_shot=False, optimizer_updates=0,
        physical_transitions=0)
    save(args.output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("left", "right", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("left-label", "right-label"):
        parser.add_argument("--" + name, required=True)
    for name in ("left-interface", "right-interface"):
        parser.add_argument("--" + name, choices=("motion", "human"))
    parser.add_argument("--library", action="store_true")
    execute(parser.parse_args())
