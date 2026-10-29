"""Describe consecutive saved foot samples without policy, FK, or integration."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


DT = .02
SIDES = ("left", "right")
WALKS = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071")
CLIP_LENGTHS = dict(zip((*WALKS, "triumph_two_handed_001__A058"), (320, 210, 271)))
LABELS = dict(base=("initial", "stage_01"), ppo=("initial", "stage_01", "stage_02"))


def identity(path):
    path = Path(path)
    content = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def runs(mask):
    if any(type(value) is not bool for value in mask):
        raise ValueError("Expected boolean saved-sample conditions")
    result, start = [], None
    for index, enabled in enumerate([*mask, False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            result.append((start, index))
            start = None
    return result


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("Expected the original boolean CSV spelling")
    return value == "True"


def read_curves(path):
    groups = defaultdict(lambda: {side: [] for side in SIDES})
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            key = raw["line"], raw["label"], int(raw["repeat"]), raw["clip"]
            if raw["side"] not in SIDES:
                raise ValueError("Unknown foot")
            sample = {name: float(raw[name]) for name in
                ("seconds", "training_seconds", "sole_z_m", "reference_sole_z_m", "native_force_N")}
            if not all(math.isfinite(value) for value in sample.values()) or sample["native_force_N"] < 0:
                raise ValueError("Nonfinite sample or negative force magnitude")
            sample.update(global_reference_frame=int(raw["global_reference_frame"]),
                intent_swing=boolean(raw["intent_swing"]))
            groups[key][raw["side"]].append(sample)
    for feet in groups.values():
        left, right = (feet[side] for side in SIDES)
        if not left or len(left) != len(right):
            raise ValueError("Missing or unequal foot samples")
        for index, (a, b) in enumerate(zip(left, right)):
            for sample in (a, b):
                if not math.isclose(sample["seconds"], (index + 1) * DT, rel_tol=0, abs_tol=1e-8):
                    raise ValueError("Saved control time has a gap, duplicate, or wrong origin")
            if (a["global_reference_frame"] != b["global_reference_frame"]
                    or (index and a["global_reference_frame"] != left[index - 1]["global_reference_frame"] + 1)
                    or a["training_seconds"] != b["training_seconds"]
                    or a["training_seconds"] != left[0]["training_seconds"]):
                raise ValueError("Saved foot identities or reference frames differ")
    return dict(groups)


def masks(feet, side, height):
    rows, other = feet[side], feet[SIDES[1 - SIDES.index(side)]]
    high = [row["sole_z_m"] > height for row in rows]
    supported = [above and row["native_force_N"] <= 5. and opposite["native_force_N"] > 5.
                 for above, row, opposite in zip(high, rows, other)]
    reference = [row["intent_swing"] and row["reference_sole_z_m"] >= height for row in rows]
    return dict(reference_window=reference, sole_high=high, low_force_high_other_loaded=supported)


def alternating_labels(events):
    labels = [side[0].upper() for start, stop, side in sorted(events)]
    longest = current = 0
    for index, label in enumerate(labels):
        current = current + 1 if index and label != labels[index - 1] else 1
        longest = max(longest, current)
    return dict(run_label_sequence="".join(labels), longest_alternating_label_chain=longest,
        same_side_adjacent_runs=sum(a == b for a, b in zip(labels, labels[1:])))


def analyze_case(key, feet, height, minimum):
    line, label, repeat, clip = key
    header = dict(line=line, label=label, repeat=repeat, clip=clip,
        training_seconds=feet["left"][0]["training_seconds"], height_threshold_m=height,
        minimum_consecutive_samples=minimum)
    count = len(feet["left"])
    foot_rows, window_rows, run_rows, qualified_events = [], [], [], []
    for side in SIDES:
        conditions = masks(feet, side, height)
        spans = {name: [(a, b) for a, b in runs(mask) if b - a >= minimum]
                 for name, mask in conditions.items()}
        qualified = spans["low_force_high_other_loaded"]
        qualified_events.extend((a, b, side) for a, b in qualified)
        matched = 0
        for number, (start, stop) in enumerate(spans["reference_window"]):
            inside = runs(conditions["low_force_high_other_loaded"][start:stop])
            longest = max((b - a for a, b in inside), default=0)
            matched += longest >= minimum
            window_rows.append(dict(**header, side=side, window=number,
                first_sample=start, stop_exclusive=stop, samples=stop - start,
                first_sample_s=(start + 1) * DT, last_sample_s=stop * DT,
                longest_qualified_samples_inside=longest,
                matched_by_consecutive_saved_samples=longest >= minimum))
        for kind, episodes in spans.items():
            for start, stop in episodes:
                run_rows.append(dict(**header, side=side, kind=kind, first_sample=start,
                    stop_exclusive=stop, samples=stop - start, first_sample_s=(start + 1) * DT,
                    last_sample_s=stop * DT, endpoint_span_s=(stop - start - 1) * DT,
                    left_censored=start == 0, right_censored=stop == count))
        reference_count = len(spans["reference_window"])
        foot_rows.append(dict(**header, side=side, saved_samples=count,
            sole_high_runs=len(spans["sole_high"]), qualified_runs=len(qualified),
            qualified_samples=sum(b - a for a, b in qualified),
            longest_qualified_run_samples=max((b - a for a, b in qualified), default=0),
            reference_windows=reference_count, matched_reference_windows=matched,
            matched_reference_window_fraction=matched / reference_count if reference_count else None))
    case_row = dict(**header, saved_samples=count, qualified_runs=len(qualified_events),
        **alternating_labels(qualified_events))
    return foot_rows, window_rows, run_rows, case_row


def validate_scope(groups):
    expected = {(line, label, repeat, clip) for line, labels in LABELS.items()
                for label in labels for repeat in range(3) for clip in CLIP_LENGTHS}
    if set(groups) != expected:
        raise ValueError("Expected exactly the original45 fixed saved clips")
    references = {}
    for key, feet in groups.items():
        if len(feet["left"]) != CLIP_LENGTHS[key[3]]:
            raise ValueError("Fixed clip length changed")
        for side in SIDES:
            name = key[3], side
            if name not in references:
                references[name] = feet[side]
            for first, current in zip(references[name], feet[side]):
                if (first["intent_swing"] != current["intent_swing"]
                        or not math.isclose(first["reference_sole_z_m"], current["reference_sole_z_m"],
                                            rel_tol=2e-5, abs_tol=2e-6)):
                    raise ValueError("Immutable reference differs between saved conditions")


def aggregate(foot_rows, case_rows):
    selected = [row for row in foot_rows if row["height_threshold_m"] == .01
                and row["minimum_consecutive_samples"] == 3 and row["clip"] in WALKS]
    result = []
    for line, labels in LABELS.items():
        for label in labels:
            for clip in WALKS:
                row = dict(line=line, label=label, clip=clip)
                for side in SIDES:
                    values = sorted((r for r in selected if (r["line"], r["label"], r["clip"], r["side"])
                                     == (line, label, clip, side)), key=lambda r: r["repeat"])
                    for metric in ("qualified_runs", "reference_windows", "matched_reference_windows"):
                        row[side + "_" + metric + "_by_repeat"] = [value[metric] for value in values]
                        row[side + "_" + metric + "_median"] = statistics.median(value[metric] for value in values)
                values = sorted((r for r in case_rows if r["height_threshold_m"] == .01
                    and r["minimum_consecutive_samples"] == 3
                    and (r["line"], r["label"], r["clip"]) == (line, label, clip)), key=lambda r: r["repeat"])
                row["run_label_sequences_by_repeat"] = [value["run_label_sequence"] for value in values]
                result.append(row)
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    for line, labels in LABELS.items():
        figure = plt.figure(figsize=(14, 6.5), layout="constrained")
        grid = figure.add_gridspec(3, 2, height_ratios=(1., .08, .08))
        axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
        legend_ax = figure.add_subplot(grid[1, :])
        note_ax = figure.add_subplot(grid[2, :])
        legend_ax.axis("off")
        note_ax.axis("off")
        for ax, clip in zip(axes, WALKS):
            ylabels = []
            for label in labels:
                feet = groups[(line, label, 0, clip)]
                for side in SIDES:
                    y = len(ylabels)
                    ylabels.append(label + " " + side)
                    conditions = masks(feet, side, .01)
                    for kind, color, filled, barheight in (
                        ("reference_window", "#aaaaaa", True, .72),
                        ("sole_high", "#207f8c", False, .55),
                        ("low_force_high_other_loaded", "#a83d51", True, .38)):
                        for start, stop in runs(conditions[kind]):
                            if stop - start >= 3:
                                ax.broken_barh([((start + .5) * DT, (stop - start) * DT)],
                                    (y - barheight / 2, barheight),
                                    facecolors=color if filled else "none", edgecolors=color, linewidth=.8)
            ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
            ax.set_ylim(len(ylabels) - .5, -.5)
            ax.set_xlim(0, CLIP_LENGTHS[clip] * DT + .02)
            ax.set_title("Stop walk, repeat 0" if clip == WALKS[0] else "Start walk, repeat 0", fontsize=11)
            ax.set_xlabel("Saved control-sample time (s)")
            ax.grid(axis="x", alpha=.2)
        figure.suptitle("Q1 " + line.upper() + ": consecutive saved samples, 10 mm / minimum 3 samples", fontsize=13)
        legend_ax.legend(handles=[Patch(facecolor="#aaaaaa", label="Reference window"),
            Patch(facecolor="none", edgecolor="#207f8c", label="Sole >10 mm"),
            Patch(facecolor="#a83d51", label="+ foot <=5 N, other >5 N")],
            loc="center", ncol=3, fontsize=9)
        note_ax.text(.5, .5, "Three samples span 40 ms. Bars show sample bins, not continuous contact. Force precedes pose by 5 ms.\nRuns and threshold windows are not stride counts; all repeats and 5 mm sensitivity are in CSV.",
            transform=note_ax.transAxes, ha="center", va="center", fontsize=9)
        figure.savefig(output / (line + "_saved_sample_runs.png"), dpi=150)
        figure.savefig(output / (line + "_saved_sample_runs.pdf"))
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError("Saved-sample analysis will not overwrite an existing root")
    manifest = json.loads(args.source_manifest.read_text())
    if manifest["stage"] != "GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030":
        raise ValueError("Expected the already published phase analysis")
    inputs = [args.source_manifest, Path(__file__)]
    expected = {Path(item["relative_path"]).name: item for item in manifest["artifacts"]["files"]}
    for name in ("foot_timeseries.csv", "summary.json"):
        path = args.input / name
        actual = identity(path)
        if any(actual[key] != expected[name][key] for key in ("bytes", "sha256")):
            raise ValueError("Published upstream evidence changed: " + name)
        inputs.append(path)
    source_ids = [identity(path) for path in inputs]
    upstream = json.loads((args.input / "summary.json").read_text())
    if not upstream["historical_initial_and_stage_trace_sha_verified"] or upstream["new_integration_steps"] != 0:
        raise ValueError("Missing saved-only upstream provenance")
    groups = read_curves(args.input / "foot_timeseries.csv")
    validate_scope(groups)
    foot_rows, windows, sample_runs, cases = [], [], [], []
    for key, feet in groups.items():
        for height in (.005, .01):
            for minimum in (1, 3):
                foot, window, episode, case = analyze_case(key, feet, height, minimum)
                foot_rows.extend(foot)
                windows.extend(window)
                sample_runs.extend(episode)
                cases.append(case)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("foot_summary", foot_rows), ("reference_windows", windows),
                       ("sample_runs", sample_runs), ("case_summary", cases)):
        write_csv(args.output / (name + ".csv"), rows)
    plot(args.output, groups)
    if source_ids != [identity(path) for path in inputs]:
        raise ValueError("Saved analysis source changed")
    summary = dict(status="SAVED_SAMPLE_RUNS_ANALYSIS_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=source_ids, saved_clips=len(groups),
        saved_foot_samples=sum(len(rows) for feet in groups.values() for rows in feet.values()),
        foot_summary_rows=len(foot_rows), reference_window_rows=len(windows), sample_run_rows=len(sample_runs),
        case_summary_rows=len(cases), default_height_m=.01, default_minimum_consecutive_samples=3,
        sensitivity_height_m=[.005, .01], sensitivity_minimum_samples=[1, 3], dt_s=DT,
        three_sample_endpoint_span_s=.04, force_sample_precedes_pose_s=.005,
        fixed_walks_default=aggregate(foot_rows, cases), new_integration_steps=0, policy_inference_calls=0,
        optimizer_updates=0, new_kinematics_or_dynamics_calls=0, goal_complete=False, user_accepted_checkpoint=None,
        limitations=["These are saved sample runs and threshold windows, not counted physical strides.",
            "Three20ms samples span40ms between endpoints, not60ms verified continuous time.",
            "Force is the last5ms solve sample, preceding the saved pose by5ms; no impulse or exact simultaneous contact claim.",
            "Reference intent is declared metadata; height thresholds can fragment a single swing.",
            "Label alternation does not constrain cadence, gap length, posture, forward progress, or stability.",
            "Threshold sensitivity is descriptive only; original physical gates, budgets and outcomes are unchanged.",
            "Upstream published CSV and summary SHA are checked; original large NPZ audits are not repeated.",
            "Timeline plots show repeat0; all repeats, stages and raised-arm clips remain in CSV."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_foot_samples", "foot_summary_rows",
        "reference_window_rows", "sample_run_rows", "case_summary_rows", "new_integration_steps", "policy_inference_calls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-manifest", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
