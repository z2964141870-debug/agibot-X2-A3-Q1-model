"""Join published saved force, sole and root-tilt samples without a rollout."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from analyze_q1_saved_sample_runs import (
    CLIP_LENGTHS, DT, LABELS, SIDES, WALKS, boolean, identity, read_curves, runs, validate_scope)
from prepare_q1_gait_review import write_csv


def read_root(path, feet):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            key = raw["line"], raw["label"], int(raw["repeat"]), raw["clip"]
            row = {name: float(raw[name]) for name in
                ("control_start_s", "reference_sample_end_s", "training_seconds", "local_y_tilt_delta_rad", "projection_y_m")}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError("Nonfinite published root sample")
            row.update(sample=int(raw["sample"]), global_reference_frame=int(raw["global_reference_frame"]),
                raised_reference_left=boolean(raw["raised_reference_left"]),
                raised_reference_right=boolean(raw["raised_reference_right"]))
            groups[key].append(row)
    if set(groups) != set(feet):
        raise ValueError("Published root and foot clip scopes differ")
    for key, values in groups.items():
        if len(values) != len(feet[key]["left"]):
            raise ValueError("Published root and foot durations differ")
        for index, row in enumerate(values):
            if (row["sample"] != index
                    or not math.isclose(row["control_start_s"], index * DT, abs_tol=1e-9)
                    or not math.isclose(row["reference_sample_end_s"], (index + 1) * DT, abs_tol=1e-9)):
                raise ValueError("Root sample order, origin or clock differs")
            for side in SIDES:
                foot = feet[key][side][index]
                raised = foot["intent_swing"] and foot["reference_sole_z_m"] >= .01
                if (row["global_reference_frame"] != foot["global_reference_frame"]
                        or row["training_seconds"] != foot["training_seconds"]
                        or row["raised_reference_" + side] != raised):
                    raise ValueError("Published reference, training budget or sample identity differs")
    return dict(groups)


def paired_indices(count, shift):
    if (type(count) is not int or count <= 0 or type(shift) is not int or shift not in (-1, 0, 1)):
        raise ValueError("Expected positive saved duration and a -1/0/+1 sample pairing")
    return [(index, index + shift) for index in range(count) if 0 <= index + shift < count]


def mean(values):
    return statistics.mean(values) if values else None


def analyze(key, feet, roots, side, threshold, shift):
    if side not in SIDES or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Expected a known side and positive finite force threshold")
    line, label, repeat, clip = key
    foot, other = feet[side], feet[SIDES[1 - SIDES.index(side)]]
    count = len(foot)
    if len(other) != count or len(roots) != count:
        raise ValueError("Aligned bilateral foot and root samples required")
    pairs = paired_indices(count, shift)
    window = [row["intent_swing"] and row["reference_sole_z_m"] >= .01 for row in foot]
    eligible = [(i, j) for i, j in pairs if window[i]]
    loaded, low, both_low = [], [], []
    for i, j in eligible:
        if foot[i]["native_force_N"] > threshold:
            loaded.append((i, j))
        elif other[i]["native_force_N"] > threshold:
            low.append((i, j))
        else:
            both_low.append((i, j))
    low_mask = [False] * count
    for i, _ in low:
        low_mask[i] = True
    intervals = runs(low_mask)
    header = dict(line=line, label=label, repeat=repeat, clip=clip, side=side,
        training_seconds=foot[0]["training_seconds"], force_threshold_N=threshold,
        pose_shift_samples=shift, pose_minus_force_sample_s=shift * DT + .005,
        root_minus_force_sample_s=shift * DT - .015)
    episodes = []
    for first, stop in intervals:
        indices = list(range(first, stop))
        heights = [foot[i + shift]["sole_z_m"] for i in indices]
        high_runs = runs([height > .01 for height in heights])
        longest_high = max((b - a for a, b in high_runs), default=0)
        episodes.append(dict(**header, first_force_sample=first, stop_exclusive=stop,
            force_first_s=first * DT + .015, force_last_s=(stop - 1) * DT + .015,
            samples=stop - first, endpoint_span_s=(stop - first - 1) * DT,
            at_least3_low_samples=stop - first >= 3, longest_high10_samples=longest_high,
            at_least3_high_samples=longest_high >= 3,
            sole_z_max_m=max(heights), sole_near5_fraction=sum(z <= .005 for z in heights) / len(heights),
            local_y_tilt_delta_mean_deg=mean([math.degrees(roots[i + shift]["local_y_tilt_delta_rad"]) for i in indices])))
    row = dict(**header, reference_frames=sum(window), eligible_reference_frames=len(eligible),
        loaded_frames=len(loaded), low_other_loaded_frames=len(low), both_low_frames=len(both_low),
        loaded_near5_frames=sum(foot[j]["sole_z_m"] <= .005 for _, j in loaded),
        low_other_loaded_near5_frames=sum(foot[j]["sole_z_m"] <= .005 for _, j in low),
        low_other_loaded_high10_frames=sum(foot[j]["sole_z_m"] > .01 for _, j in low),
        low_runs_ge3=sum(e["at_least3_low_samples"] for e in episodes),
        low_runs_ge3_with_high_ge3=sum(e["at_least3_low_samples"] and e["at_least3_high_samples"] for e in episodes),
        longest_low_run_samples=max((e["samples"] for e in episodes), default=0),
        loaded_fraction_window=len(loaded) / len(eligible) if eligible else None,
        low_other_loaded_fraction_window=len(low) / len(eligible) if eligible else None,
        low_other_loaded_near5_fraction=sum(foot[j]["sole_z_m"] <= .005 for _, j in low) / len(low) if low else None,
        local_y_tilt_delta_loaded_mean_deg=mean([math.degrees(roots[j]["local_y_tilt_delta_rad"]) for _, j in loaded]),
        local_y_tilt_delta_low_mean_deg=mean([math.degrees(roots[j]["local_y_tilt_delta_rad"]) for _, j in low]),
        side_projection_y_loaded_mean_m=mean([(1 if side == "right" else -1) * roots[j]["projection_y_m"] for _, j in loaded]),
        side_projection_y_low_mean_m=mean([(1 if side == "right" else -1) * roots[j]["projection_y_m"] for _, j in low]))
    if len(loaded) + len(low) + len(both_low) != len(eligible):
        raise AssertionError("Disjoint force categories must partition the eligible window")
    return row, episodes


def aggregate(rows):
    metrics = ("low_runs_ge3", "low_runs_ge3_with_high_ge3", "longest_low_run_samples",
        "low_other_loaded_frames", "low_other_loaded_near5_fraction", "loaded_fraction_window",
        "local_y_tilt_delta_loaded_mean_deg", "local_y_tilt_delta_low_mean_deg")
    groups = defaultdict(list)
    for row in rows:
        if row["clip"] in WALKS:
            groups[(row["line"], row["label"], row["clip"], row["side"],
                    row["force_threshold_N"], row["pose_shift_samples"])].append(row)
    result = []
    for key, values in groups.items():
        values.sort(key=lambda r: r["repeat"])
        if [r["repeat"] for r in values] != [0, 1, 2]:
            raise ValueError("Each saved condition must retain all three repeats")
        result.append(dict(line=key[0], label=key[1], clip=key[2], side=key[3],
            force_threshold_N=key[4], pose_shift_samples=key[5], metrics={name: dict(
                by_repeat=[r[name] for r in values], median=statistics.median(r[name] for r in values)
                if all(r[name] is not None for r in values) else None) for name in metrics}))
    return result


def plot(output, feet, roots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line, labels in LABELS.items():
        figure = plt.figure(figsize=(13, 10), layout="constrained")
        grid = figure.add_gridspec(5, 2, height_ratios=(1, 1, 1, .15, .18))
        axes = [[figure.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)]
        for col, clip in enumerate(WALKS):
            for label, color, title in ((labels[0], "#777777", "Initial"), (labels[-1], "#12816b", "Final")):
                key = line, label, 0, clip
                value = feet[key]
                time = [i * DT for i in range(len(value["right"]))]
                share = [b["native_force_N"] / (a["native_force_N"] + b["native_force_N"])
                    if a["native_force_N"] + b["native_force_N"] > 10 else math.nan
                    for a, b in zip(value["left"], value["right"])]
                axes[0][col].plot([t + .015 for t in time], share, color=color, lw=.8, label=title)
                axes[1][col].plot(time, [math.degrees(r["local_y_tilt_delta_rad"]) for r in roots[key]], color=color, lw=.8)
                axes[2][col].plot([t + DT for t in time], [r["sole_z_m"] * 1000 for r in value["right"]], color=color, lw=.8)
            reference = feet[line, labels[0], 0, clip]["right"]
            axes[2][col].plot([r["seconds"] for r in reference], [r["reference_sole_z_m"] * 1000 for r in reference],
                              color="#ba4556", ls="--", lw=.9, label="Reference sole")
            window = [r["intent_swing"] and r["reference_sole_z_m"] >= .01 for r in reference]
            for first, stop in runs(window):
                for row in range(3):
                    axes[row][col].axvspan((first + 1) * DT, stop * DT, color="#c3ab52", alpha=.16)
            axes[0][col].set_ylim(-.05, 1.05)
            axes[0][col].set_title("Stop walk / repeat0" if col == 0 else "Start walk / repeat0")
            axes[0][col].set_ylabel("Right |world-Z force| share")
            axes[1][col].set_ylabel("Pre-root Y tilt error (deg)")
            axes[2][col].set_ylabel("Post right sole Z (mm)")
            for row in range(3):
                axes[row][col].grid(alpha=.18)
                axes[row][col].set_xlabel("Saved sample clock (s)")
        legend = figure.add_subplot(grid[3, :])
        legend.axis("off")
        handles, names = axes[0][0].get_legend_handles_labels()
        h, n = axes[2][0].get_legend_handles_labels()
        legend.legend(handles + h[-1:], names + n[-1:], ncol=3, loc="center", fontsize=9)
        note = figure.add_subplot(grid[4, :])
        note.axis("off")
        note.text(.5, .5, "Pre-root at t; force last-solve sample at t+15ms; post-sole at t+20ms. Reference windows shaded.\nForce share uses samples with bilateral sum >10N; it is not weight share, a 20ms impulse, or a continuous unloading test.",
                  transform=note.transAxes, ha="center", va="center", fontsize=9)
        figure.suptitle("Q1 " + line.upper() + ": saved load samples, tilt and sole height", fontsize=13)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{line}_saved_load_tilt.{extension}", dpi=150)
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError("Will not overwrite a saved-load analysis root")
    inputs = [Path(__file__), Path(__file__).with_name("analyze_q1_saved_sample_runs.py"),
              Path(__file__).with_name("prepare_q1_gait_review.py")]
    for manifest_path, directory, names in (
        (args.phase_manifest, args.phase, ("summary.json", "foot_timeseries.csv")),
        (args.root_manifest, args.root, ("summary.json", "root_projection_timeseries.csv"))):
        inputs.append(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        files = manifest["artifacts"]["files"]
        for name in names:
            path = directory / name
            actual = identity(path)
            expected = next(row for row in files if row["relative_path"].endswith("/" + name))
            if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
                raise ValueError("Published saved input changed: " + name)
            inputs.append(path)
    before = [identity(path) for path in inputs]
    phase = json.loads((args.phase / "summary.json").read_text())
    root = json.loads((args.root / "summary.json").read_text())
    if (not phase["historical_initial_and_stage_trace_sha_verified"] or phase["new_integration_steps"]
            or root["status"] != "SAVED_ROOT_PROJECTION_READY" or root["new_integration_steps"]
            or root["policy_inference_calls"] or root["saved_clips"] != 45):
        raise ValueError("Expected completed published saved-only upstream analyses")
    feet = read_curves(args.phase / "foot_timeseries.csv")
    validate_scope(feet)
    roots = read_root(args.root / "root_projection_timeseries.csv", feet)
    rows, episodes = [], []
    for key, value in feet.items():
        for side in SIDES:
            for threshold in (2.5, 5., 10.):
                for shift in (-1, 0, 1):
                    row, spans = analyze(key, value, roots[key], side, threshold, shift)
                    rows.append(row)
                    episodes.extend(spans)
    if len(rows) != 810 or len(feet) != 45:
        raise ValueError("Expected45 original saved clips and810 diagnostic condition rows")
    if before != [identity(path) for path in inputs]:
        raise ValueError("Published inputs changed during saved-only analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "load_tilt_conditions.csv", rows)
    write_csv(args.output / "reference_low_force_runs.csv", episodes)
    plot(args.output, feet, roots)
    summary = dict(status="SAVED_LOAD_TILT_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=before, saved_clips=len(feet),
        saved_states=sum(len(v["left"]) for v in feet.values()), condition_rows=len(rows), episode_rows=len(episodes),
        force_thresholds_N=[2.5, 5., 10.], pose_shift_samples=[-1, 0, 1],
        default_force_threshold_N=5., default_pose_shift_samples=0, minimum_low_samples=3,
        near_ground_threshold_m=.005, raised_height_threshold_m=.01,
        force_sample_time_after_pre_root_s=.015, post_pose_after_force_sample_s=.005,
        three_sample_endpoint_span_s=.04, fixed_walk_conditions=aggregate(rows),
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        new_kinematics_or_dynamics_calls=0, goal_complete=False, user_accepted_checkpoint=None,
        limitations=["Force is the sum of absolute world-Z foot-ground contact components, not body-weight share or20ms impulse.",
            "Low force and opposite loaded are saved samples, not verified continuous unloading or true step counts.",
            "Same-index force precedes post-sole5ms and follows pre-root15ms; shifted pairs retain exact timing offsets.",
            "Reference mask stays anchored at the original force-sample index; shifted geometry does not redefine its window.",
            "Reference intent is metadata, not actual support truth; height thresholds can fragment reference swings.",
            "Conditioning on force and posture cannot establish causal event order or a unique training cause.",
            "Thresholds and pairing sensitivity do not modify original training recipes, physical gates, budgets or outcomes.",
            "No original NPZ reread, FK, environment, policy, or new substep observation was used."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "condition_rows", "episode_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("phase-manifest", "phase", "root-manifest", "root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
