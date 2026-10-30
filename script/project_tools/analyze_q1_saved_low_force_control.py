"""Measure saved joint commands inside low-force reference windows on CPU."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_gait_goal_analysis"))
from saved_control_usage import control_values
from analyze_q1_saved_sample_runs import (
    DT, LABELS, SIDES, WALKS, boolean, identity, read_curves, runs, validate_scope)
from prepare_q1_gait_review import FIXED, write_csv


JOINTS = ((0, "hip_pitch"), (3, "knee"), (4, "ankle_pitch"))
DIRECTION_EPS = .001


def previous_angles(initial, post):
    initial, post = (np.asarray(value, dtype=np.float64) for value in (initial, post))
    if (initial.shape != (22,) or post.ndim != 2 or post.shape[1] != 22 or not len(post)
            or not np.isfinite(initial).all() or not np.isfinite(post).all()):
        raise ValueError("Expected finite initial22 and post-step22 joint angles")
    return np.concatenate((initial[None], post[:-1]))


def conditions(feet, side):
    foot, other = feet[side], feet[SIDES[1 - SIDES.index(side)]]
    if not foot or len(foot) != len(other):
        raise ValueError("Aligned nonempty bilateral samples required")
    raised = np.array([r["intent_swing"] and r["reference_sole_z_m"] >= .01 for r in foot])
    force = np.array([r["native_force_N"] for r in foot])
    opposite = np.array([r["native_force_N"] for r in other])
    height = np.array([r["sole_z_m"] for r in foot])
    low = raised & (force <= 5.) & (opposite > 5.)
    return dict(raised=raised, raised_loaded=raised & (force > 5.), raised_low=low,
        raised_low_near5=low & (height <= .005), raised_low_high10=low & (height > .01),
        raised_both_low=raised & (force <= 5.) & (opposite <= 5.))


def joint_metrics(pre, post, target, reference, residual, nominal):
    values = [np.asarray(v, dtype=np.float64) for v in (pre, post, target, reference, residual, nominal)]
    if (any(v.ndim != 1 or v.shape != values[0].shape for v in values)
            or not all(np.isfinite(v).all() for v in values)):
        raise ValueError("Expected aligned finite one-joint samples")
    names = ("target_reference_rmse_rad", "post_reference_rmse_rad", "target_post_rmse_rad",
        "request_from_pre_rms_rad", "observed_step_rms_rad", "residual_rms_rad",
        "request_toward_reference_fraction", "observed_step_toward_reference_fraction",
        "observed_reference_squared_error_reduction_rad2", "residual_reference_squared_error_reduction_rad2")
    if not len(values[0]):
        return dict(samples=0, active_reference_gap_samples=0, **{name: None for name in names})
    pre, post, target, reference, residual, nominal = values
    rms = lambda v: float(np.sqrt(np.mean(v * v)))
    desired, request, observed = reference - pre, target - pre, post - pre
    active = np.abs(desired) > DIRECTION_EPS
    return dict(samples=len(pre), active_reference_gap_samples=int(active.sum()),
        target_reference_rmse_rad=rms(target - reference), post_reference_rmse_rad=rms(post - reference),
        target_post_rmse_rad=rms(target - post), request_from_pre_rms_rad=rms(request),
        observed_step_rms_rad=rms(observed), residual_rms_rad=rms(residual),
        request_toward_reference_fraction=float((request[active] * desired[active] > 0).mean()) if active.any() else None,
        observed_step_toward_reference_fraction=float((observed[active] * desired[active] > 0).mean()) if active.any() else None,
        observed_reference_squared_error_reduction_rad2=float(np.mean(desired**2 - (reference - post)**2)),
        residual_reference_squared_error_reduction_rad2=float(np.mean((reference - nominal)**2 - (reference - target)**2)))


def read_pose(path, feet):
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            key = raw["line"], raw["label"], int(raw["repeat"]), raw["clip"]
            side = raw["side"]
            if side not in SIDES:
                raise ValueError("Unknown saved foot")
            row = {name: float(raw[name]) for name in ("control_start_s", "reference_sample_end_s",
                "training_seconds", "reference_sole_z_m", "native_force_final_solve_N", "pre_sole_z_m",
                "post_sole_z_m", "target_sole_z_m", "nominal_sole_z_m", "reference_same_root_sole_z_m")}
            if not all(np.isfinite(value) for value in row.values()):
                raise ValueError("Nonfinite saved static pose")
            row.update(sample=int(raw["sample"]), global_reference_frame=int(raw["global_reference_frame"]),
                intent_swing=boolean(raw["intent_swing"]), target_leg_any_joint_outside=boolean(raw["target_leg_any_joint_outside"]))
            groups[key, side].append(row)
    if set(groups) != {(key, side) for key in feet for side in SIDES}:
        raise ValueError("Saved pose and force clip scopes differ")
    for (key, side), rows in groups.items():
        foot = feet[key][side]
        if len(rows) != len(foot):
            raise ValueError("Saved pose and force lengths differ")
        for i, (row, sample) in enumerate(zip(rows, foot)):
            if (row["sample"] != i or not np.isclose(row["control_start_s"], i * DT, atol=1e-9, rtol=0)
                    or not np.isclose(row["reference_sample_end_s"], (i + 1) * DT, atol=1e-9, rtol=0)
                    or row["training_seconds"] != sample["training_seconds"]
                    or row["global_reference_frame"] != sample["global_reference_frame"]
                    or row["intent_swing"] != sample["intent_swing"]
                    or row["native_force_final_solve_N"] != sample["native_force_N"]
                    or abs(row["reference_sole_z_m"] - sample["reference_sole_z_m"]) > 3e-7
                    or abs(row["post_sole_z_m"] - sample["sole_z_m"]) > 3e-7):
                raise ValueError("Saved pose and force clocks, identity or values differ")
    return groups


def published_inputs(manifest_path, directory, names):
    manifest = json.loads(manifest_path.read_text())
    records = manifest["artifacts"]
    if isinstance(records, dict):
        records = records["files"]
    if not isinstance(records, list):
        raise ValueError("Expected a published artifact list")
    result = [manifest_path]
    for name in names:
        path = directory / name
        expected = next(r for r in records if r["relative_path"].endswith("/" + name))
        actual = identity(path)
        if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError("Published saved analysis changed: " + str(path))
        result.append(path)
    return result


def plot(output, saved, feet):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line, labels in LABELS.items():
        figure = plt.figure(figsize=(13, 12), layout="constrained")
        grid = figure.add_gridspec(6, 2, height_ratios=(1, 1, 1, 1, .15, .25))
        axes = [[figure.add_subplot(grid[row, col]) for col in range(2)] for row in range(4)]
        for col, clip in enumerate(WALKS):
            first, last = (saved[line, label, 0, clip] for label in (labels[0], labels[-1]))
            foot = feet[line, labels[-1], 0, clip]
            mask = conditions(foot, "right")["raised_low_near5"]
            seconds = (np.arange(len(last["post"])) + 1) * DT
            for row, (offset, title) in enumerate(JOINTS):
                index = 6 + offset
                ax = axes[row][col]
                ax.plot(seconds, np.degrees(last["reference"][:, index]), color="#222222", lw=1, label="Reference")
                ax.plot(seconds, np.degrees(first["post"][:, index]), color="#888888", lw=.9, label="Initial post")
                ax.plot(seconds, np.degrees(last["target"][:, index]), color="#297d92", ls="--", lw=.9, label="Final position target")
                ax.plot(seconds, np.degrees(last["post"][:, index]), color="#087c58", lw=.9, label="Final post")
                ax.scatter(seconds[mask], np.degrees(last["post"][mask, index]), s=11, color="#b94443", zorder=3,
                           label="Final low-force / near-ground sample")
                ax.set_ylabel("Right " + title.replace("_", " ") + " (deg)")
            ax = axes[3][col]
            for values, color, title in ((feet[line, labels[0], 0, clip], "#888888", "Initial"), (foot, "#087c58", "Final")):
                ax.plot(seconds, [r["sole_z_m"] * 1000 for r in values["right"]], color=color, lw=.9, label=title)
            ax.plot(seconds, [r["reference_sole_z_m"] * 1000 for r in foot["right"]], color="#222222", ls=":", lw=.9)
            ax.scatter(seconds[mask], np.array([r["sole_z_m"] for r in foot["right"]])[mask] * 1000,
                       color="#b94443", s=11, zorder=3)
            ax.set_ylabel("Post right sole Z (mm)")
            for row in range(4):
                axes[row][col].set_xlabel("Post-state / reference sample time (s)")
                axes[row][col].grid(alpha=.18)
            axes[0][col].set_title("Stop walk / repeat0" if col == 0 else "Start walk / repeat0")
        legend = figure.add_subplot(grid[4, :])
        legend.axis("off")
        handles, names = axes[0][0].get_legend_handles_labels()
        legend.legend(handles, names, ncol=3, loc="center", fontsize=8)
        note = figure.add_subplot(grid[5, :])
        note.axis("off")
        note.text(.5, .5, "Target and pre-joints at t; last-solve force at t+15ms; plotted post/reference at t+20ms.\nMarkers: original reference raised window, right force <=5N, left >5N, post sole <=5mm.\nPosition targets are force-producing setpoints, not next poses; saved low force is not continuous unloading.",
                  transform=note.transAxes, ha="center", va="center", fontsize=9)
        figure.suptitle("Q1 " + line.upper() + ": saved commands at low-force / near-ground samples", fontsize=13)
        for ext in ("png", "pdf"):
            figure.savefig(output / f"{line}_low_force_control.{ext}", dpi=150)
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError("Will not overwrite a saved-control analysis root")
    paths = [Path(__file__), HERE / "analyze_q1_saved_sample_runs.py", HERE / "prepare_q1_gait_review.py",
             HERE.parent / "supermimic_q1_gait_goal_analysis/saved_control_usage.py",
             HERE.parent / "supermimic_q1_sole/reward_signal.py"]
    for manifest, directory, names in (
        (args.phase_manifest, args.phase, ("summary.json", "foot_timeseries.csv")),
        (args.pose_manifest, args.pose, ("summary.json", "target_pose_timeseries.csv")),
        (args.control_manifest, args.control, ("summary.json",))):
        paths.extend(published_inputs(manifest, directory, names))
    control = json.loads((args.control / "summary.json").read_text())
    if not control["historical_initial_and_stage_trace_sha_verified"] or control["new_integration_steps"]:
        raise ValueError("Expected verified saved-only control provenance")
    native = next(r for r in control["sources"] if r["path"].endswith("/supermimic/control.json"))
    native_path = Path(native["path"])
    if identity(native_path) != native:
        raise ValueError("Frozen joint/control identity differs")
    paths.append(native_path)
    before = [identity(path) for path in paths]
    names = json.loads(native_path.read_text())["joints"]
    for side_index, side in enumerate(SIDES):
        for offset, name in JOINTS:
            if names[side_index * 6 + offset] != side + "_" + name + "_joint":
                raise ValueError("Original hip/knee/ankle order differs")
    feet = read_curves(args.phase / "foot_timeseries.csv")
    validate_scope(feet)
    poses = read_pose(args.pose / "target_pose_timeseries.csv", feet)
    historical = {r["path"]: r for r in control["trace_sources"]}
    joint_rows, foot_rows, sample_rows, saved, trace_sources = [], [], [], {}, []
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        selection_path, library_path = (trial / p for p in ("run/checkpoint_selection.json", "evaluation/evaluation_library.json"))
        for path in (selection_path, library_path):
            expected = next(r for r in control["sources"] if r["path"] == str(path))
            if identity(path) != expected:
                raise ValueError("Original selection/library identity differs")
            trace_sources.append(expected)
        selection, library = (json.loads(p.read_text()) for p in (selection_path, library_path))
        if selection["user_accepted_checkpoint"] is not None or len(library["splits"]["train"]) != 330:
            raise ValueError("Original330 library and null adoption required")
        library = library["splits"]["train"]
        slots = [next(i for i, row in enumerate(library) if row["name"] == clip) for clip in FIXED]
        if tuple(r["label"] for r in selection["candidates"]) != LABELS[line]:
            raise ValueError("Original saved stage scope differs")
        for entry in selection["candidates"]:
            label = entry["label"]
            if len(entry["traces"]) != 3:
                raise ValueError("Expected all three original repeats")
            for repeat, source in enumerate(entry["traces"]):
                path = trial / f"evaluation/{label}_repeat{repeat}.npz"
                result_path = path.with_suffix(".json")
                actual, result_identity = identity(path), identity(result_path)
                if actual != source or historical.get(str(path)) != actual or historical.get(str(result_path)) != result_identity:
                    raise ValueError("Historical saved trace/result differs")
                result = json.loads(result_path.read_text())
                if result["label"] != label or result["repeat"] != repeat or len(result["per_clip"]) != 330:
                    raise ValueError("Original result stage/repeat/library differs")
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key][:, slots].copy() for key in
                              ("states", "targets", "reference_q", "residual", "frames", "force", "alive", "alpha")}
                    initial = archive["initial"][slots].copy()
                    steps, reference_steps = archive["steps"][slots], archive["reference_steps"][slots]
                for slot, clip in enumerate(FIXED):
                    key = line, label, repeat, clip
                    count = len(feet[key]["left"])
                    if (int(steps[slot]) != count or int(reference_steps[slot]) != count
                            or int(result["per_clip"][slots[slot]]["steps"]) != count
                            or not arrays["alive"][:count, slot].all() or not np.all(arrays["alpha"][:count, slot] == .5)):
                        raise ValueError("Completed original fixed prefix or static alpha.5 differs")
                    post = arrays["states"][:count, slot, 13:35].astype(np.float64)
                    pre = previous_angles(initial[slot, 13:35], post)
                    target, reference = (arrays[name][:count, slot].astype(np.float64) for name in ("targets", "reference_q"))
                    residual = arrays["residual"][:count, slot].astype(np.float64)
                    if (line == "base" or label == "initial") and np.any(residual):
                        raise ValueError("Base and initial residual must remain zero")
                    _, boundary, nominal14 = control_values(target, residual, control["native_default_q_rad"], control["action_scale_rad"])
                    if boundary.any():
                        raise ValueError("Previously interior final actions no longer identifiable")
                    nominal = target.copy()
                    nominal[:, :14] = nominal14
                    if target.shape != post.shape or reference.shape != post.shape or not np.isfinite(reference).all():
                        raise ValueError("Saved22 joint command/reference dimensions differ")
                    saved[key] = dict(pre=pre, post=post, target=target, reference=reference, residual=residual)
                    header = dict(line=line, label=label, repeat=repeat, clip=clip, training_seconds=result["training_seconds"])
                    for side_index, side in enumerate(SIDES):
                        foot = feet[key][side]
                        np.testing.assert_array_equal(arrays["frames"][:count, slot], [r["global_reference_frame"] for r in foot])
                        np.testing.assert_array_equal(arrays["force"][:count, slot, side_index], [r["native_force_N"] for r in foot])
                        if any(r["training_seconds"] != result["training_seconds"] for r in foot):
                            raise ValueError("Saved training budget differs")
                        masks = conditions(feet[key], side)
                        pose = poses[key, side]
                        for condition, mask in masks.items():
                            selected = [pose[i] for i in np.flatnonzero(mask)]
                            avg = lambda name: float(np.mean([r[name] for r in selected])) if selected else None
                            foot_rows.append(dict(**header, side=side, condition=condition, samples=len(selected),
                                pre_sole_mean_m=avg("pre_sole_z_m"), post_sole_mean_m=avg("post_sole_z_m"),
                                static_target_sole_mean_m=avg("target_sole_z_m"), static_nominal_sole_mean_m=avg("nominal_sole_z_m"),
                                same_root_reference_sole_mean_m=avg("reference_same_root_sole_z_m"),
                                target_leg_any_joint_outside_fraction=float(np.mean([r["target_leg_any_joint_outside"] for r in selected])) if selected else None))
                            for offset, joint in JOINTS:
                                index = side_index * 6 + offset
                                joint_rows.append(dict(**header, side=side, condition=condition, joint=joint,
                                    **joint_metrics(*(v[mask, index] for v in (pre, post, target, reference)),
                                                    residual[mask, index], nominal[mask, index])))
                        for i in np.flatnonzero(masks["raised_low"]):
                            for offset, joint in JOINTS:
                                index = side_index * 6 + offset
                                sample_rows.append(dict(**header, side=side, joint=joint, sample=int(i),
                                    control_start_s=i * DT, force_sample_s=i * DT + .015, post_reference_s=(i + 1) * DT,
                                    global_reference_frame=foot[i]["global_reference_frame"], native_force_N=foot[i]["native_force_N"],
                                    opposite_force_N=feet[key][SIDES[1 - side_index]][i]["native_force_N"], post_sole_z_m=foot[i]["sole_z_m"],
                                    reference_sole_z_m=foot[i]["reference_sole_z_m"], pre_rad=pre[i, index], post_rad=post[i, index],
                                    target_rad=target[i, index], nominal_rad=nominal[i, index], reference_rad=reference[i, index],
                                    residual_rad=residual[i, index]))
                if identity(path) != actual or identity(result_path) != result_identity:
                    raise ValueError("Saved trace changed during CPU reading")
                trace_sources.extend((actual, result_identity))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_only=True)), flush=True)
    if len(saved) != 45 or len(joint_rows) != 1620 or len(foot_rows) != 540:
        raise ValueError("Original fixed45 scope or diagnostic condition count differs")
    if before != [identity(path) for path in paths] or trace_sources != [identity(Path(r["path"])) for r in trace_sources]:
        raise ValueError("Saved analysis sources changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "joint_low_force_conditions.csv", joint_rows)
    write_csv(args.output / "foot_low_force_geometry.csv", foot_rows)
    write_csv(args.output / "low_force_joint_samples.csv", sample_rows)
    plot(args.output, saved, feet)
    summary = dict(status="SAVED_LOW_FORCE_CONTROL_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=before, trace_sources=trace_sources,
        saved_clips=len(saved), saved_states=sum(len(v["post"]) for v in saved.values()),
        joint_condition_rows=len(joint_rows), foot_geometry_rows=len(foot_rows), low_force_joint_sample_rows=len(sample_rows),
        direction_reference_gap_epsilon_rad=DIRECTION_EPS, default_force_threshold_N=5.,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0, new_FK_or_dynamics_calls=0,
        historical_initial_and_stage_trace_sha_verified=True, goal_complete=False, user_accepted_checkpoint=None,
        limitations=["Each repeat is conditioned separately; previously published median curves are not joined to individual forces.",
            "Target/pre at t, last-solve force at t+15ms, post/reference at t+20ms; low force is not continuous unloading.",
            "Position targets are force-producing setpoints, not feasible next poses or a desired foot path.",
            "Per-joint reference direction is angular geometry, not sole-height direction or a PPO reward correctness test.",
            "Conditional20ms joint changes do not expose actual substep torque, contact constraints, inertia or causality.",
            "Nominal target holds observed teacher output fixed; removing residual is not a new physical head-off rollout.",
            "Different stages may select different low-force masks; their conditional means do not isolate a training effect.",
            "Original budgets, physical gates, failed outcomes and pending observation plan are unchanged."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "joint_condition_rows", "low_force_joint_sample_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("phase-manifest", "phase", "pose-manifest", "pose", "control-manifest", "control", "base", "ppo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
