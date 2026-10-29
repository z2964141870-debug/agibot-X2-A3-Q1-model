"""Inspect PD target posture geometry on saved states, without a rollout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal_analysis"),
               str(HERE.parent / "supermimic_q1_sole")]
from prepare_q1_gait_review import FIXED, identity, write_csv
from reward_signal import saved_clip
from saved_control_usage import control_values, phase_rows
from saved_pd_demand import preceding_states


def substitute_joints(states, joints):
    states, joints = np.asarray(states), np.asarray(joints)
    if (states.ndim != 2 or states.shape[1] != 57 or joints.shape != (len(states), 22)
            or not len(states) or not np.isfinite(states).all() or not np.isfinite(joints).all()):
        raise ValueError("Expected nonempty finite57 states and aligned22 joint coordinates")
    result = states.copy()
    result[:, 13:35] = joints
    return result


def relative_other(height):
    height = np.asarray(height, dtype=np.float64)
    if height.ndim != 2 or height.shape[1] != 2 or not np.isfinite(height).all():
        raise ValueError("Expected finite left/right sole heights")
    return height - height[:, ::-1]


def joint_outside(joints, low, high):
    joints, low, high = [np.asarray(value) for value in (joints, low, high)]
    if (joints.ndim != 2 or joints.shape[1] != 22 or low.shape != (22,) or high.shape != low.shape
            or not all(np.isfinite(value).all() for value in (joints, low, high)) or (low >= high).any()):
        raise ValueError("Expected finite22 joint coordinates and ordered model ranges")
    return (joints < low - 1e-6) | (joints > high + 1e-6)


def describe(values, mask):
    values, mask = np.asarray(values), np.asarray(mask)
    if (values.ndim != 1 or mask.dtype != np.bool_ or mask.shape != values.shape
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite scalar samples and one boolean mask per sample")
    selected = values[mask]
    if not len(selected):
        return dict(mean=None, rms=None, abs_p95=None, abs_max=None)
    return dict(mean=float(selected.mean()), rms=float(np.sqrt(np.mean(selected ** 2))),
                abs_p95=float(np.quantile(np.abs(selected), .95)), abs_max=float(np.abs(selected).max()))


def plot(output, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line in ("base", "ppo"):
        labels = list(dict.fromkeys(key[1] for key in saved if key[0] == line))
        final = labels[-1]
        figure = plt.figure(figsize=(13, 10), layout="constrained")
        grid = figure.add_gridspec(5, 2, height_ratios=(1, 1, .65, .13, .13))
        axes = [[figure.add_subplot(grid[row, col]) for col in range(2)] for row in range(3)]
        for col, clip in enumerate(FIXED[:2]):
            repeats = [saved[line, final, repeat, clip] for repeat in range(3)]
            times = np.arange(len(repeats[0]["pre"])) * .02
            for row, use_relative in enumerate((False, True)):
                ax = axes[row][col]
                for kind, color, style, name in (
                    ("pre", "#222222", "-", "Observed pre-state"),
                    ("target", "#087f8c", "-", "Target joint substitution"),
                    ("nominal", "#888888", ":", "Target minus saved residual"),
                    ("reference_same_root", "#b34d4d", "--", "Reference joints, same root")):
                    values = np.median(np.stack([(relative_other(value[kind]) if use_relative
                                                  else value[kind])[:, 1] for value in repeats]), axis=0)
                    ax.plot(times, values * 1000, color=color, ls=style, lw=1, label=name)
                ax.axhline(0, color="#555555", lw=.6)
                ax.set_ylabel("Right minus other sole (mm)" if use_relative else "Right static sole Z (mm)")
            ax = axes[2][col]
            for value in repeats:
                delta = relative_other(value["target"])[:, 1] - relative_other(value["nominal"])[:, 1]
                ax.plot(times, delta * 1000, color="#087f8c", alpha=.22, lw=.6)
            median = np.median(np.stack([relative_other(value["target"])[:, 1]
                                       - relative_other(value["nominal"])[:, 1] for value in repeats]), axis=0)
            ax.plot(times, median * 1000, color="#087f8c", lw=1.2)
            ax.axhline(0, color="#555555", lw=.6)
            ax.set_ylabel("Residual right-other delta (mm)")
            ax.set_xlabel("Saved control-start time (s)")
            axes[0][col].set_title("Stop walk" if col == 0 else "Start walk")
            for row in range(3):
                axes[row][col].grid(alpha=.18)
        legend = figure.add_subplot(grid[3, :])
        note = figure.add_subplot(grid[4, :])
        legend.axis("off")
        note.axis("off")
        legend.legend(*axes[0][0].get_legend_handles_labels(), loc="center", ncol=2, fontsize=9)
        note.text(.5, .5, "Static FK with saved pre-state root held fixed; PD targets are not predicted next poses.\nTop panels: framewise medians of 3 repeats. Bottom: each repeat and median. Joint limits are in CSV.",
                  transform=note.transAxes, ha="center", va="center", fontsize=9)
        figure.suptitle(f"Q1 {line.upper()} {final}: saved target geometry, no integration", fontsize=13)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{line}_saved_target_pose.{extension}", dpi=150)
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime),
                   str(HERE.parent / "supermimic_q1_gait_goal")]
    import torch
    from support_geometry import BoundedTensorSole
    from sole_geometry import SoleGeometry
    from tensor_geometry import TensorSole
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES

    torch.set_num_threads(4)
    tracked = {}
    def remember(path, expected=None):
        actual = identity(path)
        if expected is not None and actual != expected:
            raise ValueError("Saved source identity differs: " + str(path))
        if str(path) in tracked and tracked[str(path)] != actual:
            raise ValueError("Source changed during analysis: " + str(path))
        tracked[str(path)] = actual
        return actual

    for manifest_path, directory, names in (
            (args.control_manifest, args.control, ("summary.json",)),
            (args.phase_manifest, args.phase, ("summary.json", "foot_timeseries.csv"))):
        remember(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        artifacts = manifest["artifacts"]
        artifacts = artifacts["files"] if isinstance(artifacts, dict) else artifacts
        for name in names:
            expected = next(row for row in artifacts if row["relative_path"].endswith("/" + name))
            actual = remember(directory / name)
            if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
                raise ValueError("Published input differs: " + name)
    prior = json.loads((args.control / "summary.json").read_text())
    if (prior["status"] != "SAVED_CONTROL_USAGE_REVIEW_READY" or prior["saved_clips"] != 45
            or prior["new_integration_steps"] or prior["policy_inference_calls"]):
        raise ValueError("Expected the completed saved-only control analysis")
    historical = {row["path"]: row for row in prior["trace_sources"]}
    phases = phase_rows(args.phase / "foot_timeseries.csv")
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    frozen_dependencies = [native / "kinematics.py", native / "robot.py",
        args.runtime / "src/unisupermimic/rl/algos/supermimic/geometry.py",
        HERE.parent / "supermimic_for_q1/supermimic/kinematics/g1_fk.py",
        HERE.parent / "supermimic_q1_sole/tensor_geometry.py",
        HERE.parent / "supermimic_q1_sole/sole_geometry.py",
        HERE.parent / "supermimic_q1_gait_goal/support_geometry.py"]
    dependencies = [Path(__file__), HERE / "prepare_q1_gait_review.py",
        HERE.parent / "supermimic_q1_sole/reward_signal.py",
        HERE.parent / "supermimic_q1_gait_goal_analysis/saved_control_usage.py",
        HERE.parent / "supermimic_q1_gait_goal_analysis/saved_pd_demand.py", *frozen_dependencies]
    for path in dependencies:
        remember(path)
    scene = args.runtime / "q1_assets/scene.xml"
    kin = Q1Kinematics(scene, "cpu")
    sole = BoundedTensorSole.from_existing(TensorSole(SoleGeometry.from_scene(scene), kin.body_names, "cpu"))
    if len(Q1_JOINT_NAMES) != 22 or not all(Q1_JOINT_NAMES[i].startswith(
            "left_" if i < 6 else "right_") for i in range(12)):
        raise ValueError("Expected original left/right six-joint leg order")
    low, high = kin.q_lo.numpy(), kin.q_hi.numpy()
    calls, poses = 0, 0
    def heights(flat):
        nonlocal calls, poses
        values = []
        with torch.no_grad():
            for begin in range(0, len(flat), 128):
                part = torch.from_numpy(np.asarray(flat[begin:begin + 128], dtype=np.float32))
                result = kin.forward(*unflatten_state(part), all_bodies=True)
                values.append(sole(result[0], result[1]).numpy().astype(np.float64))
                calls += 1
                poses += len(part)
        return np.concatenate(values)

    saved, rows, timeseries, trace_sources = {}, [], [], []
    maximum_post_error, maximum_reference_error = 0., 0.
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        inputs = [trial / name for name in ("status.json", "postdelivery_status.json", "plan.json",
                  "run/checkpoint_selection.json", "evaluation/evaluation_library.json")]
        for path in inputs:
            remember(path)
        status, post, plan, selection, library = [json.loads(path.read_text()) for path in inputs]
        if (status["status"] != ("COMPLETED" if line == "base" else "FAILED")
                or post["status"] != "COMPLETED" or selection["user_accepted_checkpoint"] is not None
                or len(library["splits"]["train"]) != 330):
            raise ValueError("Expected original terminal outcomes,330 motions and null adoption")
        frozen = {row["path"]: row for row in plan["source"]}
        assets = [Path(name) for name in frozen if name.startswith(str(scene.parent) + "/")]
        for path in [*frozen_dependencies, *assets]:
            remember(path, frozen[str(path)])
        indices = [next(index for index, row in enumerate(library["splits"]["train"])
                        if row["name"] == name) for name in FIXED]
        for entry in selection["candidates"]:
            label = entry["label"]
            for repeat, receipt in enumerate(entry["traces"]):
                trace = Path(receipt["path"])
                result_path = trace.with_suffix(".json")
                if trace != trial / f"evaluation/{label}_repeat{repeat}.npz":
                    raise ValueError("Original trace path differs")
                before = remember(trace, historical[str(trace)])
                if before != receipt:
                    raise ValueError("Original checkpoint trace differs")
                result_before = remember(result_path, historical[str(result_path)])
                result = json.loads(result_path.read_text())
                if result["label"] != label or result["repeat"] != repeat or len(result["per_clip"]) != 330:
                    raise ValueError("Original result identity differs")
                with np.load(trace, allow_pickle=False) as archive:
                    arrays = {name: archive[name][:, indices].copy() for name in
                        ("states", "targets", "force", "slip", "frames", "alive", "residual",
                         "reference_q", "reference_p", "reference_quat", "alpha")}
                    arrays.update(initial=archive["initial"][indices], steps=archive["steps"][indices],
                                  reference_steps=archive["reference_steps"][indices])
                for slot, index in enumerate(indices):
                    clip = FIXED[slot]
                    if result["per_clip"][index]["name"] != clip:
                        raise ValueError("Fixed clip library order differs")
                    value = saved_clip(arrays, slot, result["per_clip"][index]["steps"])
                    count = len(value["frames"])
                    if count != int(arrays["reference_steps"][slot]) or not np.all(arrays["alpha"][:count, slot] == .5):
                        raise ValueError("Expected complete original fixed clips and staticalpha.5")
                    residual = arrays["residual"][:count, slot]
                    if (line == "base" or label == "initial") and np.any(residual):
                        raise ValueError("Base and initial residuals must be zero")
                    _, boundary, nominal14 = control_values(value["targets"], residual,
                        prior["native_default_q_rad"], prior["action_scale_rad"])
                    if boundary.any():
                        raise ValueError("Clamped final action makes nominal target geometry unidentifiable")
                    nominal_q = value["targets"].copy()
                    nominal_q[:, :14] = nominal14
                    previous = preceding_states(arrays["initial"][slot], value["states"])
                    reference_q = arrays["reference_q"][:count, slot]
                    reference = substitute_joints(previous, reference_q)
                    reference[:, :3] = arrays["reference_p"][:count, slot]
                    reference[:, 3:7] = arrays["reference_quat"][:count, slot]
                    variants = dict(pre=previous, post=value["states"],
                        target=substitute_joints(previous, value["targets"]),
                        nominal=substitute_joints(previous, nominal_q),
                        reference_same_root=substitute_joints(previous, reference_q), reference=reference)
                    geometry = {name: heights(flat) for name, flat in variants.items()}
                    if not np.any(residual):
                        np.testing.assert_array_equal(geometry["target"], geometry["nominal"])
                    outside = joint_outside(value["targets"], low, high)
                    nominal_outside = joint_outside(nominal_q, low, high)
                    relative = {name: relative_other(z) for name, z in geometry.items()}
                    header = dict(line=line, label=label, repeat=repeat, clip=clip,
                                  training_seconds=result["training_seconds"], saved_samples=count)
                    for foot, side in enumerate(("left", "right")):
                        phase = phases[line, label, repeat, clip, side]
                        if len(phase) != count:
                            raise ValueError("Original saved phase duration differs")
                        np.testing.assert_array_equal([int(row["global_reference_frame"]) for row in phase], value["frames"])
                        np.testing.assert_array_equal([float(row["native_force_N"]) for row in phase], value["force"][:, foot])
                        actual_z = np.array([float(row["sole_z_m"]) for row in phase])
                        reference_z = np.array([float(row["reference_sole_z_m"]) for row in phase])
                        np.testing.assert_allclose(geometry["post"][:, foot], actual_z, atol=2e-6, rtol=2e-5)
                        np.testing.assert_allclose(geometry["reference"][:, foot], reference_z, atol=2e-6, rtol=2e-5)
                        maximum_post_error = max(maximum_post_error, float(np.abs(geometry["post"][:, foot] - actual_z).max()))
                        maximum_reference_error = max(maximum_reference_error, float(np.abs(geometry["reference"][:, foot] - reference_z).max()))
                        swing = np.array([row["intent_swing"] == "True" for row in phase])
                        stance = np.array([row["intent_stance"] == "True" for row in phase])
                        relative_delta = relative["target"][:, foot] - relative["nominal"][:, foot]
                        sole_delta = geometry["target"][:, foot] - geometry["nominal"][:, foot]
                        joints = slice(foot * 6, foot * 6 + 6)
                        for condition, mask in dict(full=np.ones(count, dtype=bool),
                                raised_reference_10mm=swing & (reference_z >= .01), intent_stance=stance).items():
                            row = dict(**header, side=side, condition=condition, frames=int(mask.sum()))
                            for name in ("pre", "post", "target", "nominal", "reference_same_root"):
                                row[name + "_relative_other_mean_m"] = describe(relative[name][:, foot], mask)["mean"]
                            for name, data in (("residual_relative_delta", relative_delta), ("residual_sole_delta", sole_delta),
                                    ("target_relative_reference_error", relative["target"][:, foot] - relative["reference_same_root"][:, foot])):
                                row.update({name + "_" + metric + "_m": number for metric, number in describe(data, mask).items()})
                            row.update(target_leg_joint_outside_fraction=float(outside[mask, joints].mean()) if mask.any() else None,
                                target_leg_any_joint_outside_frame_fraction=float(outside[mask, joints].any(-1).mean()) if mask.any() else None,
                                nominal_leg_any_joint_outside_frame_fraction=float(nominal_outside[mask, joints].any(-1).mean()) if mask.any() else None,
                                residual_relative_lift_positive_fraction=float((relative_delta[mask] > 1e-6).mean()) if mask.any() else None)
                            rows.append(row)
                        for sample in range(count):
                            row = dict(**header, side=side, sample=sample, control_start_s=sample * .02,
                                reference_sample_end_s=(sample + 1) * .02, global_reference_frame=int(value["frames"][sample]),
                                intent_swing=bool(swing[sample]), reference_sole_z_m=float(reference_z[sample]),
                                native_force_final_solve_N=float(value["force"][sample, foot]),
                                target_leg_any_joint_outside=bool(outside[sample, joints].any()))
                            for name in ("pre", "post", "target", "nominal", "reference_same_root"):
                                row[name + "_sole_z_m"] = float(geometry[name][sample, foot])
                                row[name + "_relative_other_m"] = float(relative[name][sample, foot])
                            row["residual_relative_delta_m"] = float(relative_delta[sample])
                            timeseries.append(row)
                    saved[line, label, repeat, clip] = geometry
                remember(trace, before)
                remember(result_path, result_before)
                trace_sources.extend((before, result_before))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, static_fk_only=True)), flush=True)
    if len(saved) != 45 or len(timeseries) != 24030:
        raise ValueError("Expected all45 saved clips and24030 foot samples")
    for path, before in tracked.items():
        if identity(Path(path)) != before:
            raise ValueError("Saved analysis dependency changed: " + path)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "foot_target_pose_conditions.csv", rows)
    write_csv(args.output / "target_pose_timeseries.csv", timeseries)
    plot(args.output, saved)
    summary = dict(status="SAVED_TARGET_POSE_GEOMETRY_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()), trace_sources=trace_sources,
        saved_clips=len(saved), saved_states=sum(len(value["pre"]) for value in saved.values()),
        foot_condition_rows=len(rows), foot_timeseries_rows=len(timeseries), static_fk_batches=calls, static_fk_poses=poses,
        maximum_published_post_height_error_m=maximum_post_error,
        maximum_published_reference_height_error_m=maximum_reference_error,
        zero_residual_geometry_identity_verified=True, original_joint_names=Q1_JOINT_NAMES,
        original_joint_low_rad=low.tolist(), original_joint_high_rad=high.tolist(),
        root_and_velocities_held_fixed_for_substitution=True, new_integration_steps=0, policy_inference_calls=0,
        optimizer_updates=0, goal_complete=False, user_accepted_checkpoint=None,
        limitations=["PD equilibrium targets are not predicted next poses, reachable postures or requested foot trajectories.",
            "Static substitution holds the observed pre-state root fixed and does not solve ground contact or balance.",
            "Signed foot-height separation removes common root Z but retains observed root tilt and other joint effects.",
            "Target configurations may exceed model joint limits; all such fractions are descriptive and retained.",
            "Removing residual holds the observed teacher target fixed, not a physical head-off policy intervention.",
            "Reference conditions use the next saved reference sample; target root uses the preceding control state.",
            "Force remains a last5ms solve sample, not simultaneous with the saved post-pose or a20ms impulse.",
            "Only saved fixed clips are covered; no inference, integration, training or unique causal isolation.",
            "Plots use framewise medians; report statistics must first aggregate per repeat."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: summary[name] for name in ("status", "saved_clips", "saved_states",
        "foot_condition_rows", "foot_timeseries_rows", "static_fk_batches", "static_fk_poses",
        "maximum_published_post_height_error_m", "maximum_published_reference_height_error_m")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "base", "ppo", "phase", "control", "control-manifest", "phase-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
