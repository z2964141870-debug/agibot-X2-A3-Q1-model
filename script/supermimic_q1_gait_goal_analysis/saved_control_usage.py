"""Measure saved targets and residuals without policy inference or integration."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "project_tools"), str(HERE.parent / "supermimic_q1_sole")]
from prepare_q1_gait_review import FIXED, identity, write_csv
from reward_signal import saved_clip

DT, RADIUS, ACTION_BOUND = .02, .15, 20.
BOUND_TOLERANCE = 2e-5


def control_values(target, residual, default, scale):
    target, residual, default, scale = [np.asarray(value, dtype=np.float64)
                                        for value in (target, residual, default, scale)]
    if (target.ndim != 2 or target.shape[1] != 22 or residual.shape != (len(target), 14)
            or default.shape != (22,) or scale.shape != (22,)
            or any(not np.isfinite(value).all() for value in (target, residual, default, scale))
            or (scale <= 0).any() or (np.abs(residual) > RADIUS + 1e-7).any()):
        raise ValueError("Expected finite saved22 targets, bounded14 residuals and native scales")
    action = (target - default) / scale
    if (np.abs(action) > ACTION_BOUND + BOUND_TOLERANCE).any():
        raise ValueError("Targets do not fit the frozen evaluation base/action bounds")
    boundary = np.abs(action) >= ACTION_BOUND - BOUND_TOLERANCE
    # At the final bound the pre-clamp teacher action is not identifiable.
    nominal = target[:, :14] - residual
    nominal_action = (nominal - default[:14]) / scale[:14]
    valid = ~boundary[:, :14]
    if (np.abs(nominal_action[valid]) > ACTION_BOUND + BOUND_TOLERANCE).any():
        raise ValueError("Unclipped target cannot be reconstructed from a bounded teacher")
    nominal[~valid] = np.nan
    return action, boundary, nominal


def response_lag(target, actual, maximum=10):
    """Descriptive centered correlation on equal-length, non-wrapped windows."""
    target, actual = [np.asarray(value, dtype=np.float64) for value in (target, actual)]
    if (target.ndim != 1 or actual.shape != target.shape or maximum < 0
            or len(target) <= maximum + 4 or not np.isfinite([target, actual]).all()):
        raise ValueError("Expected aligned finite time series longer than the lag window")
    count = len(target) - maximum
    first = target[:count] - target[:count].mean()
    if np.std(first) < 1e-4:
        return dict(best_lag_ms=None, best_correlation=None, zero_lag_correlation=None,
                    lag_comparison_frames=count)
    scores = []
    for lag in range(maximum + 1):
        second = actual[lag:lag + count] - actual[lag:lag + count].mean()
        scores.append(None if np.std(second) < 1e-4 else float(
            np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1, 1)))
    valid = [index for index, score in enumerate(scores) if score is not None]
    best = max(valid, key=lambda index: scores[index]) if valid else None
    return dict(best_lag_ms=None if best is None else int(round(best * DT * 1000)),
                best_correlation=None if best is None else scores[best],
                zero_lag_correlation=scores[0], lag_comparison_frames=count)


def target_metrics(target, actual, reference, previous, residual=None, nominal=None):
    values = [np.asarray(value, dtype=np.float64) for value in (target, actual, reference, previous)]
    if (values[0].ndim != 2 or any(value.shape != values[0].shape for value in values)
            or not all(np.isfinite(value).all() for value in values) or not values[0].size):
        raise ValueError("Expected nonempty aligned target, post-state, reference and pre-state angles")
    target, actual, reference, previous = values
    rms = lambda value: float(np.sqrt(np.mean(np.square(value))))
    result = dict(target_actual_rmse_rad=rms(target - actual),
        target_reference_rmse_rad=rms(target - reference), actual_reference_rmse_rad=rms(actual - reference),
        target_correction_rms_rad=rms(target - previous), observed_step_rms_rad=rms(actual - previous))
    if residual is not None:
        residual, nominal = np.asarray(residual), np.asarray(nominal)
        if (residual.shape != target.shape or nominal.shape != target.shape
                or not np.isfinite(residual).all()):
            raise ValueError("Residual and identifiable nominal target must match the selected joints")
        valid = np.isfinite(nominal)
        active = valid & (np.abs(residual) > 1e-6) & (np.abs(reference - nominal) > 1e-6)
        result.update(residual_rms_rad=rms(residual), residual_abs_p95_rad=float(np.quantile(np.abs(residual), .95)),
            residual_abs_max_rad=float(np.abs(residual).max()),
            residual_over95pct_fraction=float((np.abs(residual) >= .95 * RADIUS).mean()),
            residual_over99pct_fraction=float((np.abs(residual) >= .99 * RADIUS).mean()),
            identifiable_residual_elements=int(valid.sum()), active_direction_elements=int(active.sum()),
            residual_toward_reference_fraction=float((residual[active] * (reference - nominal)[active] > 0).mean())
                if active.any() else None,
            target_squared_error_reduction_rad2=float(np.mean(
                np.square(nominal[valid] - reference[valid]) - np.square(target[valid] - reference[valid])))
                if valid.any() else None)
    return result


def phase_rows(path):
    groups = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["line"], row["label"], int(row["repeat"]), row["clip"], row["side"]
            groups.setdefault(key, []).append(row)
    return groups


def evaluation_context(manager_text, builder_text):
    manager, builder = ast.parse(manager_text), ast.parse(builder_text)
    cls = next(node for node in manager.body if isinstance(node, ast.ClassDef)
               and node.name == "SuperMimicJointOffsetEvent")
    call = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    branch = next(node for node in call.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "command.eval_mode")
    zero = [ast.unparse(node) for node in branch.body]
    function = next(node for node in builder.body if isinstance(node, ast.FunctionDef)
                    and node.name == "build_env_override")
    assignments = [ast.unparse(node) for node in ast.walk(function) if isinstance(node, ast.Assign)]
    if zero != ["command.xi[ids, 1:23] = 0.0"] or "motion['eval_mode'] = True" not in assignments:
        raise ValueError("Cannot establish frozen evaluation joint-offset convention")
    return dict(joint_offset_eval_branch=zero, eval_override_assignment="motion['eval_mode'] = True")


def curves_and_plot(output, saved, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = []
    for line in ("base", "ppo"):
        labels = list(dict.fromkeys(key[1] for key in saved if key[0] == line))
        median = {}
        for label in labels:
            for clip in FIXED:
                repeats = [saved[line, label, repeat, clip] for repeat in range(3)]
                value = {key: np.median(np.stack([repeat[key] for repeat in repeats]), axis=0)
                         for key in ("target", "actual", "reference", "residual", "height", "reference_height", "force")}
                median[label, clip] = value
                for foot, side in enumerate(("left", "right")):
                    indices = [foot * 6 + index for index in (0, 3, 4)]
                    for frame in range(len(value["target"])):
                        row = dict(line=line, label=label, clip=clip, side=side, seconds=(frame + 1) * DT,
                            sole_z_m=float(value["height"][frame, foot]),
                            reference_sole_z_m=float(value["reference_height"][frame, foot]),
                            native_force_N=float(value["force"][frame, foot]))
                        for joint, index in zip(("hip_pitch", "knee", "ankle_pitch"), indices):
                            row.update({joint + "_" + key + "_rad": float(value[key][frame, index])
                                        for key in ("target", "actual", "reference", "residual")})
                        curves.append(row)
        figure, axes = plt.subplots(5, 2, figsize=(13, 12), constrained_layout=True, sharex="col")
        for column, clip in enumerate(FIXED[:2]):
            initial, final = median["initial", clip], median[labels[-1], clip]
            seconds = (np.arange(len(initial["target"])) + 1) * DT
            for row, index in enumerate((9, 10, 3)):
                ax = axes[row, column]
                ax.plot(seconds, np.degrees(initial["reference"][:, index]), color="#222222", lw=1.2, label="Reference")
                for value, color, label in ((initial, "#888888", "Initial"), (final, "#087f8c", "After training")):
                    ax.plot(seconds, np.degrees(value["target"][:, index]), color=color, ls="--", lw=.9, label=label + " target")
                    ax.plot(seconds, np.degrees(value["actual"][:, index]), color=color, lw=1, label=label + " actual")
                ax.set_ylabel(names[index].replace("_joint", "").replace("_", " ") + " (deg)")
            ax = axes[3, column]
            ax.plot(seconds, initial["reference_height"][:, 1] * 1000, color="#222222", lw=1.2, label="Reference")
            for value, color, label in ((initial, "#888888", "Initial"), (final, "#087f8c", "After training")):
                ax.plot(seconds, value["height"][:, 1] * 1000, color=color, lw=1, label=label)
            ax.axhline(5, color="#bf553c", ls=":", lw=.8)
            ax.set_ylabel("Right sole Z (mm)")
            ax = axes[4, column]
            if line == "ppo":
                for index, color in ((6, "#a34e24"), (9, "#087f8c"), (10, "#6c5b99")):
                    ax.plot(seconds, final["residual"][:, index] * 1000, color=color, lw=1,
                            label=names[index].replace("right_", "").replace("_joint", ""))
                ax.axhline(0, color="#222222", lw=.6)
                ax.set_ylabel("Right residual (mrad)")
            else:
                for value, color, label in ((initial, "#888888", "Initial"), (final, "#087f8c", "After training")):
                    ax.plot(seconds, value["force"][:, 1], color=color, lw=1, label=label)
                ax.axhline(5, color="#bf553c", ls=":", lw=.8)
                ax.set_ylabel("Right native force (N)")
            axes[0, column].set_title("Stop walk" if column == 0 else "Start walk")
            axes[4, column].set_xlabel("Saved control time (s)")
        for ax in axes.flat:
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=7, ncol=3)
        axes[4, 0].legend(fontsize=8, ncol=3)
        figure.suptitle(f"Q1 {line.upper()}: saved commands and response, framewise medians over 3 repeats", fontsize=12)
        figure.supxlabel("Targets produce the aligned post-state after 20ms. Curves are descriptive; residuals alone do not establish actuator capability.", fontsize=8)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{line}_saved_control_usage.{extension}", dpi=150)
        plt.close(figure)
    write_csv(output / "median_control_timeseries.csv", curves)
    return len(curves)


def run(args):
    import mujoco

    if args.output.exists():
        raise FileExistsError(args.output)
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    builder = args.runtime / "src/unisupermimic/scripts/train_supermimic.py"
    dependencies = [Path(__file__), HERE.parent / "project_tools/prepare_q1_gait_review.py",
        HERE.parent / "supermimic_q1_sole/reward_signal.py", native / "robot.py", native / "control.json",
        native / "manager_terms.py", builder, args.runtime / "q1_config.yaml",
        HERE.parent / "supermimic_q1_hp/worker.py", HERE.parent / "supermimic_q1_balance/control.py",
        HERE.parent / "supermimic_q1_gait_comp/physical.py", HERE.parent / "supermimic_q1_gait_goal/trial_support.py",
        HERE.parent / "supermimic_q1_gait_goal/train.py", HERE.parent / "supermimic_q1_gait_goal_review/train.py",
        HERE.parent / "supermimic_q1_balance/balance_adapter.py"]
    tracked = {}
    def remember(paths):
        for path in paths:
            value = identity(path)
            if str(path) in tracked and value != tracked[str(path)]:
                raise ValueError("Analysis dependency changed: " + str(path))
            tracked[str(path)] = value
    remember(dependencies + [args.phase / "summary.json", args.phase / "foot_timeseries.csv"])
    context = evaluation_context((native / "manager_terms.py").read_text(), builder.read_text())
    config = json.loads((args.runtime / "q1_config.yaml").read_text())
    control = json.loads((native / "control.json").read_text())
    names = control["joints"]
    if (len(names) != 22 or config["env"]["scene"]["entities"]["robot"]["joint_names"] != names
            or config["env"]["scene"]["default_keyframe_name"] != "stand"
            or config["env"]["ctrl_dt"] != DT):
        raise ValueError("Native22 joint order, stand default and20ms control required")
    model = mujoco.MjModel.from_xml_path(str(args.runtime / "q1_assets/scene.xml"))
    default = np.array([model.keyframe("stand").qpos[model.jnt_qposadr[model.joint(name).id]] for name in names])
    np.testing.assert_allclose(default, [control["control"][name]["default"] for name in names], atol=1e-7, rtol=0)
    scale = np.array([.25 * control["control"][name]["effort"] / control["control"][name]["kp"] for name in names])
    phase_summary = json.loads((args.phase / "summary.json").read_text())
    phase_sources = {row["path"]: row for row in phase_summary["trace_sources"]}
    phases = phase_rows(args.phase / "foot_timeseries.csv")
    saved, joint_rows, foot_rows, trace_sources, training = {}, [], [], [], []
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        paths = [trial / name for name in ("status.json", "postdelivery_status.json", "plan.json",
            "run/checkpoint_selection.json", "evaluation/evaluation_library.json", "run/training_evidence.jsonl")]
        remember(paths)
        status, post, plan, selection, library = [json.loads(path.read_text()) for path in paths[:-1]]
        if (status["status"] != ("COMPLETED" if line == "base" else "FAILED") or post["status"] != "COMPLETED"
                or selection["user_accepted_checkpoint"] is not None or len(library["splits"]["train"]) != 330):
            raise ValueError("Original saved outcomes, original330 library and null adoption required")
        rows = library["splits"]["train"]
        indices = [next(index for index, row in enumerate(rows) if row["name"] == clip) for clip in FIXED]
        frozen = {row["path"]: row for row in plan["source"]}
        assets = [Path(path) for path in frozen if path.startswith(str(args.runtime / "q1_assets") + "/")]
        for path in dependencies[3:] + assets:
            if str(path) not in frozen or identity(path) != frozen[str(path)]:
                raise ValueError("Frozen evaluation/control source differs: " + str(path))
        remember(assets)
        for entry in selection["candidates"]:
            label = entry["label"]
            for repeat, historical in enumerate(entry["traces"]):
                trace = Path(historical["path"])
                before, result_path = identity(trace), trace.with_suffix(".json")
                result_before = identity(result_path)
                result = json.loads(result_path.read_text())
                if (before != historical or phase_sources.get(str(trace)) != before
                        or phase_sources.get(str(result_path)) != result_before
                        or trace != trial / f"evaluation/{label}_repeat{repeat}.npz"
                        or result["repeat"] != repeat or result["label"] != label or len(result["per_clip"]) != 330):
                    raise ValueError("Historical trace, result or phase provenance differs")
                with np.load(trace, allow_pickle=False) as archive:
                    arrays = {key: archive[key][:, indices].copy() for key in
                        ("states", "targets", "force", "slip", "frames", "alive", "residual", "reference_q", "alpha")}
                    arrays.update(steps=archive["steps"][indices], reference_steps=archive["reference_steps"][indices],
                                  initial=archive["initial"][indices])
                for slot, index in enumerate(indices):
                    clip = FIXED[slot]
                    if result["per_clip"][index]["name"] != clip:
                        raise ValueError("Saved library order differs")
                    value = saved_clip(arrays, slot, int(result["per_clip"][index]["steps"]))
                    count = len(value["frames"])
                    if arrays["residual"].shape[2:] != (14,) or not np.all(arrays["alpha"][:count, slot] == .5):
                        raise ValueError("Original residual14 and staticalpha.5 required")
                    residual = arrays["residual"][:count, slot].astype(np.float64)
                    if (line == "base" or label == "initial") and np.any(residual):
                        raise ValueError("Base and initial traces must have zero residual")
                    target, actual, reference = value["targets"].astype(np.float64), value["states"][:, 13:35].astype(np.float64), arrays["reference_q"][:count, slot].astype(np.float64)
                    previous = np.concatenate((arrays["initial"][slot:slot+1, 13:35], actual[:-1]))
                    action, boundary, nominal = control_values(target, residual, default, scale)
                    header = dict(line=line, label=label, repeat=repeat, clip=clip,
                                  training_seconds=result["training_seconds"], steps=count)
                    for joint, name in enumerate(names):
                        extra = dict(residual=residual[:, joint:joint+1], nominal=nominal[:, joint:joint+1]) if joint < 14 else {}
                        joint_rows.append(dict(**header, joint=name,
                            **target_metrics(*(value[:, joint:joint+1] for value in (target, actual, reference, previous)), **extra),
                            action_abs_max=float(np.abs(action[:, joint]).max()),
                            final_action_boundary_fraction=float(boundary[:, joint].mean()),
                            target_range_rad=float(np.ptp(target[:, joint])), actual_range_rad=float(np.ptp(actual[:, joint])),
                            reference_range_rad=float(np.ptp(reference[:, joint])),
                            target_step_delta_rms_rad=float(np.sqrt(np.mean(np.diff(target[:, joint])**2))),
                            **response_lag(target[:, joint], actual[:, joint])))
                    heights, reference_heights, forces = [], [], []
                    for foot, side in enumerate(("left", "right")):
                        phase = phases[line, label, repeat, clip, side]
                        if len(phase) != count:
                            raise ValueError("Saved foot time series duration differs")
                        np.testing.assert_array_equal([int(row["global_reference_frame"]) for row in phase], value["frames"])
                        force = np.array([float(row["native_force_N"]) for row in phase])
                        np.testing.assert_array_equal(force, value["force"][:, foot])
                        height = np.array([float(row["sole_z_m"]) for row in phase])
                        reference_height = np.array([float(row["reference_sole_z_m"]) for row in phase])
                        swing, stance = (np.array([row[key] == "True" for row in phase]) for key in ("intent_swing", "intent_stance"))
                        conditions = dict(full=np.ones(count, dtype=bool), intent_swing=swing,
                            raised_reference_10mm=swing & (reference_height >= .01), intent_stance=stance, native_loaded=force > 5.)
                        joints = slice(foot * 6, foot * 6 + 6)
                        for condition, mask in conditions.items():
                            metrics = target_metrics(*(array[mask, joints] for array in (target, actual, reference, previous)),
                                residual=residual[mask, joints], nominal=nominal[mask, joints]) if mask.any() else {}
                            foot_rows.append(dict(**header, side=side, condition=condition, frames=int(mask.sum()),
                                **metrics, final_action_boundary_fraction=float(boundary[mask, joints].mean()) if mask.any() else None,
                                sole_z_mean_m=float(height[mask].mean()) if mask.any() else None,
                                reference_sole_z_mean_m=float(reference_height[mask].mean()) if mask.any() else None,
                                native_loaded_fraction=float((force[mask] > 5.).mean()) if mask.any() else None))
                        heights.append(height)
                        reference_heights.append(reference_height)
                        forces.append(force)
                    saved[line, label, repeat, clip] = dict(target=target, actual=actual, reference=reference,
                        residual=residual, height=np.stack(heights, -1), reference_height=np.stack(reference_heights, -1),
                        force=np.stack(forces, -1))
                if identity(trace) != before or identity(result_path) != result_before:
                    raise ValueError("Saved source changed during analysis")
                trace_sources.extend((before, result_before))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_only=True)), flush=True)
        for text in paths[-1].read_text().splitlines():
            row = json.loads(text)
            if line == "ppo":
                training.append(dict(step=row["step"], training_seconds=row["training_seconds"],
                    transitions=row["transitions"], sampled_residual14_rms_rad=float(np.sqrt(
                        row["residual_squared_sum"] / (row["transitions"] * 14))),
                    latent_mean_rms=row.get("latent_mean_rms"), actor_optimizer_steps=row["actor_optimizer_steps"]))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("joint_metrics", joint_rows), ("foot_control_conditions", foot_rows), ("ppo_training_residual", training)):
        write_csv(args.output / (name + ".csv"), rows)
    curve_rows = curves_and_plot(args.output, saved, names)
    if list(tracked.values()) != [identity(Path(path)) for path in tracked]:
        raise ValueError("Offline analysis source changed")
    summary = dict(status="SAVED_CONTROL_USAGE_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()), trace_sources=trace_sources,
        saved_clips=len(saved), saved_states=sum(len(value["actual"]) for value in saved.values()),
        joint_rows=len(joint_rows), foot_condition_rows=len(foot_rows), median_timeseries_rows=curve_rows,
        ppo_training_rows=len(training), historical_initial_and_stage_trace_sha_verified=True,
        evaluation_context=context, native_default_q_rad=default.tolist(), action_scale_rad=scale.tolist(),
        residual_radius_rad=RADIUS, action_bound=ACTION_BOUND, action_boundary_tolerance=BOUND_TOLERANCE,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        goal_complete=False, user_accepted_checkpoint=None,
        limits=["Saved targets produce the aligned post-state after20ms; preceding state is initial or prior saved state.",
            "Saved residual precedes final action clamp. Nominal target is inferred only for elements strictly inside final bounds.",
            "Boundary membership is not proof of a nonzero clipped amount; upstream teacher clipping cannot be measured directly.",
            "Pose targets and residuals are not motor torque/current. Actuator saturation and feasible authority cannot be concluded.",
            "Command-to-reference error reduction holds the observed teacher command fixed; not a new physical policy intervention.",
            "Best0..200ms centered correlation is descriptive, uses equal sample counts, and is not a causal actuator delay.",
            "Deterministic evaluation head output and sampled training residual RMS describe different distributions.",
            "Phase/geometry/force are reused from the prior saved-only audit; native force is final5ms, not20ms impulse.",
            "Framewise medians in figures differ from medians of per-repeat summary statistics.",
            "Cross-stage differences do not isolate a failure cause or establish that increasing residual radius would help."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "joint_rows", "foot_condition_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "base", "ppo", "phase", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
