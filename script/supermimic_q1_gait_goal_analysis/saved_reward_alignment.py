"""Compare analytic pose rewards on completed saved trials, without a rollout."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "project_tools"),
               str(HERE.parent / "supermimic_q1_sole")]
from prepare_q1_gait_review import FIXED, identity, write_csv
from reward_signal import saved_clip

GRADIENT_EPSILON = 1e-8


def alignment(first, second, epsilon=GRADIENT_EPSILON):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if (first.ndim != 2 or first.shape != second.shape or first.shape[1] not in (6, 12)
            or not np.isfinite(first).all() or not np.isfinite(second).all() or epsilon <= 0):
        raise ValueError("Expected aligned finite leg-radian gradients")
    left, right = np.linalg.norm(first, axis=-1), np.linalg.norm(second, axis=-1)
    valid = (left > epsilon) & (right > epsilon)
    cosine = np.full(len(first), np.nan)
    cosine[valid] = np.clip(np.sum(first[valid] * second[valid], -1) / (left[valid] * right[valid]), -1, 1)
    return cosine, valid, left, right


def summarize_alignment(first, second, mask):
    cosine, valid, left, right = alignment(first, second)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != valid.shape:
        raise ValueError("Expected one boolean condition per saved state")
    selected = mask & valid
    return dict(frames=int(mask.sum()), valid_gradient_pairs=int(selected.sum()),
        first_zero_frames=int((mask & (left <= GRADIENT_EPSILON)).sum()),
        second_zero_frames=int((mask & (right <= GRADIENT_EPSILON)).sum()),
        cosine_median=float(np.median(cosine[selected])) if selected.any() else None,
        opposing_fraction=float((cosine[selected] < 0).mean()) if selected.any() else None,
        first_to_second_norm_median=float(np.median(left[selected] / right[selected])) if selected.any() else None,
        second_derivative_along_first_mean=float(np.mean(
            np.sum(first[selected] * second[selected], -1) / left[selected])) if selected.any() else None)


def finite_mean(values, mask):
    selected = np.asarray(values)[mask]
    return float(selected.mean()) if len(selected) else None


def median_over_repeats(values):
    values = np.stack(values)
    result = np.full(values.shape[1], np.nan)
    valid = np.isfinite(values).any(0)
    result[valid] = np.nanmedian(values[:, valid], axis=0)
    return result


def score_components(mdp, candidate, state, frames):
    import torch
    from reward_candidates import lowest_material_velocity, phase_mean
    from unisupermimic.rl.algos.supermimic.geometry import yaw_mat

    zeros = torch.zeros(len(frames), 22)
    _, detail = mdp.reward(state, frames, zeros, zeros, terms=True)
    cfg, dt = mdp.cfg, .02
    scores = {"body_position": dt * torch.exp(-detail["e_body_pos"] / cfg.body_pos_std**2),
        "body_orientation": dt * torch.exp(-detail["e_body_ori"] / cfg.body_ori_std**2),
        "body_linear_velocity": dt * torch.exp(-detail["e_body_linvel"] / cfg.body_linvel_std**2),
        "body_angular_velocity": dt * torch.exp(-detail["e_body_angvel"] / cfg.body_angvel_std**2),
        "root_position": .5 * dt * torch.exp(-detail["e_root_pos"] / cfg.root_pos_std**2)}
    from unisupermimic.rl.algos.supermimic.geometry import rot_angle_sq
    scores["root_orientation"] = .5 * dt * torch.exp(
        -rot_angle_sq(mdp.motion.root_R[frames], state[1]) / cfg.root_ori_std**2)
    torch.testing.assert_close(sum(scores.values()), detail["track"], atol=2e-8, rtol=2e-6)
    scores["native_tracking"] = detail["track"]
    scores["joint_limit"] = cfg.w_joint_limit * dt * detail["joint_limit"]
    reference = candidate.foot_reference
    full = mdp.kin.forward(*state, all_bodies=True)
    offsets = reference.sole.offsets(full[1])
    height = full[0][:, reference.sole.body_ids, 2] + offsets[..., 2]
    error = height - reference.reference_sole[frames]
    swing, stance = reference.swing[frames], reference.stance[frames]
    height_score = torch.exp(-(error / .02).square())
    velocity = lowest_material_velocity(reference.sole, full, offsets)
    speed_score = torch.exp(-velocity[..., :2].square().sum(-1) / .15**2)
    scores["base_swing"] = .005 * phase_mean(height_score, swing)
    scores["base_stance"] = .005 * phase_mean(height_score * speed_score, stance)
    scores["ppo_z"] = .01 * phase_mean(height_score, swing)
    for foot, side in enumerate(("left", "right")):
        scores[side + "_z"] = .01 * height_score[:, foot] * swing[:, foot] / swing.sum(-1).clamp(min=1)
    actual, target = full[0][:, reference.ids], reference.positions[frames]
    local = lambda rotation, value: (yaw_mat(rotation).transpose(-1, -2).unsqueeze(1)
                                     @ value.unsqueeze(-1)).squeeze(-1)
    xy_error = (local(state[1], actual - state[0][:, None])
                - local(mdp.motion.root_R[frames], target - mdp.motion.root_p[frames, None]))[..., :2]
    scores["ppo_xy"] = .01 * phase_mean(torch.exp(-(xy_error / .1).square().sum(-1)), swing)
    still = reference.targets(frames)[-1]
    gravity_error = (state[1][:, 2] - mdp.motion.root_R[frames, 2]).square().sum(-1)
    scores["stationary"] = .01 * torch.exp(-state[2][:, :2].square().sum(-1) / .1**2
        - state[3].square().sum(-1) / .5**2 - gravity_error / .2**2) * still
    return scores, height, swing, still


def derivatives(scores, q):
    import torch

    gradients = {}
    for name, value in scores.items():
        gradient = torch.autograd.grad(value.sum(), q, retain_graph=True, allow_unused=True)[0] if value.requires_grad else None
        gradients[name] = torch.zeros_like(q) if gradient is None else gradient
        if not bool(torch.isfinite(gradients[name]).all()):
            raise ValueError("Nonfinite analytic gradient: " + name)
    return {name: value[:, :12].detach().numpy().astype(np.float64) for name, value in gradients.items()}


def plot(output, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line in ("base", "ppo"):
        labels = list(dict.fromkeys(key[1] for key in saved if key[0] == line))
        figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True, sharex="col")
        for column, clip in enumerate(FIXED[:2]):
            for label, color, title in (("initial", "#777777", "Initial"),
                                         (labels[-1], "#087f8c", "After training")):
                values = [saved[(line, label, repeat, clip)] for repeat in range(3)]
                seconds = (np.arange(len(values[0]["still"])) + 1) * .02
                cosine = median_over_repeats([value["cosine"] for value in values])
                ratio = median_over_repeats([value["norm_ratio"] for value in values])
                root_score = median_over_repeats([value["root_score"] for value in values])
                axes[0, column].plot(seconds, ratio, color=color, lw=1, label=title)
                axes[1, column].plot(seconds, cosine, color=color, lw=1)
                axes[2, column].plot(seconds, root_score, color=color, lw=1)
            axes[0, column].set_title("Stop walk" if column == 0 else "Start walk")
            axes[0, column].set_yscale("symlog", linthresh=.1)
            axes[1, column].axhline(0, color="#bf553c", lw=.8)
            axes[1, column].set_ylim(-1.05, 1.05)
            axes[2, column].plot(seconds, values[0]["still"].astype(float), color="#252525", ls=":", lw=1,
                                 label="Reference stationary gate")
            axes[2, column].set_ylim(-.05, 1.05)
            axes[2, column].set_xlabel("Saved control time (s)")
        for row, label in enumerate(("Z / tracking gradient norm", "Cosine: Z vs tracking", "Root orientation score / maximum")):
            axes[row, 0].set_ylabel(label)
        for ax in axes.flat:
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=9)
        axes[2, 0].legend(fontsize=8)
        figure.suptitle(f"Q1 {line.upper()}: reward alignment on saved poses, medians over 3 repeats", fontsize=12)
        figure.supxlabel("Gradients use the same 12 leg angles, with root, velocities and contact measurements fixed. Gaps retain undefined cosines.", fontsize=8)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{line}_saved_reward_alignment.{extension}", dpi=150)
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(HERE.parent / "supermimic_q1_gait_goal"),
        str(HERE.parent / "supermimic_q1_contact_comp"), str(args.runtime / "src"), str(args.runtime)]
    import torch
    from reward_candidates import PhaseSoleMDP
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, quat_to_mat
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import default_joint_pos_list, Q1_JOINT_NAMES

    torch.set_num_threads(8)
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    dependencies = [Path(__file__), HERE.parent / "supermimic_q1_sole/reward_signal.py",
        HERE.parent / "project_tools/prepare_q1_gait_review.py", args.scene]
    dependencies.extend(native / name for name in ("mdp.py", "mdp_base.py", "candidate_options.json",
                                                 "kinematics.py", "motion.py", "motion_base.py", "robot.py"))
    dependencies.extend(HERE.parent / path for path in (
        "supermimic_q1_gait_goal/reward_candidates.py", "supermimic_q1_gait_goal/support_geometry.py",
        "supermimic_q1_sole/sole_reference.py", "supermimic_q1_sole/tensor_geometry.py",
        "supermimic_q1_sole/sole_geometry.py", "supermimic_q1_contact_comp/contact_reference.py",
        "supermimic_q1_gait_comp/rewards.py", "supermimic_for_q1/supermimic/kinematics/g1_fk.py"))
    dependencies.append(args.runtime / "src/unisupermimic/rl/algos/supermimic/geometry.py")
    options = json.loads((native / "candidate_options.json").read_text())
    if options["joint_tracking_reward"] or options["joint_error_observation"]:
        raise ValueError("Expected frozen sampler-only native MDP")
    tracked, trace_sources = {}, []
    def remember(paths):
        for path in paths:
            value = identity(path)
            if str(path) in tracked and tracked[str(path)] != value:
                raise ValueError("Analysis dependency changed: " + str(path))
            tracked[str(path)] = value
    remember(dependencies)
    library_path = args.base / "evaluation/evaluation_library.json"
    library = json.loads(library_path.read_text())["splits"]["train"]
    if len(library) != 330 or len({row["name"] for row in library}) != 330:
        raise ValueError("Expected original unique330 library")
    indices = [next(index for index, row in enumerate(library) if row["name"] == clip) for clip in FIXED]
    fixed_rows = [library[index] for index in indices]
    for row in fixed_rows:
        path = Path(row["path"])
        if identity(path)["sha256"] != row["sha256"]:
            raise ValueError("Original reference changed")
        remember((path, path.with_suffix(".json")))
    kin = Q1Kinematics(args.scene, "cpu")
    motion = MotionLib([row["path"] for row in fixed_rows], kin, "cpu", src_fps=30.)
    default = torch.tensor(default_joint_pos_list())
    mdp = MDP(kin, motion, default, observation_mode="root_xy")
    candidate = PhaseSoleMDP(kin, motion, default, mdp.cfg, scene=args.scene, library=library_path)
    if len(Q1_JOINT_NAMES[:12]) != 12 or any(not name.startswith(("left_", "right_")) for name in Q1_JOINT_NAMES[:12]):
        raise ValueError("Original twelve leg joint coordinates required")
    rows, feet, curves, coverage, saved = [], [], [], [], {}
    direct_checks = 0
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        paths = [trial / name for name in ("status.json", "postdelivery_status.json", "plan.json",
                                          "run/checkpoint_selection.json", "evaluation/evaluation_library.json")]
        remember(paths)
        status, post, plan, selection, source_library = (json.loads(path.read_text()) for path in paths)
        if (status["status"] != ("COMPLETED" if line == "base" else "FAILED")
                or post["status"] != "COMPLETED" or selection["user_accepted_checkpoint"] is not None
                or source_library["splits"]["train"] != library):
            raise ValueError("Completed original outcomes and null adoption required")
        frozen = {row["path"]: row for row in plan["source"]}
        frozen_paths = [path for path in dependencies[3:] if str(path) in frozen]
        frozen_paths.extend(Path(row["path"]) for row in fixed_rows)
        frozen_paths.extend(Path(row["path"]).with_suffix(".json") for row in fixed_rows)
        for path in frozen_paths:
            if identity(path) != frozen[str(path)]:
                raise ValueError("Frozen reward/reference source changed: " + str(path))
        remember(Path(path) for path in frozen if path.startswith(str(args.scene.parent) + "/"))
        for entry in selection["candidates"]:
            label = entry["label"]
            for repeat, historical in enumerate(entry["traces"]):
                trace_path = Path(historical["path"])
                before = identity(trace_path)
                result_path = trace_path.with_suffix(".json")
                result_before = identity(result_path)
                result = json.loads(result_path.read_text())
                if before != historical or trace_path != trial / f"evaluation/{label}_repeat{repeat}.npz":
                    raise ValueError("Historical saved trace identity differs")
                if result["label"] != label or result["repeat"] != repeat or len(result["per_clip"]) != 330:
                    raise ValueError("Original saved result order differs")
                with np.load(trace_path, allow_pickle=False) as archive:
                    reference_steps = archive["reference_steps"]
                    starts = np.r_[0, np.cumsum(reference_steps + 1)[:-1]]
                    arrays = {key: archive[key][:, indices].copy() for key in (
                        "states", "targets", "force", "slip", "frames", "alive", "reference_p", "reference_quat", "reference_q")}
                    arrays.update(steps=archive["steps"][indices], reference_steps=reference_steps[indices])
                for slot, index in enumerate(indices):
                    clip = result["per_clip"][index]
                    if clip["name"] != FIXED[slot]:
                        raise ValueError("Saved fixed clip identity differs")
                    value = saved_clip(arrays, slot, clip["steps"])
                    count = len(value["frames"])
                    np.testing.assert_array_equal(value["frames"], starts[index] + np.arange(1, count + 1))
                    if int(motion.clip_len[slot]) != int(reference_steps[index]) + 1:
                        raise ValueError("Reduced reference duration differs")
                    frames = motion.clip_start[slot] + torch.arange(1, count + 1)
                    torch.testing.assert_close(motion.root_p[frames], torch.from_numpy(arrays["reference_p"][:count, slot]), atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(motion.root_R[frames], quat_to_mat(torch.from_numpy(arrays["reference_quat"][:count, slot])), atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(motion.q[frames], torch.from_numpy(arrays["reference_q"][:count, slot]), atol=2e-6, rtol=2e-5)
                    gradient_parts, score_parts, height_parts, swing_parts, still_parts = {}, {}, [], [], []
                    for begin in range(0, count, 64):
                        end = min(begin + 64, count)
                        flat = torch.from_numpy(value["states"][begin:end])
                        state = list(unflatten_state(flat))
                        state[4] = state[4].detach().clone().requires_grad_()
                        scores, height, swing, still = score_components(mdp, candidate, tuple(state), frames[begin:end])
                        gradients = derivatives(scores, state[4])
                        for name, gradient in gradients.items():
                            gradient_parts.setdefault(name, []).append(gradient)
                            score_parts.setdefault(name, []).append(scores[name].detach().numpy())
                        height_parts.append(height.detach().numpy())
                        swing_parts.append(swing.numpy())
                        still_parts.append(still.numpy())
                        if begin == 0:
                            with torch.no_grad():
                                take, zeros = min(3, end), torch.zeros(min(3, end), 22)
                                sample = tuple(part[:take] for part in state)
                                _, native_parts = mdp.reward(sample, frames[:take], zeros, zeros, terms=True)
                                _, base_parts = candidate.reward(sample, frames[:take], zeros, zeros, terms=True)
                                torch.testing.assert_close(base_parts["phase_bonus"],
                                    scores["base_swing"][:take] + scores["base_stance"][:take], atol=2e-8, rtol=2e-5)
                                _, ppo_parts = candidate.foot_reference.reward(native_parts["track"], flat[:take], frames[:take],
                                    zeros, zeros, torch.zeros(take, 14), torch.from_numpy(value["force"][:take]),
                                    torch.from_numpy(value["slip"][:take]))
                                for name, expected in (("swing_bonus", scores["ppo_z"][:take] + scores["ppo_xy"][:take]),
                                                       ("stationary_bonus", scores["stationary"][:take])):
                                    torch.testing.assert_close(ppo_parts[name], expected, atol=3e-8, rtol=3e-5)
                                direct_checks += take
                    gradients = {name: np.concatenate(parts) for name, parts in gradient_parts.items()}
                    scores = {name: np.concatenate(parts) for name, parts in score_parts.items()}
                    height, swing, still = np.concatenate(height_parts), np.concatenate(swing_parts), np.concatenate(still_parts)
                    reference_height = candidate.foot_reference.reference_sole[frames].numpy()
                    raised = swing & (reference_height >= .01)
                    header = dict(line=line, label=label, repeat=repeat, clip=FIXED[slot], training_seconds=result["training_seconds"])
                    masks = dict(full=np.ones(count, bool), intent_swing=swing.any(-1),
                        raised_reference_10mm=raised.any(-1), raised_and_underheight5mm=(raised & (height <= .005)).any(-1),
                        stationary_gate=still, moving_reference=(motion.root_v[frames, :2].norm(dim=-1) >= .05).numpy(),
                        second_half=np.arange(count) >= count // 2)
                    z_name = "base_swing" if line == "base" else "ppo_z"
                    for condition, mask in masks.items():
                        for other in ("native_tracking", "body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity", "ppo_xy"):
                            rows.append(dict(**header, condition=condition, first=z_name, second=other,
                                             **summarize_alignment(gradients[z_name], gradients[other], mask)))
                    for foot, side in enumerate(("left", "right")):
                        leg = slice(foot * 6, (foot + 1) * 6)
                        for condition, mask in (("intent_swing", swing[:, foot]), ("raised_reference_10mm", raised[:, foot])):
                            feet.append(dict(**header, side=side, condition=condition,
                                **summarize_alignment(gradients[side + "_z"][:, leg], gradients["native_tracking"][:, leg], mask)))
                    coverage.append(dict(**header, frames=count, stationary_gate_frames=int(still.sum()),
                        stationary_gate_fraction=float(still.mean()),
                        stationary_bonus_mean=float(scores["stationary"].mean()) if line == "ppo" else None,
                        native_body_orientation_score_mean=float(scores["body_orientation"].mean()),
                        native_root_orientation_score_mean=float(scores["root_orientation"].mean()),
                        native_tracking_score_mean=float(scores["native_tracking"].mean()),
                        swing_z_score_mean=float(scores[z_name].mean()),
                        moving_reference_frames=int(masks["moving_reference"].sum()),
                        root_orientation_score_mean_when_moving=finite_mean(scores["root_orientation"], masks["moving_reference"])))
                    cosine, valid, znorm, native_norm = alignment(gradients[z_name], gradients["native_tracking"])
                    ratio = np.full(count, np.nan)
                    ratio[valid] = znorm[valid] / native_norm[valid]
                    saved[(line, label, repeat, FIXED[slot])] = dict(cosine=cosine, norm_ratio=ratio,
                        root_score=scores["root_orientation"] / .01, still=still)
                    for frame in range(count):
                        row = dict(**header, seconds=(frame + 1) * .02,
                            global_reference_frame=int(value["frames"][frame]),
                            stationary_gate=bool(still[frame]), swing_feet=int(swing[frame].sum()),
                            raised_reference_feet=int(raised[frame].sum()),
                            z_tracking_cosine=float(cosine[frame]) if valid[frame] else None,
                            z_to_tracking_norm=float(ratio[frame]) if valid[frame] else None)
                        row.update({name + "_score": float(score[frame]) for name, score in scores.items()})
                        row.update({name + "_leg_gradient_norm": float(np.linalg.norm(gradient[frame])) for name, gradient in gradients.items()})
                        curves.append(row)
                if identity(trace_path) != before or identity(result_path) != result_before:
                    raise ValueError("Saved physical evidence changed")
                trace_sources.extend((before, result_before))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_only=True)), flush=True)
    if list(tracked.values()) != [identity(Path(row["path"])) for row in tracked.values()]:
        raise ValueError("Analysis or native sources changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("gradient_conditions", rows), ("foot_gradient_conditions", feet),
                          ("stationary_coverage", coverage), ("reward_gradient_timeseries", curves)):
        write_csv(args.output / (name + ".csv"), records)
    plot(args.output, saved)
    summary = dict(status="SAVED_REWARD_ALIGNMENT_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()), trace_sources=trace_sources,
        native_candidate_options=options, native_reward_cfg=asdict(mdp.cfg), leg_joint_names=Q1_JOINT_NAMES[:12],
        reference_clips=FIXED, reduced_reference_pose_alignment_verified=True,
        historical_initial_and_stage_trace_sha_verified=True, saved_clips=len(saved), saved_states=len(curves),
        gradient_condition_rows=len(rows), foot_gradient_rows=len(feet), stationary_coverage_rows=len(coverage),
        direct_native_candidate_score_checks=direct_checks, gradient_zero_threshold_reward_per_rad=GRADIENT_EPSILON,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        goal_complete=False, user_accepted_checkpoint=None,
        limits=["Instantaneous pose reward gradients are not policy/action/H32/model-dynamics gradients or physical interventions.",
            "Only the same12 leg joint angles vary. Root pose, root/joint velocities, previous targets and measured force/slip are fixed.",
            "Root orientation reward is present, but has zero partial derivative in this root-fixed leg-angle comparison.",
            "Stationary gate is reference-conditioned; base training does not use the extra PPO stationary bonus.",
            "Native velocity reward still depends on leg pose through forward kinematics; these derivatives are retained.",
            "Zero/missing gradients retain undefined cosines and separate counts; they are not zero-angle alignment.",
            "Collision-sole minimum is piecewise differentiable; the native selected support-point branch is used at each saved pose.",
            "Opposing directions on saved outcomes do not isolate the cause of training failure or prove attainable foot unloading.",
            "Reference raised10mm and actual below5mm conditions describe saved data; they are not new training/acceptance gates."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "direct_native_candidate_score_checks")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "base", "ppo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
