"""Inspect reward sensitivity on saved RF16 states, without policy or physics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def position_z_replacement(error, ankle_error, sole_error, body_count):
    """Replace two Z components in the native mean squared position error."""
    error = np.asarray(error, dtype=np.float64)
    ankle_error, sole_error = (np.asarray(v, dtype=np.float64) for v in (ankle_error, sole_error))
    if (isinstance(body_count, bool) or not isinstance(body_count, int) or body_count < 2
            or ankle_error.shape != error.shape + (2,) or sole_error.shape != ankle_error.shape
            or any(not np.isfinite(v).all() for v in (error, ankle_error, sole_error))):
        raise ValueError("Expected a finite native mean and two aligned foot-Z errors")
    remaining = error - np.square(ankle_error).sum(-1) / body_count
    if (error < 0).any() or (remaining < -2e-7).any():
        raise ValueError("Foot components exceed the native mean error")
    return np.maximum(remaining, 0.) + np.square(sole_error).sum(-1) / body_count


def saved_clip(arrays, index, expected_steps):
    steps, reference = int(arrays["steps"][index]), int(arrays["reference_steps"][index])
    if not 1 < steps <= reference or steps != expected_steps:
        raise ValueError("Saved pre-reset clip duration disagrees with its evaluation JSON")
    result = {}
    widths = {"states": (57,), "targets": (22,), "force": (2,), "slip": (2,), "frames": ()}
    for name, width in widths.items():
        value = np.asarray(arrays[name])
        if value.ndim != 2 + len(width) or value.shape[2:] != width or steps > len(value):
            raise ValueError("Saved trace dimensions disagree: " + name)
        selected = value[:steps, index].copy()
        if not np.isfinite(selected).all():
            raise ValueError("Nonfinite saved trace: " + name)
        result[name] = selected
    frames = result["frames"]
    if frames.dtype.kind not in "iu" or not np.all(np.diff(frames) == 1):
        raise ValueError("Saved reference frames are not a contiguous pre-reset prefix")
    alive = np.asarray(arrays["alive"])
    if alive.ndim != 2 or alive.shape[:2] != arrays["states"].shape[:2] or not alive[:steps, index].all():
        raise ValueError("Saved prefix contains an inactive or reset suffix")
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(HERE), str(HERE.parent / "project_tools"),
                   str(HERE.parent / "supermimic_q1_contact_comp"),
                   str(args.runtime / "src"), str(args.runtime)]
    import torch
    from check_contract import identity
    from contact_reference import ContactReference, configure
    from prepare_q1_gait_review import FIXED, write_csv
    from tensor_geometry import TensorSole
    from sole_geometry import SoleGeometry
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib, CONTROL_DT
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import (
        default_joint_pos_list, Q1_JOINT_NAMES, action_scale)

    torch.set_num_threads(8)
    status_path = args.trial / "status.json"
    if json.loads(status_path.read_text())["status"] != "COMPLETED":
        raise ValueError("Read only completed RF16 evidence")
    library_path = args.trial / "evaluation/evaluation_library.json"
    stage_path = args.trial / "run/stage_decisions.json"
    evaluation_receipt = args.trial / "evaluation/receipt.json"
    library = json.loads(library_path.read_text())["splits"]["train"]
    if len(library) != 330 or len({r["name"] for r in library}) != 330:
        raise ValueError("Expected the original unique330 evaluation motions")
    stages = json.loads(stage_path.read_text())
    stage_traces = {row["path"]: row for stage in stages for row in stage["traces"]}
    initial = json.loads(evaluation_receipt.read_text())["initial_traces"]
    if initial != [str(args.trial / "evaluation" / f"initial_static075_repeat{i}.npz") for i in range(3)]:
        raise ValueError("Initial trace receipt paths differ")
    source_paths = [Path(__file__), args.scene, library_path, stage_path, status_path, evaluation_receipt,
                    HERE / "tensor_geometry.py", HERE / "sole_geometry.py",
                    HERE.parent / "supermimic_q1_contact_comp/contact_reference.py",
                    HERE.parent / "supermimic_q1_gait_comp/rewards.py"]
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    source_paths.extend(native / name for name in
                        ("mdp.py", "mdp_base.py", "candidate_options.json", "kinematics.py", "motion.py", "robot.py"))
    sources = [identity(p) for p in source_paths]
    kin = Q1Kinematics(args.scene, "cpu")
    motion = MotionLib([r["path"] for r in library], kin, "cpu", src_fps=30.)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()), observation_mode="root_xy")
    configure(library_path)
    phase = ContactReference(mdp)
    sole = TensorSole(SoleGeometry.from_scene(args.scene), kin.body_names, "cpu")
    foot_slots = [kin.tracked_idx.index(i) for i in sole.body_ids]
    indices = {name: next(i for i, r in enumerate(library) if r["name"] == name) for name in FIXED}
    scale = np.array([action_scale(name) for name in Q1_JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid native action scales")
    body_count, std, dt = len(kin.tracked_idx), mdp.cfg.body_pos_std, CONTROL_DT
    reference_sole = []
    with torch.no_grad():
        for start in range(0, motion.num_frames, 1024):
            frames = torch.arange(start, min(start + 1024, motion.num_frames))
            fk = kin.forward(*motion.ref_state(frames), all_bodies=True)
            reference_sole.append(sole(fk[0], fk[1]))
    reference_sole = torch.cat(reference_sole)
    rows, curves, trace_sources = [], [], []
    labels = ["initial_static075"] + [f"stage_{i:02d}" for i in range(1, len(stages) + 1)]
    for label in labels:
        for repeat in range(3):
            trace_path = args.trial / "evaluation" / f"{label}_repeat{repeat}.npz"
            trace_identity = identity(trace_path)
            if label != "initial_static075" and stage_traces.get(str(trace_path)) != trace_identity:
                raise ValueError("Saved stage trace differs from its historical SHA")
            json_path = trace_path.with_suffix(".json")
            result = json.loads(json_path.read_text())
            if result["repeat"] != repeat or len(result["per_clip"]) != 330:
                raise ValueError("Saved evaluation repeat or library differs")
            with np.load(trace_path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in
                          ("states", "targets", "force", "slip", "frames", "steps", "reference_steps", "alive")}
            with torch.no_grad():
                for name, index in indices.items():
                    clip = result["per_clip"][index]
                    if clip["name"] != name:
                        raise ValueError("Saved clip identity differs")
                    saved = saved_clip(arrays, index, clip["steps"])
                    frames = torch.from_numpy(saved["frames"]).long()
                    expected = motion.clip_start[index] + torch.arange(1, len(frames) + 1)
                    if not torch.equal(frames, expected):
                        raise ValueError("Saved global frame IDs differ from the original motion order")
                    flat = torch.from_numpy(saved["states"])
                    state = unflatten_state(flat)
                    zeros = torch.zeros(len(frames), 22)
                    _, terms = mdp.reward(state, frames, zeros, zeros, terms=True)
                    fk = kin.forward(*state, all_bodies=True)
                    measured_sole = sole(fk[0], fk[1])
                    ankle_error = (fk[0][:, sole.body_ids, 2] - motion.body_p[frames][:, foot_slots, 2]).numpy()
                    sole_error = (measured_sole - reference_sole[frames]).numpy()
                    e_body = terms["e_body_pos"].numpy()
                    replacement = position_z_replacement(e_body, ankle_error, sole_error, body_count)
                    original_score = dt * np.exp(-e_body / std**2)
                    sole_score = dt * np.exp(-replacement / std**2)
                    perfect_sole_score = dt * np.exp(-np.maximum(
                        replacement - np.square(sole_error).sum(-1) / body_count, 0.) / std**2)
                    base_gain = perfect_sole_score - sole_score
                    _, _, _, stance_t, swing_t, _ = phase.targets(frames)
                    stance, swing = stance_t.numpy(), swing_t.numpy()
                    counts = np.maximum(swing.sum(-1), 1)
                    ankle_z_score = np.exp(-np.square(ankle_error / .04))
                    sole_z_score = np.exp(-np.square(sole_error / .04))
                    ppo_change = .01 * ((sole_z_score - ankle_z_score) * swing).sum(-1) / counts
                    ppo_gain = .01 * ((1 - sole_z_score) * swing).sum(-1) / counts
                    loaded = saved["force"] > 5.
                    target_delta = np.diff(saved["targets"], axis=0)
                    base_action_cost = -mdp.cfg.w_action_rate * dt * np.square(target_delta / scale).sum(-1)
                    ppo_smooth_cost = .002 * np.minimum(np.square(target_delta / .1).mean(-1), 25.)
                    equivalent = np.sqrt(base_gain / (-mdp.cfg.w_action_rate * dt * np.square(1 / scale[:12]).sum()))
                    active_swing = swing.any(-1)
                    def conditional_mean(value, mask):
                        return float(value[mask].mean()) if mask.any() else None
                    row = dict(label=label, repeat=repeat, clip=name,
                        training_seconds=result["training_seconds"], steps=len(frames), failed=clip["failed"],
                        native_body_position_score_mean=float(original_score.mean()),
                        native_tracking_sum_mean=float(terms["track"].mean()),
                        base_sole_substitution_mean=float((sole_score-original_score).mean()),
                        base_perfect_sole_z_gain_mean=float(base_gain.mean()),
                        base_perfect_sole_z_gain_p95=float(np.quantile(base_gain, .95)),
                        base_equivalent_simultaneous12leg_target_change_median_rad=float(np.median(equivalent)),
                        base_saved_action_rate_cost_mean=float(base_action_cost.mean()),
                        ppo_saved_smooth_cost_mean=float(ppo_smooth_cost.mean()),
                        ppo_sole_substitution_mean=float(ppo_change.mean()),
                        ppo_perfect_swing_z_gain_mean=float(ppo_gain.mean()),
                        ppo_perfect_swing_z_gain_on_swing_frames=conditional_mean(ppo_gain, active_swing),
                        reference_swing_foot_frames=int(swing.sum()),
                        reference_swing_loaded_fraction=conditional_mean(loaded, swing),
                        reference_swing_below5mm_fraction=conditional_mean(measured_sole.numpy() <= .005, swing),
                        reference_stance_unloaded_fraction=conditional_mean(~loaded, stance))
                    rows.append(row)
                    curves.extend(dict(label=label, repeat=repeat, clip=name, seconds=(i+1)*dt,
                        base_perfect_sole_z_gain=float(base_gain[i]), ppo_perfect_swing_z_gain=float(ppo_gain[i]),
                        base_sole_substitution=float(sole_score[i]-original_score[i]),
                        ppo_sole_substitution=float(ppo_change[i]),
                        base_saved_action_rate_cost=None if i == 0 else float(base_action_cost[i-1]),
                        ppo_saved_smooth_cost=None if i == 0 else float(ppo_smooth_cost[i-1])) for i in range(len(frames)))
            if identity(trace_path) != trace_identity:
                raise ValueError("Saved physical source changed during analysis")
            trace_sources.extend((trace_identity, identity(json_path)))
            print(json.dumps(dict(label=label, repeat=repeat, saved_state_only=True)), flush=True)
    if sources != [identity(p) for p in source_paths]:
        raise ValueError("Analysis or native sources changed")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "fixed_reward_signal.csv", rows)
    write_csv(args.output / "fixed_reward_signal_timeseries.csv", curves)
    value = dict(status="SAVED_REWARD_SIGNAL_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        sources=sources, trace_sources=trace_sources, contact_contract=phase.contract,
        fixed_rows=len(rows), timeseries_rows=len(curves), labels=labels,
        native_body_count=body_count, body_position_std_m=std, control_dt=dt,
        action_scales=scale.tolist(), action_rate_weight=mdp.cfg.w_action_rate,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        limitations=["Read-only saved states and analytic reward counterfactuals; no feasible physical correction or action is proven.",
            "Perfect sole-Z gain holds all other body errors fixed; no claim that a realizable motion has this gain.",
            "Equivalent target delta compares only position reward and native action-rate penalty, not velocity/orientation/critic/future rewards.",
            "Saved action-rate and smooth costs exclude the first control transition because its preceding target was not stored.",
            "Declared contact phases are retarget intentions; saved force is the last5ms substep, not a20ms impulse.",
            "Stage traces have historical SHA; initial traces have path/alignment and current before/after SHA, without an initial historical SHA.",
            "Different stages are saved policy outcomes, not new matched interventions or proof of gait benefit."])
    (args.output / "summary.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:value[k] for k in ("status", "fixed_rows", "timeseries_rows")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "trial", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
