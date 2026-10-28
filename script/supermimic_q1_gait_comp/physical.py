"""Fixed330-clip evaluation and common-prefix comparison for the gait head."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_balance"),
               str(HERE.parent / "supermimic_q1_gait"), str(HERE.parent / "supermimic_q1_ppo")]
from evaluation import NativeContacts, FIXED_DEMOS, summarize, compare, clip_metrics, preserve_random_state
from balance_evaluation import preserve_balance
from control import Fallback, inputs, compose, risk_features
from rewards import FootReference
from metrics import enrich, gait_clip, gait_summary
from decisions import relative


@torch.no_grad()
def rollout(env, teacher, head, thresholds, foot, mode="trained075", progress=None):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, mat_to_quat
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
    if mode not in ("teacher", "static075", "trained075"):
        raise ValueError(mode)
    mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
    contact = NativeContacts(env).install()
    try:
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current = info[SUPERMIMIC_STATE_KEY]
        initial = current["s"].cpu().numpy()
        history = None
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        failed = done.clone()
        steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        returns = torch.zeros(env.num_envs, device=env.device)
        records = {key: [] for key in ("states", "frames", "alive", "targets", "force", "slip", "alpha", "residual")}
        scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=env.device)
        gate = Fallback(env.num_envs, thresholds, env.device)
        cap = torch.full((env.num_envs,), 0. if mode == "teacher" else .75, device=env.device)
        gate.reset(~done, cap)
        for iteration in range(int(motion.clip_len.max())):
            measured = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
            history = measured[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], measured[:, None]), 1)
            base = env._supermimic_command.default_q + current["xi"][:, 1:23]
            next_frames = torch.minimum(current["f"]+1, motion.clip_end[motion.clip_id[current["f"]]])
            features, nominal, reference = inputs(teacher, history, mdp.goal(current["f"], 10),
                motion.q[next_frames], base, scale, gate.alpha, gate.filtered, gate.cooldown, cap)
            features = torch.cat((features, foot.features(current["s"], current["f"])), -1)
            latent = head(features) if mode == "trained075" else torch.zeros(env.num_envs, 14, device=env.device)
            action, residual = compose(nominal, reference, latent, gate.alpha, scale)
            targets = base + scale * action
            alpha = gate.alpha.clone()
            state = env.step(action)
            tr = state.info[SUPERMIMIC_TRANSITION_KEY]
            alive = ~done
            values = dict(states=tr["s1"], frames=tr["f"]+1, alive=alive, targets=targets,
                          force=contact.last[0], slip=contact.last[1], alpha=alpha, residual=residual)
            for key, value in values.items():
                records[key].append(value.cpu().numpy())
            risk = risk_features(tr["s1"], motion.root_p[tr["f"]+1], mat_to_quat(motion.root_R[tr["f"]+1]), *contact.last)
            gate.advance(risk, cap)
            gate.alpha = cap.clone()
            gate.reset(tr["done"].bool(), cap)
            returns += torch.where(alive, tr["r"], 0)
            steps += alive
            failed |= alive & tr["term"]
            done |= alive & tr["done"]
            current = state.info[SUPERMIMIC_STATE_KEY]
            if progress and iteration % 50 == 0:
                progress(iteration)
            if bool(done.all()):
                break
        if not bool(done.all()):
            raise AssertionError("Physical evaluation did not terminate every clip")
        trace = {k:np.stack(v) for k,v in records.items()}
        trace.update(initial=initial, steps=steps.cpu().numpy(), failed=failed.cpu().numpy(), returns=returns.cpu().numpy())
        wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(side+"_wrist_yaw_link")) for side in ("left", "right")]
        shape = trace["states"].shape[:2]
        flat = torch.from_numpy(trace["states"].reshape(-1, 57)).to(env.device)
        points = [mdp.kin.forward(*unflatten_state(block))[0][:, wrists].cpu().numpy() for block in flat.split(8192)]
        trace["wrists"] = np.concatenate(points).reshape(*shape, 2, 3)
        frames = torch.from_numpy(trace["frames"]).to(env.device)
        trace.update(reference_q=motion.q[frames].cpu().numpy(), reference_p=motion.root_p[frames].cpu().numpy(),
                     reference_quat=mat_to_quat(motion.root_R[frames]).cpu().numpy(),
                     reference_wrists=motion.body_p[frames][..., wrists, :].cpu().numpy(),
                     reference_steps=(motion.clip_len-1).cpu().numpy())
        return enrich(trace, mdp)
    finally:
        contact.close()


def comparison(candidate, baseline, rows):
    old = compare(candidate, baseline, rows)
    walked, bases = [], []
    for name in FIXED_DEMOS[:2]:
        index = next(i for i,r in enumerate(rows) if r["name"] == name)
        length = min(int(candidate["steps"][index]), int(baseline["steps"][index]))
        walked.append(gait_clip(candidate, index, length))
        bases.append(gait_clip(baseline, index, length))
    keys = {"walking_slip": "loaded_slip_mps", "swing_error": "swing_local_foot_rmse_m",
            "walking_tilt": "torso_tilt_rms_rad", "target_change": "target_delta_rms_rad"}
    ratios = {}
    for label, key in keys.items():
        values, previous = [r[key] for r in walked], [r[key] for r in bases]
        ratios[label] = relative(float(np.mean(values)), float(np.mean(previous))) if all(v is not None for v in values+previous) else None
    ratios.update(upper_rmse=old["ratios"]["upper_rmse"], wrist_rmse=old["ratios"]["wrist_relative_rmse"],
                  raised_peak_error=old["ratios"]["raised_peak_error"])
    raised = next(i for i,r in enumerate(rows) if r["name"] == FIXED_DEMOS[2])
    return {"ratios": ratios, "validation_failures": old["failures"], "validation_completion": old["completion"],
            "fixed_demos_complete": old["fixed_demos_complete"],
            "walking_progress_preserved": all(v["forward_displacement_m"] >= max(.05, .75*b["forward_displacement_m"])
                                              for v,b in zip(walked, bases)),
            "stationary_drift_m": gait_clip(candidate, raised)["root_displacement_m"],
            "baseline_stationary_drift_m": gait_clip(baseline, raised)["root_displacement_m"],
            "walking_common_prefix": walked, "baseline_walking_common_prefix": bases,
            "validation_tracking_diagnostic": old}
