"""Deterministic physical evaluation of frozen teacher and residual candidates."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from evaluation import NativeContacts, summarize, compare, clip_metrics, FIXED_DEMOS, preserve_learner, preserve_random_state
from control import Fallback, MAX_ALPHA, inputs, compose, risk_features


def extra_fingerprint(adapter):
    digest = hashlib.sha256()
    tensors = list(adapter.gate.state_tensors()) + list(adapter.observations.values())
    tensors += [adapter.keep_baseline, adapter.nominal, adapter.reference_action]
    if adapter.contacts.last is not None:
        tensors += list(adapter.contacts.last)
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(str((adapter.cap, adapter.fallback_enabled)).encode())
    return digest.hexdigest()


@contextmanager
def preserve_balance(adapter):
    before = extra_fingerprint(adapter)
    with preserve_learner(adapter):
        yield
    if before != extra_fingerprint(adapter):
        raise AssertionError("Evaluation changed live gate, features, contacts or curriculum")


@torch.no_grad()
def rollout(env, teacher, head, thresholds, mode="gated", progress=None):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, mat_to_quat
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale

    if mode not in ("gated", "trained_static075", "static075", "teacher"):
        raise ValueError(mode)
    motion, mdp = env.supermimic_mdp.motion, env.supermimic_mdp
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
        scale = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES], device=env.device)
        gate = Fallback(env.num_envs, thresholds, env.device)
        cap = torch.full((env.num_envs,), 0. if mode == "teacher" else MAX_ALPHA, device=env.device)
        gate.reset(~done, cap)
        for iteration in range(int(motion.clip_len.max())):
            obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
            history = obs[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            base = env._supermimic_command.default_q + current["xi"][:, 1:23]
            next_frames = torch.minimum(current["f"] + 1, motion.clip_end[motion.clip_id[current["f"]]])
            features, nominal, reference = inputs(teacher, history, mdp.goal(current["f"], 10),
                                                  motion.q[next_frames], base, scale, gate.alpha, gate.filtered,
                                                  gate.cooldown, cap)
            latent = head(features) if mode in ("gated", "trained_static075") else torch.zeros(env.num_envs, 14, device=env.device)
            action, residual = compose(nominal, reference, latent, gate.alpha, scale)
            applied_alpha = gate.alpha.clone()
            targets = base + scale * action
            state = env.step(action)
            tr = state.info[SUPERMIMIC_TRANSITION_KEY]
            alive = ~done
            for key, value in dict(states=tr["s1"], frames=tr["f"] + 1, alive=alive, targets=targets,
                                   force=contact.last[0], slip=contact.last[1], alpha=applied_alpha,
                                   residual=residual).items():
                records[key].append(value.cpu().numpy())
            if mode in ("gated", "trained_static075"):
                risk = risk_features(tr["s1"], motion.root_p[tr["f"] + 1],
                                     mat_to_quat(motion.root_R[tr["f"] + 1]), *contact.last)
                gate.advance(risk, cap)
                if mode == "trained_static075":
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
            raise AssertionError("Every evaluated clip must reach a terminal condition")
        trace = {key: np.stack(values) for key, values in records.items()}
        trace.update(initial=initial, steps=steps.cpu().numpy(), failed=failed.cpu().numpy(), returns=returns.cpu().numpy())
        wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(side + "_wrist_yaw_link")) for side in ("left", "right")]
        shape = trace["states"].shape[:2]
        flat = torch.from_numpy(trace["states"].reshape(-1, 57)).to(env.device)
        positions = [mdp.kin.forward(*unflatten_state(chunk))[0][:, wrists].cpu().numpy() for chunk in flat.split(8192)]
        trace["wrists"] = np.concatenate(positions).reshape(*shape, 2, 3)
        frames = torch.from_numpy(trace["frames"]).to(env.device)
        trace["reference_q"] = motion.q[frames].cpu().numpy()
        trace["reference_p"] = motion.root_p[frames].cpu().numpy()
        trace["reference_quat"] = mat_to_quat(motion.root_R[frames]).cpu().numpy()
        trace["reference_wrists"] = motion.body_p[frames][..., wrists, :].cpu().numpy()
        trace["reference_steps"] = (motion.clip_len - 1).cpu().numpy()
        return trace
    finally:
        contact.close()


def summarize_balance(trace, rows):
    result = summarize(trace, rows)
    active = trace["alive"].astype(bool)
    result["intervention"] = {"mean_alpha": float(trace["alpha"][active].mean()),
        "fallback_fraction": float((trace["alpha"][active] < MAX_ALPHA - 1e-5).mean()),
        "residual_rms_rad": float(np.sqrt(np.mean(trace["residual"][active] ** 2))),
        "residual_max_rad": float(np.abs(trace["residual"][active]).max())}
    result["fixed_demos"] = [dict(name=row["name"], **clip_metrics(trace, i))
                              for i, row in enumerate(rows) if row["name"] in FIXED_DEMOS]
    return result
