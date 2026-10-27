"""Fixed-library physical evaluation, including native solved contact slip."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from gates import ratio
from resources import save

FIXED_DEMOS = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071",
               "triumph_two_handed_001__A058")


def learner_fingerprint(adapter):
    import warp as wp

    digest = hashlib.sha256()
    tensors = list(adapter.current.values()) + [adapter.history, adapter.episode_return,
               adapter.episode_normalized, adapter.episode_original, adapter.episode_steps]
    data = adapter.env.backend._device_data
    for key in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time"):
        if hasattr(data, key):
            tensors.append(wp.to_torch(getattr(data, key)))
    for value in tensors:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def preserve_learner(adapter):
    before = learner_fingerprint(adapter)
    cpu, cuda = torch.get_rng_state().clone(), [value.clone() for value in torch.cuda.get_rng_state_all()]
    with preserve_random_state():
        yield
    if learner_fingerprint(adapter) != before or not torch.equal(cpu, torch.get_rng_state()):
        raise AssertionError("Evaluation changed live learner state or CPU RNG")
    after = torch.cuda.get_rng_state_all()
    if len(cuda) != len(after) or any(not torch.equal(a, b) for a, b in zip(cuda, after)):
        raise AssertionError("Evaluation changed live learner CUDA RNG")


@contextmanager
def preserve_random_state():
    python, numpy = random.getstate(), np.random.get_state()
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)


class NativeContacts:
    """Observe the last solved substep before manager refresh/autoreset."""

    def __init__(self, env):
        import warp as wp

        self.env = env
        self.backend = env.backend
        self.original = self.backend.step_tensor
        self.wp = wp
        # The existing Q1 native diagnostic uses these backend-owned handles.
        self.model = self.backend._cpu_model
        self.data = self.backend._device_data
        self.gpu_model = self.backend._device_model
        model = self.model
        self.geom_body = torch.tensor(model.geom_bodyid, device=env.device)
        self.root_ids = torch.tensor(model.body_rootid, device=env.device)
        self.foot_owner = torch.full((model.nbody,), -1, dtype=torch.long, device=env.device)
        for side, prefix in enumerate(("left", "right")):
            roots = [i for i in range(model.nbody) if model.body(i).name.endswith(prefix + "_ankle_roll_link")]
            if len(roots) != 1:
                raise ValueError("Ambiguous native foot body")
            for body in range(model.nbody):
                parent = body
                while parent and parent != roots[0]:
                    parent = int(model.body_parentid[parent])
                if parent == roots[0]:
                    self.foot_owner[body] = side
        self.contact_ids = wp.array(np.arange(self.data.naconmax, dtype=np.int32), dtype=int, device=str(env.device))
        self.forces = wp.zeros(self.data.naconmax, dtype=wp.spatial_vector, device=str(env.device))
        self.last = None

    def install(self):
        import mujoco_warp as mw

        def step(ctrl, nsteps=1):
            outcome = self.original(ctrl, nsteps)
            wp, data = self.wp, self.data
            wp.synchronize()
            count = int(wp.to_torch(data.nacon)[0])
            preserved = {key: wp.to_torch(getattr(data, key)).clone()
                         for key in ("qpos", "qvel", "qacc_warmstart")}
            self.forces.zero_()
            mw.contact_force(self.gpu_model, data, self.contact_ids, True, self.forces)
            wp.synchronize()
            for key, before in preserved.items():
                if not torch.equal(before, wp.to_torch(getattr(data, key))):
                    raise AssertionError("Native contact observer changed " + key)
            worlds = wp.to_torch(data.contact.worldid)[:count].long()
            bodies = self.geom_body[wp.to_torch(data.contact.geom)[:count].long()]
            points = wp.to_torch(data.contact.pos)[:count]
            vertical = wp.to_torch(self.forces)[:count, 2].abs()
            cvel, com = wp.to_torch(data.cvel), wp.to_torch(data.subtree_com)
            normal = torch.zeros(self.env.num_envs, 2, device=self.env.device)
            weighted_speed = torch.zeros_like(normal)
            for side in range(2):
                first = (self.foot_owner[bodies[:, 0]] == side) & (bodies[:, 1] == 0)
                second = (self.foot_owner[bodies[:, 1]] == side) & (bodies[:, 0] == 0)
                keep = first | second
                world, point = worlds[keep], points[keep]
                body = torch.where(first[keep], bodies[keep, 0], bodies[keep, 1])
                velocity = cvel[world, body, 3:] + torch.linalg.cross(
                    cvel[world, body, :3], point - com[world, self.root_ids[body]])
                normal[:, side].index_add_(0, world, vertical[keep])
                weighted_speed[:, side].index_add_(0, world, vertical[keep] * velocity[:, :2].norm(dim=-1))
            self.last = normal.clone(), weighted_speed / normal.clamp(min=1e-8)
            return outcome

        self.backend.step_tensor = step
        return self

    def close(self):
        self.backend.step_tensor = self.original


def local_wrists(wrists, position, quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
    delta = wrists - position[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack((c * delta[..., 0] + s * delta[..., 1],
                     -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), -1)


@torch.no_grad()
def rollout(env, policy, progress=None):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, mat_to_quat
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale

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
        records = {key: [] for key in ("states", "frames", "alive", "targets", "force", "slip")}
        scale = torch.tensor([action_scale(name) for name in Q1_JOINT_NAMES], device=env.device)
        for iteration in range(int(motion.clip_len.max())):
            obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
            history = obs[:, None].repeat(1, 10, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            action = policy(history, mdp.goal(current["f"], 10)).clamp(-20, 20)
            targets = env._supermimic_command.default_q + current["xi"][:, 1:23] + scale * action
            state = env.step(action)
            tr = state.info[SUPERMIMIC_TRANSITION_KEY]
            alive = ~done
            values = dict(states=tr["s1"], frames=tr["f"] + 1, alive=alive, targets=targets,
                          force=contact.last[0], slip=contact.last[1])
            for key, value in values.items():
                records[key].append(value.cpu().numpy())
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
            raise AssertionError("Evaluation failed to reach every clip end")
        trace = {key: np.stack(values) for key, values in records.items()}
        trace.update(initial=initial, steps=steps.cpu().numpy(), failed=failed.cpu().numpy(), returns=returns.cpu().numpy())
        wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(side + "_wrist_yaw_link"))
                  for side in ("left", "right")]
        shape = trace["states"].shape[:2]
        flat = torch.from_numpy(trace["states"].reshape(-1, 57)).to(env.device)
        positions = []
        for chunk in flat.split(8192):
            positions.append(mdp.kin.forward(*unflatten_state(chunk))[0][:, wrists].cpu().numpy())
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


def clip_metrics(trace, index, count=None):
    count = int(trace["steps"][index]) if count is None else int(count)
    actual, ref_q = trace["states"][:count, index], trace["reference_q"][:count, index]
    ref_p = trace["reference_p"][:count, index]
    wrists = local_wrists(trace["wrists"][:count, index], actual[:, :3], actual[:, 3:7])
    ref_wrists = local_wrists(trace["reference_wrists"][:count, index], ref_p,
                             trace["reference_quat"][:count, index])
    loaded = trace["force"][:count, index] > 5.
    def rmse(value):
        return float(np.sqrt(np.mean(value**2)))
    target_delta = np.diff(trace["targets"][:count, index], axis=0)
    support = loaded[:, 0].astype(int) + 2 * loaded[:, 1].astype(int)
    single = support[(support == 1) | (support == 2)]
    return {"root_xy_rmse": rmse(actual[:, :2] - ref_p[:, :2]),
            "upper_rmse": rmse(actual[:, 27:35] - ref_q[:, 14:]),
            "leg_rmse": rmse(actual[:, 13:25] - ref_q[:, :12]),
            "wrist_relative_rmse": rmse(wrists - ref_wrists),
            "wrist_world_rmse": rmse(trace["wrists"][:count, index] - trace["reference_wrists"][:count, index]),
            "raised_peak_error": float(np.abs(wrists[..., 2].max(0) - ref_wrists[..., 2].max(0)).mean()),
            "actual_wrist_peak_height_m": wrists[..., 2].max(0).tolist(),
            "reference_wrist_peak_height_m": ref_wrists[..., 2].max(0).tolist(),
            "actual_wrist_peak_time_seconds": (wrists[..., 2].argmax(0) * .02).tolist(),
            "reference_wrist_peak_time_seconds": (ref_wrists[..., 2].argmax(0) * .02).tolist(),
            "target_oscillation": rmse(target_delta) if len(target_delta) else None,
            "loaded_slip": float(trace["slip"][:count, index][loaded].mean()) if loaded.any() else None,
            "single_support_fraction": float(((support == 1) | (support == 2)).mean()),
            "single_support_side_changes": int(np.count_nonzero(np.diff(single))),
            "double_support_fraction": float((support == 3).mean()),
            "steps": int(trace["steps"][index]), "reference_steps": int(trace["reference_steps"][index]),
            "common_prefix_steps": count, "failed": bool(trace["failed"][index]),
            "completion": float(trace["steps"][index] / trace["reference_steps"][index])}


def summarize(trace, rows):
    per_clip = [dict(name=row["name"], split=row["split"], group=row["group"], **clip_metrics(trace, i))
                for i, row in enumerate(rows)]
    summary = {}
    for split in ("train", "validation"):
        chosen = [r for r in per_clip if r["split"] == split]
        summary[split] = {"clips": len(chosen), "failures": sum(r["failed"] for r in chosen),
                          **{key: float(np.mean([r[key] for r in chosen]))
                             for key in ("completion", "root_xy_rmse", "upper_rmse", "leg_rmse", "wrist_relative_rmse")}}
    return {"summary": summary, "per_clip": per_clip,
            "contact_protocol": "Solved last 5ms substep before reset; world-Z force >5N, force-weighted tangential contact-point speed; not an integrated 20ms impulse.",
            "units": {"target_oscillation": "PD-target RMS change in radians per20ms", "loaded_slip": "m/s"}}


def compare(candidate, baseline, rows):
    common, base_common = [], []
    for i, row in enumerate(rows):
        if row["split"] == "validation":
            length = min(int(candidate["steps"][i]), int(baseline["steps"][i]))
            common.append(clip_metrics(candidate, i, length))
            base_common.append(clip_metrics(baseline, i, length))
    metrics = ("root_xy_rmse", "upper_rmse", "leg_rmse", "wrist_relative_rmse", "target_oscillation", "loaded_slip")
    measured = {}
    for key in metrics:
        values, bases = [r[key] for r in common], [r[key] for r in base_common]
        measured[key] = {"candidate": float(np.mean(values)) if all(v is not None for v in values) else None,
                         "baseline": float(np.mean(bases)) if all(v is not None for v in bases) else None}
    raised = next(i for i, row in enumerate(rows) if row["name"] == FIXED_DEMOS[2])
    measured["raised_peak_error"] = {"candidate": clip_metrics(candidate, raised)["raised_peak_error"],
                                     "baseline": clip_metrics(baseline, raised)["raised_peak_error"]}
    indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    demos = [i for i, row in enumerate(rows) if row["name"] in FIXED_DEMOS]
    if len(indices) != 30 or len(demos) != len(FIXED_DEMOS):
        raise AssertionError("Fixed RF12 validation/demo identity changed")
    return {"ratios": {key: ratio(value["candidate"], value["baseline"]) for key, value in measured.items()},
            "measurements": measured, "failures": int(candidate["failed"][indices].sum()),
            "completion": float(np.mean(candidate["steps"][indices] / candidate["reference_steps"][indices])),
            "fixed_demos_complete": bool((~candidate["failed"][demos]).all()
                                         and (candidate["steps"][demos] == candidate["reference_steps"][demos]).all()),
            "common_prefix_mean_seconds": float(np.mean([r["common_prefix_steps"] for r in common]) * .02)}
