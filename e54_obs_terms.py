"""E54 obs terms: two 6-channel feature families replacing the E53-I channels.

P (standardized): clip((c_I - mu_m) / sigma_m, -5, +5) with per-motion offline
    reference statistics from e54_channel_stats.json (no learned parameters).
Q (error form):   [disp_a(ref) - disp_a(sim), vel_a(ref) - vel_a(sim)] in the
    reference anchor heading frame; zero at perfect tracking.

Geometry is untouched E53 v19: vendor anchor block + 6 channels repeated over
the 10 future slots; the 035 checkpoint zero-column gating provides the t=0
identity (E11 anchor check receipt reused).
"""

import json
import os

import torch

TRACKING_COMMAND_CLASS = None


def _tracking_command(env):
    global TRACKING_COMMAND_CLASS
    if TRACKING_COMMAND_CLASS is None:
        from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
        TRACKING_COMMAND_CLASS = TrackingCommand
    terms = getattr(env.command_manager, "_terms", {})
    for name in sorted(terms):
        if isinstance(terms[name], TRACKING_COMMAND_CLASS):
            return terms[name]
    raise ValueError("E54 obs term: no TrackingCommand in command manager")


def _rotmat(quat_wxyz):
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    rows = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return rows.view(-1, 3, 3)


def _raw_channels(command, env):
    """E53-I raw channels: displacement-so-far + velocity of the robot anchor in the reference heading frame."""
    pos = command.robot_anchor_pos_w
    rot = _rotmat(command.anchor_heading_quat)
    reset_now = bool((env.episode_length_buf == 0).any().item())
    origin = getattr(env, "_e54_origin_sim", None)
    prev = getattr(env, "_e54_prev_sim", None)
    if origin is None or prev is None or reset_now:
        env._e54_origin_sim = pos.detach().clone()
        env._e54_prev_sim = pos.detach().clone()
        vel = torch.zeros_like(pos)
    else:
        dt = float(getattr(env, "step_dt", 0.02))
        vel = (pos - prev) / max(dt, 1e-6)
        env._e54_prev_sim = pos.detach().clone()
    disp = torch.bmm(rot.transpose(1, 2), (pos - env._e54_origin_sim).unsqueeze(-1)).squeeze(-1)
    vel_anchor = torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)
    return torch.cat([disp, vel_anchor], dim=-1)


_MANIFEST = None


def _manifest():
    global _MANIFEST
    if _MANIFEST is None:
        path = os.environ.get("E54_CHANNEL_STATS")
        if not path:
            raise ValueError("E54 P arm: E54_CHANNEL_STATS env var is missing")
        _MANIFEST = json.loads(open(path).read())
        if not _MANIFEST:
            raise ValueError("E54 P arm: empty stats manifest")
    return _MANIFEST


_TENSOR_CACHE = {}


def _stat_tensors(device):
    if not _TENSOR_CACHE:
        for key, row in _manifest().items():
            mu = torch.tensor([float(v) for v in row["mu"]], dtype=torch.float32, device=device)
            sigma = torch.tensor([max(float(v), 1e-6) for v in row["sigma"]],
                                 dtype=torch.float32, device=device)
            _TENSOR_CACHE[key] = (mu, sigma)
    return _TENSOR_CACHE


def _stem(name):
    text = str(name)
    text = text[:-4] if text.endswith(".csv") else text
    return os.path.splitext(os.path.basename(text))[0] if "/" in text or text != os.path.basename(text) else text


def _resolve_stats(name, cache):
    text = str(name)
    if text in cache:
        return cache[text]
    stem = _stem(text)
    if stem in cache:
        return cache[stem]
    raise ValueError("E54 P arm: motion stats missing for " + repr(text))


def e54_root_channels_standardized(env):
    command = _tracking_command(env)
    raw = _raw_channels(command, env)
    cache = _stat_tensors(raw.device)
    keys = [_stem(k) for k in command.motion_lib._motion_data_keys]  # noqa: SLF001 - read-only name order
    ids = command.motion_ids.tolist() if hasattr(command.motion_ids, "tolist") else list(command.motion_ids)
    mus, sigmas = [], []
    for i in ids:
        mu, sigma = _resolve_stats(keys[int(i)], cache)
        mus.append(mu)
        sigmas.append(sigma)
    z = (raw - torch.stack(mus)) / torch.stack(sigmas)
    return torch.clamp(z, -5.0, 5.0)


def e54_root_channels_errorform(env):
    command = _tracking_command(env)
    rot = _rotmat(command.anchor_heading_quat)
    ref = command.anchor_pos_w
    sim = command.robot_anchor_pos_w
    reset_now = bool((env.episode_length_buf == 0).any().item())
    origin_ref = getattr(env, "_e54_origin_ref", None)
    prev_ref = getattr(env, "_e54_prev_ref", None)
    origin_sim = getattr(env, "_e54_origin_sim", None)
    prev_sim = getattr(env, "_e54_prev_sim", None)
    if reset_now or None in (origin_ref, prev_ref, origin_sim, prev_sim):
        env._e54_origin_ref = ref.detach().clone()
        env._e54_prev_ref = ref.detach().clone()
        env._e54_origin_sim = sim.detach().clone()
        env._e54_prev_sim = sim.detach().clone()
        return torch.zeros_like(sim)
    dt = max(float(getattr(env, "step_dt", 0.02)), 1e-6)
    rt = rot.transpose(1, 2)
    err_disp = torch.bmm(rt, ((ref - origin_ref) - (sim - origin_sim)).unsqueeze(-1)).squeeze(-1)
    err_vel = torch.bmm(rt, (((ref - prev_ref) - (sim - prev_sim)) / dt).unsqueeze(-1)).squeeze(-1)
    env._e54_prev_ref = ref.detach().clone()
    env._e54_prev_sim = sim.detach().clone()
    return torch.cat([err_disp, err_vel], dim=-1)


try:  # sim-env only; plain-python control runs skip this import
    from isaaclab.utils import configclass

    from gear_sonic.envs.manager_env.mdp.observations import TokenizerCfg

    @configclass
    class E54TokenizerCfg(TokenizerCfg):
        """Tokenizer group plus the E54 6-channel terms (registered via argv only)."""

        e54_root_channels_standardized = None
        e54_root_channels_errorform = None
except Exception:  # pragma: no cover
    E54TokenizerCfg = None


def _with_channels(env, channels, command_name="motion", non_flatten=False):
    from gear_sonic.envs.manager_env.mdp import observations as vendor_obs

    base = vendor_obs.motion_anchor_ori_b(env, command_name=command_name, non_flatten=non_flatten)
    repeated = channels.unsqueeze(1).expand(-1, base.shape[1], -1)
    return torch.cat([base, repeated], dim=-1)


def e54_anchor_ori_b_mf_standardized(env, command_name="motion", non_flatten=False):
    return _with_channels(env, e54_root_channels_standardized(env), command_name, non_flatten)


def e54_anchor_ori_b_mf_errorform(env, command_name="motion", non_flatten=False):
    return _with_channels(env, e54_root_channels_errorform(env), command_name, non_flatten)
