"""E53 obs term: 6 root channels in the anchor heading frame (HP26 PC definition).

Channels (per env, all in the anchor heading frame):
  0:2  root displacement-so-far, xy + z, relative to the episode-start anchor position
  3:5  root linear velocity, finite difference of the anchor body position
Origin and previous position are cached on the env and refreshed at episode reset.
"""

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
    raise ValueError("E53 obs term: no TrackingCommand in command manager")


def _rotmat(quat_wxyz):
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    rows = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return rows.view(-1, 3, 3)


_E53_DBG_CALLS = 0


def e53_root_channels(env):
    global _E53_DBG_CALLS
    command = _tracking_command(env)
    pos = command.robot_anchor_pos_w
    rot = _rotmat(command.anchor_heading_quat)
    if _E53_DBG_CALLS < 3:
        _E53_DBG_CALLS += 1
        print("E53 DBG ch", pos.shape, rot.shape,
              "nff", getattr(command, "num_future_frames", None),
              "elh", getattr(env, "episode_length_buf", None) is not None and env.episode_length_buf.shape, flush=True)
    reset_now = bool((env.episode_length_buf == 0).any().item())
    origin = getattr(env, "_e53_origin", None)
    prev = getattr(env, "_e53_prev_pos", None)
    if origin is None or prev is None or reset_now:
        env._e53_origin = pos.detach().clone()
        env._e53_prev_pos = pos.detach().clone()
        vel = torch.zeros_like(pos)
    else:
        dt = float(getattr(env, "step_dt", 0.02))
        vel = (pos - prev) / max(dt, 1e-6)
        env._e53_prev_pos = pos.detach().clone()
    disp = torch.bmm(rot.transpose(1, 2), (pos - env._e53_origin).unsqueeze(-1)).squeeze(-1)
    vel_anchor = torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)
    return torch.cat([disp, vel_anchor], dim=-1)


try:  # sim-env only; plain-python control runs skip this import
    from isaaclab.utils import configclass

    from gear_sonic.envs.manager_env.mdp.observations import TokenizerCfg

    @configclass
    class E53TokenizerCfg(TokenizerCfg):
        """Tokenizer group plus the E53 6-channel term (registered via argv only)."""

        e53_root_channels = None
except Exception:  # pragma: no cover - no sim env in control runs
    E53TokenizerCfg = None


def e53_anchor_ori_b_mf_with_channels(env, command_name="motion", non_flatten=False):
    """Arm I g1 anchor term: original motion_anchor_ori_b block concatenated
    with the 6 E53 root channels (repeated across the 10 future slots). Slot
    layout per frame: [command_multi_future_nonflat 580 | anchor_ori 60+6 ] so
    the g1 first Linear auto-sizes 580+120=700 and the zero-padded weight
    columns [640:700] multiply exactly the injected channels."""

    from gear_sonic.envs.manager_env.mdp import observations as vendor_obs

    base = vendor_obs.motion_anchor_ori_b(env, command_name=command_name, non_flatten=non_flatten)
    channels = e53_root_channels(env)
    repeated = channels.unsqueeze(1).expand(-1, base.shape[1], -1)
    return torch.cat([base, repeated], dim=-1)
