"""E55 channel math, single source for the MuJoCo entry side.

Formula mirrors e54_obs_terms._raw_channels exactly (fixture-pinned parity):
disp = R^T (pos - origin); vel = R^T (pos - prev)/dt; channels = [disp, vel]
with R from the anchor heading quaternion (wxyz). Origin/prev bookkeeping is
the caller's (the entry keeps episode state; the fixture pins parity).
"""

import torch


def _rotmat(quat_wxyz):
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    rows = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return rows.view(*quat_wxyz.shape[:-1], 3, 3)


def entry_channels(pos, prev_pos, heading_now, heading_prev, dt, origin, prev):
    """pos/prev_pos/origin/prev: (B,3) world anchor positions; heading: (B,4) wxyz."""
    rot = _rotmat(heading_now)
    vel = (pos - prev) / max(float(dt), 1e-6)
    disp = torch.bmm(rot.transpose(1, 2), (pos - origin).unsqueeze(-1)).squeeze(-1)
    vel_anchor = torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)
    return torch.cat([disp, vel_anchor], dim=-1)
