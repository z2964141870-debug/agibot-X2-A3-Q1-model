"""Build e54_channel_stats.json from E53 N8 rollout reference arrays (offline, no learning params).

Uses the same math as the training-time obs term (heading-rotation displacement-so-far
+ finite-difference velocity, dt=0.02) applied to each motion's REFERENCE anchor
trajectory (ref_anchor_pos_w / ref_anchor_quat_wxyz from the E53 N8 eval timeseries,
which are projections of the same MotionLib data used at train time).
"""

import json
import numpy as np
from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
E53 = Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_rootchannel_20261019_E53")
OUT = OPS / "data/experiments/a3_zhy_channel_probe_20261019_E54"
DT = 0.02


def heading_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rows = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return rows.reshape(-1, 3, 3)


def rot_transpose_vec(R, v):
    return np.einsum("tji,tj->ti", R, v)


def stats_from_trace(ts):
    p = np.asarray(ts["ref_anchor_pos_w"], dtype=float)
    q = np.asarray(ts["ref_anchor_quat_wxyz"], dtype=float)
    R = heading_rot(q)
    disp = rot_transpose_vec(R, p - p[0])
    vel_raw = np.zeros_like(p)
    vel_raw[1:] = (p[1:] - p[:-1]) / DT
    vel = rot_transpose_vec(R, vel_raw)
    X = np.concatenate([disp, vel], axis=1)
    return dict(mu=[float(v) for v in X.mean(axis=0)],
                sigma=[float(v) for v in X.std(axis=0)],
                n_frames=int(len(X)), motion=ts["motion_name"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for kind in ("cloth", "official"):
        directory = E53 / ("eval_arm_n_tier8_" + kind)
        for f in sorted(directory.glob("*.timeseries.json")):
            ts = json.loads(f.read_text())["motions"][0]
            name = ts["motion_name"]
            manifest[name] = stats_from_trace(ts)
            manifest[name]["source_file"] = str(f)
    if len(manifest) != 10:
        raise SystemExit("E54 stats: expected 10 motions, got %d" % len(manifest))
    path = OUT / "e54_channel_stats.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + chr(10))
    print("E54 STATS OK", len(manifest), sorted(manifest))


if __name__ == "__main__":
    main()
