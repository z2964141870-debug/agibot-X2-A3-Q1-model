#!/usr/bin/env python3
"""Convert named A3 CSV joints into a pinned 50 Hz full-state motion library."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic.a3_assets import A3Assets, TRACKED, identity


def convert(assets, path, output):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    n = len(rows)
    if n < 30:
        raise ValueError(f"Motion too short: {path}")
    required = ["root_translate" + c for c in "XYZ"] + ["root_rotate" + c for c in "XYZ"]
    required += assets.active_names
    if not set(required) <= set(rows[0]):
        raise ValueError(f"Missing named CSV fields in {path}")
    pos = np.array([[float(row["root_translate" + c]) for c in "XYZ"] for row in rows]) * .01
    euler = np.array([[float(row["root_rotate" + c]) for c in "XYZ"] for row in rows])
    angles = np.deg2rad(np.array([[float(row[name]) for name in assets.active_names] for row in rows]))
    source_t = np.arange(n) / 30.
    times = np.arange(int(np.floor(source_t[-1] * 50)) + 1) / 50.
    qpos = np.tile(assets.model.qpos0, (len(times), 1))
    qpos[:, :3] = np.column_stack([np.interp(times, source_t, pos[:, i]) for i in range(3)])
    quats = Slerp(source_t, Rotation.from_euler("xyz", euler, degrees=True))(times).as_quat()
    qpos[:, 3:7] = quats[:, [3, 0, 1, 2]]
    qpos[:, assets.qadr] = np.column_stack([np.interp(times, source_t, angles[:, i]) for i in range(29)])
    qpos[:, assets.model.jnt_qposadr[assets.head_jids]] = 0.
    qvel = np.empty((len(times), assets.model.nv))
    for i in range(len(times) - 1):
        mujoco.mj_differentiatePos(assets.model, qvel[i], .02, qpos[i], qpos[i + 1])
    qvel[-1] = qvel[-2]
    d = mujoco.MjData(assets.model)
    body = {key: [] for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    heights = []
    # Reference floor contact uses the actual mesh support function, not a body origin.
    mesh_vertices = {}
    for gid in assets.foot_geoms:
        if assets.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Register support geometry for any non-mesh foot")
        mid = assets.model.geom_dataid[gid]
        start, count = assets.model.mesh_vertadr[mid], assets.model.mesh_vertnum[mid]
        mesh_vertices[gid] = assets.model.mesh_vert[start:start + count].copy()
    for i in range(len(times)):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        mujoco.mj_forward(assets.model, d)
        body["body_pos_w"].append(d.xpos[assets.body_ids].copy())
        body["body_quat_w"].append(d.xquat[assets.body_ids].copy())
        velocities = []
        for bid in assets.body_ids:
            vel = np.empty(6)
            mujoco.mj_objectVelocity(assets.model, d, mujoco.mjtObj.mjOBJ_XBODY, int(bid), vel, 0)
            velocities.append(vel)
        velocities = np.asarray(velocities)
        body["body_lin_vel_w"].append(velocities[:, 3:])
        body["body_ang_vel_w"].append(velocities[:, :3])
        heights.append(min(float((vertices @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid])[:, 2].min())
                           for gid, vertices in mesh_vertices.items()))
    heights = np.asarray(heights)
    limit_excess = np.maximum(assets.model.jnt_range[assets.jids, 0] - qpos[:, assets.qadr], 0)
    limit_excess += np.maximum(qpos[:, assets.qadr] - assets.model.jnt_range[assets.jids, 1], 0)
    receipt = {"source": identity(path), "frames": len(times), "source_hz": 30, "target_hz": 50,
               "foot_min_height_m": float(heights.min()), "foot_height_p05_m": float(np.quantile(heights, .05)),
               "joint_limit_excess_max_rad": float(limit_excess.max()),
               "joint_velocity_max_rad_s": float(np.abs(qvel[:, assets.vadr]).max()),
               "transform": "cm/degrees to m/radians; xyz Euler; named active mapping; Slerp; no floor or limit correction"}
    np.savez_compressed(output, qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
                        joint_pos=qpos[:, 7:].astype(np.float32), joint_vel=qvel[:, 6:].astype(np.float32),
                        body_names=np.array(TRACKED), fps=np.array(50.),
                        **{k: np.asarray(v, dtype=np.float32) for k, v in body.items()})
    receipt["output"] = identity(output)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assets = A3Assets(args.root)
    sources = sorted((assets.vendor / "a3_data/agibot_a3").glob("*.csv"))
    results = [convert(assets, file, args.output / (file.stem + ".npz")) for file in sources]
    with (args.output / "manifest.json").open("x") as stream:
        json.dump({"status": "CONVERTED_NOT_PHYSICS_VALIDATED", "assets": assets.manifest(), "motions": results}, stream, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
