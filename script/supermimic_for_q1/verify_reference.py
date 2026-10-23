#!/usr/bin/env python3
"""Recompute Q1 reference timing, velocities, FK and geometric quality gates."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.q1_assets import Q1Assets, TRACKED, identity


def lowest_foot(model, data, geom_ids):
    lows = []
    for gid in geom_ids:
        matrix = data.geom_xmat[gid].reshape(3, 3)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            lows.append(float((vertices @ matrix[2] + data.geom_xpos[gid, 2]).min()))
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            lows.append(float(data.geom_xpos[gid, 2] - np.sum(np.abs(matrix[2]) * model.geom_size[gid])))
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            lows.append(float(data.geom_xpos[gid, 2] - model.geom_size[gid, 0]))
        else:
            raise ValueError("Unsupported foot collision geometry")
    return min(lows)


def check_clip(assets, row, mapping):
    result = {"clip": row["clip"], "qualified_for_training_pilot": False}
    if "output" not in row:
        result["source_conversion_error"] = row.get("error")
        return result
    file = Path(row["output"]["path"])
    trace_file = Path(row["kinematic_trace"]["path"])
    for path, expected in ((file, row["output"]), (trace_file, row["kinematic_trace"])):
        if identity(path)["sha256"] != expected["sha256"]:
            raise ValueError(f"Input changed: {path}")
    model, data = assets.model, mujoco.MjData(assets.model)
    with np.load(file, allow_pickle=False) as saved, np.load(trace_file, allow_pickle=False) as trace:
        qpos, qvel = saved["qpos"].astype(float), saved["qvel"].astype(float)
        times = saved["time_s"]
        if qpos.shape != (len(times), 29) or qvel.shape != (len(times), 28) or len(times) < 2:
            raise ValueError("Wrong Q1 state dimensions")
        if saved["body_names"].tolist() != list(TRACKED) or float(saved["fps"]) != 50.:
            raise ValueError("Body order or frequency changed")
        finite = all(np.isfinite(saved[key]).all() for key in saved.files if saved[key].dtype.kind in "fiu")
        velocity = np.empty_like(qvel)
        for index in range(len(times) - 1):
            mujoco.mj_differentiatePos(model, velocity[index], .02, qpos[index], qpos[index + 1])
        velocity[-1] = velocity[-2]
        trace_ids = [model.body(mapping[n]).id for n in trace["source_body_names"].tolist()]
        keys = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        fk_error = {key: 0. for key in keys}
        residuals, feet = [], []
        shift = row["diagnostics"]["global_ground_origin_shift_m"]
        before = trace["qpos_before_ground_origin"].copy()
        before[:, 2] += shift
        state_difference = float(np.abs(before - qpos).max())
        for frame, (q, v) in enumerate(zip(qpos, qvel)):
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(model, data)
            positions = data.xpos[assets.body_ids]
            quats = data.xquat[assets.body_ids]
            velocities = np.empty((len(TRACKED), 6))
            for i, bid in enumerate(assets.body_ids):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocities[i], 0)
            for key, recomputed in zip(keys, (positions, quats, velocities[:, 3:], velocities[:, :3])):
                difference = np.abs(recomputed - saved[key][frame])
                if key == "body_quat_w":
                    difference = np.minimum(difference.max(-1), np.abs(recomputed + saved[key][frame]).max(-1))
                fk_error[key] = max(fk_error[key], float(difference.max()))
            target = trace["target_position_m_world"][frame].copy()
            target[:, 2] += shift
            residuals.append(np.linalg.norm(data.xpos[trace_ids] - target, axis=-1))
            feet.append(lowest_foot(model, data, assets.foot_geoms))
        limits = model.jnt_range[assets.jids]
        excess = np.maximum(limits[:, 0] - qpos[:, 7:], 0) + np.maximum(qpos[:, 7:] - limits[:, 1], 0)
        residual_p95 = float(np.quantile(residuals, .95))
        velocity_difference = float(np.abs(velocity - qvel).max())
        duration_error = float(row["source_duration_s"] - times[-1])
        gates = {"finite": bool(finite),
                 "native_timing": bool(row["source_hz"] == 120. and times[0] == 0.
                                       and np.max(np.abs(np.diff(times) - .02)) < 1e-10
                                       and -.000001 <= duration_error < .020001),
                 "quaternions": bool(np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=-1) - 1)) < 1e-6),
                 "joint_limits": bool(excess.max() <= 1e-6),
                 "configured_velocity": bool(np.abs(velocity[:, assets.vadr]).max() <= 2.00002),
                 "cartesian_residual": residual_p95 <= .05,
                 "ground_origin_shift": abs(shift) <= .10,
                 "foot_collision_floor": min(feet) >= -.001,
                 "ground_proximity": float(np.mean(np.asarray(feet) <= .03)) >= .5,
                 "root_above_floor": bool(qpos[:, 2].min() > .15),
                 "qpos_export": state_difference < 1e-6,
                 "velocity_export": velocity_difference < 3e-5,
                 "body_fk_export": max(fk_error.values()) < 2e-6}
        result.update({"reference": identity(file), "trace": identity(trace_file),
                       "gates": gates, "qualified_for_training_pilot": all(gates.values()),
                       "frames": len(times), "duration_s": float(times[-1]),
                       "duration_loss_s": duration_error, "qpos_export_error": state_difference,
                       "qvel_export_error": velocity_difference, "body_fk_export_errors": fk_error,
                       "IK_position_p95_m": residual_p95, "foot_floor_min_m": min(feet),
                       "joint_velocity_max_rad_s": float(np.abs(velocity[:, assets.vadr]).max())})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = Q1Assets(args.root)
    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mapping = json.loads((Path(__file__).parent / "retarget_mapping.json").read_text())["body_map"]
    clips = []
    for row in manifest["motions"]:
        result = check_clip(assets, row, mapping)
        clips.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    report = {"status": "INDEPENDENT_REFERENCE_CHECK_COMPLETED", "input_manifest": identity(manifest_path),
              "clips": clips, "qualified_clips": [r["clip"] for r in clips if r["qualified_for_training_pilot"]],
              "rejected_clips": [r["clip"] for r in clips if not r["qualified_for_training_pilot"]],
              "physics_steps": 0, "optimizer_updates": 0, "source": identity(__file__),
              "storage_status": "LOCAL_ONLY", "physics_tracking_validated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "qualified": report["qualified_clips"],
                      "rejected": report["rejected_clips"], "output": identity(args.output)}), flush=True)
    if not report["qualified_clips"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
