#!/usr/bin/env python3
"""Retarget four native-speed A3 references into independently checked Q1 motions."""

import argparse
import csv
import json
from pathlib import Path
import sys
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from supermimic.q1_assets import Q1Assets, TRACKED, identity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1"))
from q1_retarget import BodyReference, Q1Retargeter


SOURCE_BODIES = {"pelvis": "pelvis_link", "spine3": "torso_Link",
                 "left_hip": "left_hip_pitch_Link", "right_hip": "right_hip_pitch_Link",
                 "left_knee": "left_knee_Link", "right_knee": "right_knee_Link",
                 "left_ankle": "left_ankle_roll_Link", "right_ankle": "right_ankle_roll_Link",
                 "left_shoulder": "left_shoulder_pitch_Link", "right_shoulder": "right_shoulder_pitch_Link",
                 "left_elbow": "left_elbow_Link", "right_elbow": "right_elbow_Link",
                 "left_wrist": "left_wrist_yaw_Link", "right_wrist": "right_wrist_yaw_Link"}
SOURCE_XML_SHA = "3915053acd573f0cc093a080d64197574fdc146ad4cf91ec1b0ceff71983b75f"
SOURCE_DOC_SHA = "7240718c9ee7bf626d9b39e8ed64142c3ed4b06b24e6fe1f931d43a453a70635"
REFERENCE_PROTOCOL = "P03_NATIVE_TIME_GROUNDED_CALIBRATION_V2"


def source_foot_geometries(model):
    ankles = {model.body(name).id for name in ("left_ankle_roll_Link", "right_ankle_roll_Link")}
    gids = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body and body not in ankles:
            body = int(model.body_parentid[body])
        if body in ankles and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            gids.append(gid)
    if not gids:
        raise ValueError("No source foot collision geometry")
    return gids


def support_height(model, data, gids):
    lows = []
    for gid in gids:
        kind = model.geom_type[gid]
        rotation = data.geom_xmat[gid].reshape(3, 3)
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start + count]
            lows.append((vertices @ rotation.T + data.geom_xpos[gid])[:, 2].min())
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            lows.append(data.geom_xpos[gid, 2] - np.abs(rotation[2]) @ model.geom_size[gid])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            lows.append(data.geom_xpos[gid, 2] - model.geom_size[gid, 0])
        else:
            raise ValueError(f"Register support function for Q1 foot geometry type {kind}")
    return float(min(lows))


def source_reference(model, path, names, source_hz):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ["root_translate" + c for c in "XYZ"] + ["root_rotate" + c for c in "XYZ"]
    joint_names = [model.joint(int(model.actuator_trnid[a, 0])).name
                   for a in range(model.nu)
                   if not model.joint(int(model.actuator_trnid[a, 0])).name.startswith("head_")]
    if len(rows) < 2 or not set(required + joint_names) <= set(rows[0]):
        raise ValueError(f"Source CSV fields or length invalid: {path}")
    source_t = np.arange(len(rows)) / source_hz
    times = np.arange(int(np.floor(source_t[-1] * 50)) + 1) / 50.
    pos = np.array([[float(row["root_translate" + c]) for c in "XYZ"] for row in rows]) * .01
    euler = np.array([[float(row["root_rotate" + c]) for c in "XYZ"] for row in rows])
    qpos = np.tile(model.qpos0, (len(times), 1))
    qpos[:, :3] = np.column_stack([np.interp(times, source_t, pos[:, i]) for i in range(3)])
    qpos[:, 3:7] = Slerp(source_t, Rotation.from_euler("xyz", euler, degrees=True))(times).as_quat()[:, [3, 0, 1, 2]]
    for name in joint_names:
        angles = np.deg2rad([float(row[name]) for row in rows])
        qpos[:, model.jnt_qposadr[model.joint(name).id]] = np.interp(times, source_t, angles)
    if not np.isfinite(qpos).all():
        raise ValueError("Nonfinite source state")
    ids = np.array([model.body(SOURCE_BODIES[name]).id for name in names])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cal_pos, cal_quat = data.xpos[ids].copy(), data.xquat[ids].copy()
    source_calibration_floor = support_height(model, data, source_foot_geometries(model))
    # The source XML's default pose is suspended above its motion floor.
    cal_pos[:, 2] -= source_calibration_floor
    positions, orientations = [], []
    for state in qpos:
        data.qpos[:] = state
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[ids].copy())
        orientations.append(data.xquat[ids].copy())
    reference = BodyReference(times, names, positions, orientations, cal_pos, cal_quat,
                              names, "native_120Hz_A3_robot_reference_to_Q1")
    return reference, {"source": identity(path), "source_frames": len(rows), "source_hz": source_hz,
                       "source_duration_s": float(source_t[-1]), "target_frames": len(times),
                       "target_duration_s": float(times[-1]), "target_hz": 50,
                       "source_calibration_floor_m": source_calibration_floor,
                       "source_joint_fields": joint_names,
                       "unobserved_source_hinges": [model.joint(j).name for j in range(1, model.njnt)
                                                     if model.joint(j).name not in joint_names]}


def export_q1(assets, qpos, residual, destination):
    qpos = np.asarray(qpos).copy()
    model, data = assets.model, mujoco.MjData(assets.model)
    heights = []
    for state in qpos:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        heights.append(support_height(model, data, assets.foot_geoms))
    # One fixed ground-origin shift preserves the native time axis and all velocities.
    shift = -float(min(heights))
    qpos[:, 2] += shift
    qvel = np.empty((len(qpos), model.nv))
    for i in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[i], .02, qpos[i], qpos[i+1])
    qvel[-1] = qvel[-2]
    body = {key: [] for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    feet = []
    for state, velocity in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = state, velocity
        mujoco.mj_forward(model, data)
        body["body_pos_w"].append(data.xpos[assets.body_ids].copy())
        body["body_quat_w"].append(data.xquat[assets.body_ids].copy())
        velocities = []
        for bid in assets.body_ids:
            value = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), value, 0)
            velocities.append(value)
        body["body_lin_vel_w"].append(np.asarray(velocities)[:, 3:])
        body["body_ang_vel_w"].append(np.asarray(velocities)[:, :3])
        feet.append(support_height(model, data, assets.foot_geoms))
    limits = model.jnt_range[assets.jids]
    excess = np.maximum(limits[:, 0] - qpos[:, assets.qadr], 0) + np.maximum(qpos[:, assets.qadr] - limits[:, 1], 0)
    p95 = float(np.quantile(np.linalg.norm(residual, axis=-1), .95))
    diagnostics = {"joint_limit_excess_max_rad": float(excess.max()),
                   "joint_velocity_max_rad_s": float(np.abs(qvel[:, assets.vadr]).max()),
                   "cartesian_IK_residual_p95_m": p95,
                   "global_ground_origin_shift_m": shift,
                   "foot_collision_min_z_m": float(min(feet)),
                   "ground_contact_candidate_fraction": float(np.mean(np.asarray(feet) <= .03)),
                   "root_height_min_m": float(qpos[:, 2].min())}
    gates = {"finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
             "joint_limits": bool(excess.max() <= 1e-6),
             "configured_velocity": bool(np.abs(qvel[:, assets.vadr]).max() <= 2.00001),
             "cartesian_residual": p95 <= .05,
             "ground_origin_shift": abs(shift) <= .10,
             "foot_collision_floor": min(feet) >= -.001,
             "ground_proximity": diagnostics["ground_contact_candidate_fraction"] >= .5,
             "root_above_floor": diagnostics["root_height_min_m"] > .15}
    np.savez_compressed(destination, qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32),
                        joint_pos=qpos[:, 7:].astype(np.float32), joint_vel=qvel[:, 6:].astype(np.float32),
                        fps=np.array(50.), time_s=np.arange(len(qpos)) / 50., body_names=np.array(TRACKED),
                        **{key: np.asarray(value, dtype=np.float32) for key, value in body.items()})
    return {"output": identity(destination), "diagnostics": diagnostics, "kinematic_gates": gates,
            "qualified_for_training_pilot": all(gates.values()), "physics_tracking_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-vendor", type=Path, required=True)
    parser.add_argument("--source-hz", type=float, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_hz != 120:
        raise ValueError("This pinned source requires the documented 120Hz time base")
    xml = args.source_vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml"
    doc = args.source_vendor / "docs/a3_training2sim2deploy.md"
    if identity(xml)["sha256"] != SOURCE_XML_SHA or identity(doc)["sha256"] != SOURCE_DOC_SHA:
        raise ValueError("Source robot model or time-base documentation changed")
    args.output.mkdir(parents=True, exist_ok=False)
    assets = Q1Assets(args.root)
    source_model = mujoco.MjModel.from_xml_path(str(xml))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    target_data = mujoco.MjData(assets.model)
    target_data.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(assets.model, target_data)
    source_leg = np.linalg.norm(source_data.xpos[source_model.body("pelvis_link").id]
                                - source_data.xpos[source_model.body("left_ankle_roll_Link").id])
    target_leg = np.linalg.norm(target_data.xpos[assets.model.body("pelvis").id]
                                - target_data.xpos[assets.model.body("left_ankle_roll_link").id])
    scale = float(target_leg / source_leg)
    motions = []
    for name in args.clips:
        path = args.source_vendor / "a3_data/agibot_a3" / (name + ".csv")
        item = {"clip": name, "qualified_for_training_pilot": False}
        try:
            solver = Q1Retargeter(HERE / "retarget_control.json", HERE / "retarget_mapping.json")
            target_calibration_floor = support_height(solver.sim.model, solver.sim.data, assets.foot_geoms)
            solver.sim.data.qpos[2] -= target_calibration_floor
            mujoco.mj_forward(solver.sim.model, solver.sim.data)
            solver.robot_positions = solver.sim.data.xpos[solver.bodies].copy()
            solver.robot_rotations = solver.sim.data.xmat[solver.bodies].reshape(-1, 3, 3).copy()
            solver.posture.set_target(solver.sim.data.qpos)
            reference, details = source_reference(source_model, path, solver.names, args.source_hz)
            item.update(details)
            item["target_calibration_floor_m"] = target_calibration_floor
            result = solver.retarget(reference, scale)
            qpos, residual = result[1], result[4]
            trace_dir = args.output / "diagnostics"
            trace_dir.mkdir(exist_ok=True)
            trace_path = trace_dir / (name + ".npz")
            np.savez_compressed(trace_path, time_s=reference.times, qpos_before_ground_origin=qpos,
                                target_position_m_world=result[2], target_rotation_matrix=result[3],
                                position_error_m=residual, source_position_m_world=reference.positions,
                                source_orientation_wxyz_world=reference.orientations,
                                source_body_names=np.array(reference.names))
            item.update(export_q1(assets, qpos, residual, args.output / (name + ".npz")))
            item["kinematic_trace"] = identity(trace_path)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc()
        motions.append(item)
        print(json.dumps({"name": name, **item.get("diagnostics", {}),
                          "gates": item.get("kinematic_gates"), "error": item.get("error")}), flush=True)
    qualified = sum(m["qualified_for_training_pilot"] for m in motions)
    status = ("KINEMATIC_REFERENCE_PASSED" if qualified == len(motions) else
              "KINEMATIC_REFERENCE_PARTIAL" if qualified else "REFERENCE_QUALITY_FAILED")
    receipt = {"status": status, "reference_protocol": REFERENCE_PROTOCOL,
               "qualified_clips": [m["clip"] for m in motions if m["qualified_for_training_pilot"]],
               "rejected_clips": [m["clip"] for m in motions if not m["qualified_for_training_pilot"]],
               "scale": scale, "source_model": identity(xml), "timebase_source": identity(doc),
               "q1_assets": assets.manifest(), "motions": motions,
               "sources": [identity(path) for path in (Path(__file__), HERE / "retarget_control.json", HERE / "retarget_mapping.json", HERE.parent / "q1/q1_retarget.py", HERE.parent / "q1/q1_sim.py")],
               "note": "Robot-reference retargeting; native timing; grounded canonical poses and one clip floor shift; not human GT or dynamics validation",
               "storage_status": "LOCAL_ONLY"}
    with (args.output / "manifest.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "motions": len(motions), "manifest": identity(args.output / "manifest.json")}), flush=True)
    if receipt["status"] != "KINEMATIC_REFERENCE_PASSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
