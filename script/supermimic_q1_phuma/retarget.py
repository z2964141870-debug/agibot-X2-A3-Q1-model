#!/usr/bin/env python3
"""Retarget grouped PHUMA G1 motions to Q1 at their native time scale."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

import mujoco
import mink
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
sys.path.insert(0, str(HERE.parent / "q1"))
from contact_protocol import ContactCorrectedAssets
from prepare_motion import export_q1, support_height
from q1_retarget import BodyReference, Q1Retargeter, heading_rotation
from supermimic.q1_assets import identity

from acquire_source import COMMIT
from motion_io import JOINT_NAMES_23, decode_g1
from scan_corpus import ARCHIVE_SHA, resource_check


PROTOCOL = "PHUMA_G1_30HZ_TO_Q1_50HZ_NATIVE_TIME_V2"
BODY_MAP = json.loads((HERE / "retarget_mapping.json").read_text())["body_map"]


def source_feet(model):
    ankles = {model.body(name).id for name in ("left_ankle_roll_link", "right_ankle_roll_link")}
    result = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body and body not in ankles:
            body = int(model.body_parentid[body])
        if body in ankles and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            result.append(gid)
    if not result:
        raise ValueError("Source model has no foot collision geometry")
    return result


def resample_qpos(model, motion):
    names = tuple(model.joint(j).name for j in range(1, model.njnt))
    if (model.nq, model.nv, model.nu) != (30, 29, 23) or names != JOINT_NAMES_23:
        raise ValueError("Pinned G1 joint order does not match the decoded columns")
    source_time = np.arange(len(motion["root_trans"])) / motion["fps"]
    times = np.arange(int(np.floor(source_time[-1] * 50 + 1e-9)) + 1) / 50.
    root = motion["root_trans"].copy()
    root[:, :2] -= root[0, :2]
    qpos = np.tile(model.qpos0, (len(times), 1))
    for axis in range(3):
        qpos[:, axis] = np.interp(times, source_time, root[:, axis])
    qpos[:, 3:7] = Slerp(source_time, Rotation.from_quat(motion["root_ori"]))(times).as_quat()[:, [3, 0, 1, 2]]
    for column, name in enumerate(names):
        address = model.jnt_qposadr[model.joint(name).id]
        qpos[:, address] = np.interp(times, source_time, motion["dof_pos_23"][:, column])
    return times, qpos


def source_reference(model, motion, names):
    times, qpos = resample_qpos(model, motion)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.body(BODY_MAP[name]).id for name in names]
    calibration_positions = data.xpos[ids].copy()
    calibration_orientations = data.xquat[ids].copy()
    floor = support_height(model, data, source_feet(model))
    calibration_positions[:, 2] -= floor
    positions, orientations = [], []
    for state in qpos:
        data.qpos[:] = state
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[ids].copy())
        orientations.append(data.xquat[ids].copy())
    reference = BodyReference(times, names, positions, orientations,
                              calibration_positions, calibration_orientations,
                              names, "PHUMA_G1_robot_30Hz_native_time_to_Q1")
    return reference, {"source_frames": len(motion["root_trans"]), "source_hz": 30.,
                       "source_duration_s": (len(motion["root_trans"]) - 1) / 30.,
                       "target_frames": len(times), "target_duration_s": float(times[-1]),
                       "target_hz": 50., "source_calibration_floor_m": floor,
                       "source_horizontal_origin_m": motion["root_trans"][0, :2].tolist()}


def grounded_solver(assets):
    solver = Q1Retargeter(HERE / "retarget_control.json", HERE / "retarget_mapping.json")
    floor = support_height(solver.sim.model, solver.sim.data, assets.foot_geoms)
    solver.sim.data.qpos[2] -= floor
    mujoco.mj_forward(solver.sim.model, solver.sim.data)
    solver.robot_positions = solver.sim.data.xpos[solver.bodies].copy()
    solver.robot_rotations = solver.sim.data.xmat[solver.bodies].reshape(-1, 3, 3).copy()
    solver.posture.set_target(solver.sim.data.qpos)
    return solver, floor


def initialize_first_pose(solver, source, scale):
    root = solver.root
    cal = Rotation.from_quat(source.calibration_orientations, scalar_first=True).as_matrix()
    rotations = Rotation.from_quat(source.orientations[0], scalar_first=True).as_matrix()
    alignment = solver.robot_rotations[root] @ cal[root].T
    heading = heading_rotation(solver.robot_rotations[root]) @ heading_rotation(cal[root]).T
    root_rotation = alignment @ rotations[root]
    root_position = solver.robot_positions[root] + scale * heading @ (source.positions[0, root] - source.calibration_positions[root])
    source_cal_local = (source.calibration_positions - source.calibration_positions[root]) @ cal[root]
    robot_cal_local = (solver.robot_positions - solver.robot_positions[root]) @ solver.robot_rotations[root]
    local = (source.positions[0] - source.positions[0, root]) @ rotations[root]
    targets = root_position + (robot_cal_local + scale * (local - source_cal_local)) @ root_rotation.T
    target_rotations = alignment @ rotations @ cal.transpose(0, 2, 1) @ alignment.T @ solver.robot_rotations
    q = solver.sim.data.qpos.copy()
    q[:3], q[3:7] = root_position, Rotation.from_matrix(root_rotation).as_quat(scalar_first=True)
    solver.configuration.update(q)
    solver.root_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(root_rotation), root_position))
    for index, task in solver.tasks:
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(target_rotations[index]), targets[index]))
    if not isinstance(solver.limits[0], mink.ConfigurationLimit):
        raise ValueError("Expected the established joint configuration limit")
    # Fit a static initial pose before applying temporal speed limits to real frames.
    for iteration in range(32):
        velocity = mink.solve_ik(solver.configuration, [task for _, task in solver.tasks] + [solver.posture],
                                 .02, solver.config["solver"], damping=solver.config["damping"],
                                 safety_break=True, limits=[solver.limits[0]], constraints=[solver.root_task])
        solver.configuration.integrate_inplace(velocity, .02)
        if np.linalg.norm(velocity) < 1e-5:
            break
    solver.sim.data.qpos[:] = solver.configuration.data.qpos
    mujoco.mj_forward(solver.sim.model, solver.sim.data)
    residual = np.linalg.norm(solver.sim.data.xpos[solver.bodies] - targets, axis=-1)
    return {"method": "static_initial_pose_IK_without_temporal_velocity_limit",
            "iterations": iteration + 1, "initial_body_error_p95_m": float(np.quantile(residual, .95)),
            "added_reference_frames": 0, "time_scale": 1.0}


def convert(root, selection_path, vendor, output, pilot):
    if output.exists():
        raise FileExistsError(output)
    baseline_oom = resource_check()
    selection = json.loads(selection_path.read_text())
    if selection["upstream_commit"] != COMMIT or selection["archive"]["sha256"] != ARCHIVE_SHA:
        raise ValueError("Unexpected PHUMA input lineage")
    if identity(selection["archive"]["path"]) != selection["archive"]:
        raise ValueError("PHUMA ZIP changed")
    source_receipt = vendor / "source_receipt.json"
    pinned = json.loads(source_receipt.read_text())
    if identity(source_receipt) != selection["source_receipt"] or any(identity(x["path"]) != x for x in pinned["files"]):
        raise ValueError("Pinned PHUMA assets changed")
    selected = selection["selected"]
    if pilot:
        used, first = set(), []
        for row in selected:
            key = row["split"], row["bucket"]
            if key not in used:
                first.append(row)
                used.add(key)
        selected = first
    if len({row["source_group"] for row in selected}) != len(selected):
        raise ValueError("A source group appeared more than once")
    assets = ContactCorrectedAssets(root)
    xml = vendor / "asset/humanoid_model/g1/custom.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target = mujoco.MjData(assets.model)
    target.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(assets.model, target)
    source_leg = np.linalg.norm(data.xpos[model.body("pelvis").id] - data.xpos[model.body("left_ankle_roll_link").id])
    target_leg = np.linalg.norm(target.xpos[assets.model.body("pelvis").id] - target.xpos[assets.model.body("left_ankle_roll_link").id])
    scale = float(target_leg / source_leg)
    output.mkdir(parents=True)
    (output / "diagnostics").mkdir()
    started, motions = time.monotonic(), []
    with zipfile.ZipFile(selection["archive"]["path"]) as package:
        for row in selected:
            clip = row["split"] + "_" + row["bucket"] + "_" + hashlib.sha256(row["member"].encode()).hexdigest()[:16]
            item = {"clip": clip, "phuma_source": row, "split": row["split"],
                    "bucket": row["bucket"], "qualified_for_training_pilot": False}
            try:
                raw = package.read(row["member"])
                if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("Selected PHUMA source bytes changed")
                solver, floor = grounded_solver(assets)
                reference, details = source_reference(model, decode_g1(raw), solver.names)
                item.update(details)
                item["target_calibration_floor_m"] = floor
                item["first_pose_initialization"] = initialize_first_pose(solver, reference, scale)
                result = solver.retarget(reference, scale)
                trace = output / "diagnostics" / (clip + ".npz")
                np.savez_compressed(trace, time_s=reference.times,
                                    qpos_before_ground_origin=result[1],
                                    target_position_m_world=result[2], target_rotation_matrix=result[3],
                                    position_error_m=result[4], source_position_m_world=reference.positions,
                                    source_orientation_wxyz_world=reference.orientations,
                                    source_body_names=np.array(reference.names))
                item.update(export_q1(assets, result[1], result[4], output / (clip + ".npz")))
                item["kinematic_trace"] = identity(trace)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                item["traceback"] = traceback.format_exc()
            motions.append(item)
            if resource_check() != baseline_oom:
                raise RuntimeError("OOM count changed during PHUMA retargeting")
            print(json.dumps({"clip": clip, "source": row["member"],
                              "gates": item.get("kinematic_gates"),
                              "diagnostics": item.get("diagnostics"), "error": item.get("error")}), flush=True)
    qualified = [m for m in motions if m["qualified_for_training_pilot"]]
    sources = [Path(__file__), HERE / "motion_io.py", HERE / "retarget_mapping.json", HERE / "retarget_control.json",
               HERE.parent / "q1/q1_retarget.py", HERE.parent / "q1/q1_sim.py",
               HERE.parent / "supermimic_for_q1/prepare_motion.py"]
    manifest = {"status": "PHUMA_RETARGETING_COMPLETED", "reference_protocol": PROTOCOL,
                "selection": identity(selection_path), "source_receipt": identity(source_receipt),
                "archive": selection["archive"], "source_model": identity(xml),
                "q1_assets": assets.manifest(), "scale": scale, "pilot": pilot,
                "motions": motions, "qualified_clips": [m["clip"] for m in qualified],
                "qualified_split_counts": dict(collections.Counter(m["split"] for m in qualified)),
                "sources": [identity(p) for p in sources], "seconds": time.monotonic() - started,
                "new_physics_steps": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
                "limits": "Kinematics only; independent export, support and self-collision checks remain required."}
    with (output / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": manifest["status"], "qualified_split_counts": manifest["qualified_split_counts"],
                      "manifest": identity(output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    convert(args.root.resolve(), args.selection.resolve(), args.vendor.resolve(), args.output.resolve(), args.pilot)
