#!/usr/bin/env python3
"""Independently recompute BONES source FK, Q1 mapping, export and geometry gates."""

import argparse
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))
from contact_protocol import ContactCorrectedAssets
from supermimic.q1_assets import TRACKED, identity
from verify_reference import foot_bottom

from intake_bones_archive import frozen_selection
from inventory_amass import health


PROTOCOL = "BONES_G1_120HZ_TO_Q1_50HZ_NATIVE_TIME_V1"


def read_source(payload, model, expected_frames):
    if len(payload) > 64 * 1024 ** 2:
        raise ValueError("Source CSV exceeds the admitted size")
    root_fields = ["root_translate" + axis for axis in "XYZ"] + ["root_rotate" + axis for axis in "XYZ"]
    joints = [model.joint(i).name for i in range(1, model.njnt)]
    required = ["Frame", *root_fields, *(name + "_dof" for name in joints)]
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is not None:
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
    if (reader.fieldnames is None or len(reader.fieldnames) != 36
            or len(set(reader.fieldnames)) != 36 or set(reader.fieldnames) != set(required)):
        raise ValueError("Independent reader rejects the documented G1 header")
    rows = []
    for row in reader:
        if None in row or any(row[name] is None for name in required):
            raise ValueError("Malformed source row")
        rows.append([float(row[name]) for name in required])
        if len(rows) > 100000:
            raise ValueError("Source exceeds the frame bound")
    values = np.asarray(rows, dtype=float)
    if (len(values) < 2 or len(values) != expected_frames or not np.isfinite(values).all()
            or np.any(values[:, 0] != np.floor(values[:, 0])) or values[0, 0] < 0
            or np.any(np.diff(values[:, 0]) != 1)):
        raise ValueError("Source frame count, numeric values or native Frame sequence differs")
    times = (values[:, 0] - values[0, 0]) / 120.
    q = np.column_stack((values[:, 1:4] / 100.,
                         Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat()[:, [3, 0, 1, 2]],
                         values[:, 7:] * np.pi / 180.))
    initial_yaw = np.arctan2(Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_matrix()[1, 0],
                              Rotation.from_quat(q[0, [4, 5, 6, 3]]).as_matrix()[0, 0])
    heading = Rotation.from_euler("z", initial_yaw)
    normalized = q.copy()
    normalized[:, :3] = (q[:, :3] - q[0, :3]) @ heading.as_matrix()
    quats = (heading.inv() * Rotation.from_quat(q[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    for index, quaternion in enumerate(quats):
        nonzero = np.flatnonzero(np.abs(quaternion) > 1e-12)
        if len(nonzero) and quaternion[nonzero[0]] < 0:
            quats[index] *= -1
    normalized[:, 3:7] = quats
    normalized = np.round(normalized, 6)
    normalized[normalized == 0] = 0.
    fingerprint = hashlib.sha256(str(normalized.shape).encode() + b":120Hz:" +
                                 normalized.astype("<f8").tobytes()).hexdigest()
    return times, q, fingerprint


def source_foot_ids(model):
    ankles = {model.body(side + "_ankle_roll_link").id for side in ("left", "right")}
    result = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body and body not in ankles:
            body = int(model.body_parentid[body])
        if body in ankles and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            result.append(gid)
    if not result:
        raise ValueError("No declared G1 foot collision geometry")
    return result


def mapped_source(model, assets, native_times, raw_qpos, times, body_names, scale):
    mapping = json.loads((HERE / "bones_retarget_mapping.json").read_text())["body_map"]
    if list(mapping) != body_names:
        raise ValueError("Trace semantic body names differ from the frozen mapping")
    ids = [[current.body(mapping[name]).id for name in body_names] for current in (model, assets.model)]
    source, target = mujoco.MjData(model), mujoco.MjData(assets.model)
    mujoco.mj_forward(model, source)
    target.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(assets.model, target)
    source_cal, target_cal = source.xpos[ids[0]].copy(), target.xpos[ids[1]].copy()
    source_cal[:, 2] -= foot_bottom(model, source, source_foot_ids(model))
    target_cal[:, 2] -= foot_bottom(assets.model, target, assets.foot_geoms)
    a, b = source.xmat[ids[0]].reshape(-1, 3, 3).copy(), target.xmat[ids[1]].reshape(-1, 3, 3).copy()
    root = body_names.index("pelvis")
    alignment = b[root] @ a[root].T
    yaw = lambda r: np.arctan2(r[1, 0], r[0, 0])
    world_alignment = Rotation.from_euler("z", yaw(b[root]) - yaw(a[root])).as_matrix()
    source_cal_local = (source_cal - source_cal[root]) @ a[root]
    target_cal_local = (target_cal - target_cal[root]) @ b[root]
    qpos = np.empty((len(times), 36))
    for column in (*range(3), *range(7, 36)):
        qpos[:, column] = np.interp(times, native_times, raw_qpos[:, column])
    qpos[:, :2] -= raw_qpos[0, :2]
    qpos[:, 3:7] = Slerp(native_times, Rotation.from_quat(raw_qpos[:, [4, 5, 6, 3]]))(times).as_quat()[:, [3, 0, 1, 2]]
    source_positions, source_quats, positions, rotations, roots = [], [], [], [], []
    for state in qpos:
        source.qpos[:] = state
        mujoco.mj_kinematics(model, source)
        p = source.xpos[ids[0]].copy()
        r = source.xmat[ids[0]].reshape(-1, 3, 3).copy()
        root_rotation = alignment @ r[root]
        root_position = target_cal[root] + scale * world_alignment @ (p[root] - source_cal[root])
        local = (p - p[root]) @ r[root]
        positions.append(root_position + (target_cal_local + scale * (local - source_cal_local)) @ root_rotation.T)
        rotations.append(alignment @ r @ a.transpose(0, 2, 1) @ alignment.T @ b)
        source_positions.append(p)
        source_quats.append(source.xquat[ids[0]].copy())
        roots.append(np.concatenate((root_position, Rotation.from_matrix(root_rotation).as_quat()[[3, 0, 1, 2]])))
    return np.asarray(source_positions), np.asarray(source_quats), np.asarray(positions), np.asarray(rotations), np.asarray(roots), ids[1]


def check(assets, source_model, row, scale):
    report = {"clip": row["clip"], "split": row["split"], "bucket": row["bucket"],
              "source_group": row["bones_source"]["source_group"], "qualified_for_training_pilot": False}
    if "output" not in row:
        report["conversion_error"] = row.get("error")
        return report
    for key in ("source_csv", "output", "kinematic_trace"):
        if identity(row[key]["path"]) != row[key]:
            raise ValueError("Source CSV, Q1 export or trace changed")
    native_times, raw, fingerprint = read_source(Path(row["source_csv"]["path"]).read_bytes(), source_model,
                                                row["bones_source"]["move_duration_frames"])
    model, data = assets.model, mujoco.MjData(assets.model)
    with np.load(row["output"]["path"], allow_pickle=False) as saved, np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
        q, v, times = saved["qpos"].astype(float), saved["qvel"].astype(float), saved["time_s"].astype(float)
        expected_count = int(np.floor(native_times[-1] * 50 + 1e-9)) + 1
        if (q.shape != (len(times), 29) or v.shape != (len(times), 28) or len(times) != expected_count
                or not np.allclose(times, np.arange(expected_count) / 50., atol=1e-12, rtol=0)
                or saved["body_names"].tolist() != list(TRACKED) or float(saved["fps"]) != 50.):
            raise ValueError("Q1 dimensions, native time or tracked body order differs")
        finite = all(np.isfinite(saved[key]).all() for key in saved.files if saved[key].dtype.kind in "fiu")
        finite = finite and all(np.isfinite(trace[key]).all() for key in trace.files if trace[key].dtype.kind in "fiu")
        source_pos, source_quat, expected_pos, expected_rot, roots, trace_ids = mapped_source(
            source_model, assets, native_times, raw, times, trace["source_body_names"].tolist(), scale)
        source_error = float(np.abs(trace["source_position_m_world"] - source_pos).max())
        source_rotation_error = float(np.minimum(np.abs(trace["source_orientation_wxyz_world"] - source_quat).max(-1),
                                                  np.abs(trace["source_orientation_wxyz_world"] + source_quat).max(-1)).max())
        mapping_error = float(np.abs(trace["target_position_m_world"] - expected_pos).max())
        mapping_rotation_error = float(np.abs(trace["target_rotation_matrix"] - expected_rot).max())
        shift = float(row["diagnostics"]["global_ground_origin_shift_m"])
        expected_pos[:, :, 2] += shift
        roots[:, 2] += shift
        before = trace["qpos_before_ground_origin"].copy()
        before[:, 2] += shift
        state_error = float(np.abs(before - q).max())
        root_error = float(np.abs(q[:, :3] - roots[:, :3]).max())
        root_quat_error = float(np.minimum(np.abs(q[:, 3:7] - roots[:, 3:7]).max(-1),
                                          np.abs(q[:, 3:7] + roots[:, 3:7]).max(-1)).max())
        velocity = np.empty_like(v)
        for index in range(len(q) - 1):
            mujoco.mj_differentiatePos(model, velocity[index], .02, q[index], q[index + 1])
        velocity[-1] = velocity[-2]
        velocity_error = float(np.abs(velocity - v).max())
        residuals, feet, collision = [], [], []
        fk_error = {key: 0. for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        for frame in range(len(q)):
            data.qpos[:], data.qvel[:] = q[frame], v[frame]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            velocities = np.empty((len(TRACKED), 6))
            for index, bid in enumerate(assets.body_ids):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocities[index], 0)
            values = (data.xpos[assets.body_ids], data.xquat[assets.body_ids], velocities[:, 3:], velocities[:, :3])
            for key, actual in zip(fk_error, values):
                delta = np.abs(saved[key][frame] - actual)
                if key == "body_quat_w":
                    delta = np.minimum(delta.max(-1), np.abs(saved[key][frame] + actual).max(-1))
                fk_error[key] = max(fk_error[key], float(delta.max()))
            residuals.append(np.linalg.norm(data.xpos[trace_ids] - expected_pos[frame], axis=-1))
            feet.append(foot_bottom(model, data, assets.foot_geoms))
            collision.append(max([0.] + [-float(c.dist) for c in data.contact
                                        if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
        limits = model.jnt_range[assets.jids]
        excess = np.maximum(limits[:, 0] - q[:, 7:], 0) + np.maximum(q[:, 7:] - limits[:, 1], 0)
        duration_error = float(native_times[-1] - times[-1])
        residual_p95, feet = float(np.quantile(residuals, .95)), np.asarray(feet)
        support_fraction, collision_max = float(np.mean(feet <= .015)), float(max(collision))
        reference_action = float(np.abs((q[:, 7:] - assets.q0) / assets.scale).max())
        gates = {"finite": bool(finite),
                 "native_120_to_50hz": bool(row["source_hz"] == 120. and row["time_scale"] == 1.
                                            and -.000001 <= duration_error < .020001),
                 "source_fk": max(source_error, source_rotation_error) < 1e-8,
                 "independent_target_mapping": max(mapping_error, mapping_rotation_error) < 1e-8,
                 "unit_quaternions": bool(np.abs(np.linalg.norm(q[:, 3:7], axis=-1) - 1).max() < 1e-6),
                 "joint_limits": bool(excess.max() <= 1e-6),
                 "joint_speed": bool(np.abs(velocity[:, assets.vadr]).max() <= 2.00002),
                 "cartesian_residual": residual_p95 <= .05, "ground_origin_shift": abs(shift) <= .10,
                 "foot_collision_floor": bool(feet.min() >= -.001), "support_within_15mm": support_fraction >= .9,
                 "support_gap_at_most_30mm": bool(feet.max() <= .03), "self_penetration_at_most_3mm": collision_max <= .003,
                 "root_above_floor": bool(q[:, 2].min() > .15), "root_displacement_preserved": root_error < 1e-6,
                 "root_orientation_preserved": root_quat_error < 1e-6, "qpos_export": state_error < 1e-6,
                 "qvel_export": velocity_error < 3e-5,
                 "joint_export": bool(np.abs(saved["joint_pos"] - q[:, 7:]).max() == 0
                                      and np.abs(saved["joint_vel"] - v[:, 6:]).max() == 0),
                 "body_fk_export": max(fk_error.values()) < 2e-6, "action_coverage": reference_action <= 20.,
                 "converter_gates": bool(row["converter_kinematic_candidate"])}
        report.update(reference=row["output"], trace=row["kinematic_trace"], source_csv=row["source_csv"],
                      gates=gates, qualified_for_training_pilot=all(gates.values()), frames=len(times),
                      duration_s=float(times[-1]), duration_loss_s=duration_error,
                      normalized_native_numeric_sha256=fingerprint, numeric_identity_precision_decimals=6,
                      source_fk_error_m=source_error, source_quaternion_error=source_rotation_error,
                      target_mapping_error_m=mapping_error, target_mapping_rotation_error=mapping_rotation_error,
                      IK_position_p95_m=residual_p95, foot_floor_min_m=float(feet.min()),
                      foot_support_gap_max_m=float(feet.max()), support_within_15mm_fraction=support_fraction,
                      self_penetration_max_m=collision_max, root_retarget_error_m=root_error,
                      root_quaternion_error=root_quat_error,
                      reference_horizontal_displacement_m=float(np.linalg.norm(q[:, :2] - q[:1, :2], axis=1).max()),
                      joint_velocity_max_rad_s=float(np.abs(velocity[:, assets.vadr]).max()),
                      qpos_export_error=state_error, qvel_export_error=velocity_error,
                      body_fk_export_errors=fk_error, max_required_action=reference_action)
    return report


def verify(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.motions / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["reference_protocol"] != PROTOCOL or manifest["status"] != "BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING":
        raise ValueError("Expected completed BONES conversion under the native-time protocol")
    versions = [identity(item) for item in (Path(__file__).resolve(), HERE / "bones_retarget_mapping.json",
                HERE / "inventory_amass.py", HERE / "intake_bones_archive.py",
                HERE.parent / "supermimic_q1_phuma/verify_reference.py")]
    frozen = [identity(path), manifest["intake"], manifest["model_comparison"], manifest["source_model"],
              *manifest["sources"], *versions]
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("Frozen converter sources, model or inputs changed")
    intake = json.loads(Path(manifest["intake"]["path"]).read_text())
    _, selected, _ = frozen_selection(Path(intake["frozen"]["metadata_summary"]["path"]))
    expected = selected[manifest["offset"]:manifest["offset"] + manifest["count"]]
    if expected != [row["bones_source"] for row in manifest["motions"]]:
        raise ValueError("Converted source groups or splits differ from P01")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("Registered Q1 assets changed")
    source = mujoco.MjModel.from_xml_path(manifest["source_model"]["path"])
    leg_lengths = []
    for current in (source, assets.model):
        stance = mujoco.MjData(current)
        if current is assets.model:
            stance.qpos[assets.qadr] = assets.q0
        mujoco.mj_kinematics(current, stance)
        leg_lengths.append(np.linalg.norm(stance.xpos[current.body("pelvis").id]
                                          - stance.xpos[current.body("left_ankle_roll_link").id]))
    if not np.isclose(manifest["scale"], leg_lengths[1] / leg_lengths[0], rtol=0, atol=1e-12):
        raise ValueError("Declared displacement scale differs from independent standing leg lengths")
    baseline, reports = health(args.expected_oom_kill), []
    for row in manifest["motions"]:
        result = check(assets, source, row, manifest["scale"])
        reports.append(result)
        health(baseline.get("oom_kill"))
        print(json.dumps(result, allow_nan=False), flush=True)
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("Frozen artifacts changed during independent verification")
    qualified = [row for row in reports if row["qualified_for_training_pilot"]]
    report = {"status": "BONES_INDEPENDENT_REFERENCE_CHECK_COMPLETED", "reference_protocol": PROTOCOL,
              "input_manifest": frozen[0], "clips": reports, "qualified_clips": [row["clip"] for row in qualified],
              "qualified_split_counts": dict(collections.Counter(row["split"] for row in qualified)),
              "source": identity(Path(__file__)), "sources": versions, "physics_steps": 0, "optimizer_updates": 0,
              "corpus_ready_for_training": False, "storage_status": "LOCAL_ONLY",
              "limits": "Per-clip kinematic/native-time checks under the declared source model only. Cross-batch numeric duplicate isolation, corpus quotas, independent moving development/test sets and registered training remain required. No dynamics tracking or hardware validation."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": identity(args.output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    verify(parser.parse_args())
