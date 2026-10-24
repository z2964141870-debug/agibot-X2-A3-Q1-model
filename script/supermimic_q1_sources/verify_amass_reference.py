#!/usr/bin/env python3
"""Independently check AMASS body mapping and diagnostic Q1 export geometry."""

import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))

from prepare_amass_body import identity
from retarget_amass_body import PROTOCOL, body_rows


def independent_mapping(body, robot_positions, robot_rotations, scale):
    import numpy as np
    from scipy.spatial.transform import Rotation

    root = body.names.index("pelvis")
    calibration = Rotation.from_quat(body.calibration_orientations, scalar_first=True).as_matrix()
    rotations = Rotation.from_quat(body.orientations.reshape(-1, 4), scalar_first=True).as_matrix()
    rotations = rotations.reshape(len(body.times), len(body.names), 3, 3)
    alignment = robot_rotations[root] @ calibration[root].T
    yaw = lambda r: np.arctan2(r[1, 0], r[0, 0])
    heading = Rotation.from_euler("z", yaw(robot_rotations[root]) - yaw(calibration[root])).as_matrix()
    root_positions = robot_positions[root] + scale * (
        body.positions[:, root] - body.calibration_positions[root]) @ heading.T
    root_rotations = alignment @ rotations[:, root]
    source_cal_local = (body.calibration_positions - body.calibration_positions[root]) @ calibration[root]
    robot_cal_local = (robot_positions - robot_positions[root]) @ robot_rotations[root]
    local = (body.positions - body.positions[:, root:root + 1]) @ rotations[:, root]
    positions = root_positions[:, None] + (robot_cal_local + scale * (local - source_cal_local)) @ root_rotations.transpose(0, 2, 1)
    target_rotations = alignment @ rotations @ calibration.transpose(0, 2, 1) @ alignment.T @ robot_rotations
    root_quaternions = Rotation.from_matrix(root_rotations).as_quat(scalar_first=True)
    return positions, target_rotations, root_positions, root_quaternions


def check(assets, row, body):
    import mujoco
    import numpy as np
    from supermimic.q1_assets import TRACKED
    from verify_reference import foot_bottom

    report = {"clip": row["clip"], "split": "UNASSIGNED_DIAGNOSTIC_ONLY",
              "qualified_for_training_pilot": False, "geometry_qualified": False}
    if "output" not in row:
        report["conversion_error"] = row.get("error")
        return report
    for key in ("body_reference", "output", "kinematic_trace"):
        if identity(row[key]["path"]) != row[key]:
            raise ValueError("AMASS body, Q1 export or diagnostic trace changed")
    if identity(row["source"]["source"]["path"]) != row["source"]["source"]:
        raise ValueError("Original AMASS source changed")
    model, data = assets.model, mujoco.MjData(assets.model)
    data.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(model, data)
    mapping = json.loads((HERE.parent / "supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"]
    ids = [model.body(mapping[name]).id for name in body.names]
    robot_cal = data.xpos[ids].copy()
    robot_cal[:, 2] -= foot_bottom(model, data, assets.foot_geoms)
    robot_rot = data.xmat[ids].reshape(-1, 3, 3).copy()
    robot_lengths, source_lengths = [], []
    for side in ("left", "right"):
        chain = [body.names.index(name) for name in ("pelvis", side + "_hip", side + "_knee", side + "_ankle")]
        robot_lengths.append(np.linalg.norm(np.diff(robot_cal[chain], axis=0), axis=-1).sum())
        source_lengths.append(np.median(np.linalg.norm(np.diff(body.positions[:, chain], axis=1), axis=-1).sum(-1)))
    scale = float(np.mean(robot_lengths) / np.mean(source_lengths))
    expected_pos, expected_rot, roots, root_quats = independent_mapping(body, robot_cal, robot_rot, scale)
    with np.load(row["output"]["path"], allow_pickle=False) as saved, np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
        q, v, times = saved["qpos"].astype(float), saved["qvel"].astype(float), saved["time_s"].astype(float)
        if (q.shape != (201, 29) or v.shape != (201, 28)
                or not np.allclose(times, np.arange(201) / 50., atol=1e-12, rtol=0)
                or saved["body_names"].tolist() != list(TRACKED) or float(saved["fps"]) != 50.
                or trace["source_body_names"].tolist() != body.names):
            raise ValueError("Q1 time, dimensions or semantic body order differs")
        finite = all(np.isfinite(z[key]).all() for z in (saved, trace) for key in z.files if z[key].dtype.kind in "fiu")
        mapping_error = float(np.abs(trace["target_position_m_world"] - expected_pos).max())
        rotation_error = float(np.abs(trace["target_rotation_matrix"] - expected_rot).max())
        body_error = float(np.abs(trace["source_position_m_world"] - body.positions).max())
        body_quat_error = float(np.minimum(np.abs(trace["source_orientation_wxyz_world"] - body.orientations).max(-1),
                                           np.abs(trace["source_orientation_wxyz_world"] + body.orientations).max(-1)).max())
        shift = float(row["diagnostics"]["global_ground_origin_shift_m"])
        expected_pos[:, :, 2] += shift
        roots[:, 2] += shift
        before = trace["qpos_before_ground_origin"].copy()
        before[:, 2] += shift
        state_error = float(np.abs(before - q).max())
        root_error = float(np.abs(q[:, :3] - roots).max())
        root_quat_error = float(np.minimum(np.abs(q[:, 3:7] - root_quats).max(-1),
                                          np.abs(q[:, 3:7] + root_quats).max(-1)).max())
        velocity = np.empty_like(v)
        for i in range(200):
            mujoco.mj_differentiatePos(model, velocity[i], .02, q[i], q[i + 1])
        velocity[-1] = velocity[-2]
        velocity_error = float(np.abs(velocity - v).max())
        residuals, feet, penetration = [], [], []
        fk_error = {key: 0. for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        for frame in range(201):
            data.qpos[:], data.qvel[:] = q[frame], v[frame]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            velocities = np.empty((len(TRACKED), 6))
            for i, bid in enumerate(assets.body_ids):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocities[i], 0)
            actual = (data.xpos[assets.body_ids], data.xquat[assets.body_ids], velocities[:, 3:], velocities[:, :3])
            for key, value in zip(fk_error, actual):
                difference = np.abs(saved[key][frame] - value)
                if key == "body_quat_w":
                    difference = np.minimum(difference.max(-1), np.abs(saved[key][frame] + value).max(-1))
                fk_error[key] = max(fk_error[key], float(difference.max()))
            residuals.append(np.linalg.norm(data.xpos[ids] - expected_pos[frame], axis=-1))
            feet.append(foot_bottom(model, data, assets.foot_geoms))
            penetration.append(max([0.] + [-float(c.dist) for c in data.contact
                                          if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
        limits = model.jnt_range[assets.jids]
        excess = np.maximum(limits[:, 0] - q[:, 7:], 0) + np.maximum(q[:, 7:] - limits[:, 1], 0)
        residual_p95, feet = float(np.quantile(residuals, .95)), np.asarray(feet)
        support_fraction, collision = float(np.mean(feet <= .015)), float(max(penetration))
        action = float(np.abs((q[:, 7:] - assets.q0) / assets.scale).max())
        gates = {"finite": bool(finite),
                 "native_120_to_50hz": bool(row["source_hz"] == 120. and row["time_scale"] == 1.
                                            and np.array_equal(trace["time_s"], body.times)),
                 "independent_leg_scale": abs(scale - row["scale"]) < 1e-12,
                 "independent_target_mapping": max(mapping_error, rotation_error, body_error, body_quat_error) < 1e-8,
                 "unit_quaternions": bool(np.abs(np.linalg.norm(q[:, 3:7], axis=-1) - 1).max() < 1e-6),
                 "joint_limits": bool(excess.max() <= 1e-6),
                 "joint_speed": bool(np.abs(velocity[:, assets.vadr]).max() <= 2.00002),
                 "cartesian_residual": residual_p95 <= .05, "ground_origin_shift": abs(shift) <= .10,
                 "foot_collision_floor": bool(feet.min() >= -.001), "support_within_15mm": support_fraction >= .9,
                 "support_gap_at_most_30mm": bool(feet.max() <= .03), "self_penetration_at_most_3mm": collision <= .003,
                 "root_above_floor": bool(q[:, 2].min() > .15), "root_displacement_preserved": root_error < 1e-6,
                 "root_orientation_preserved": root_quat_error < 1e-6, "qpos_export": state_error < 1e-6,
                 "qvel_export": velocity_error < 3e-5,
                 "joint_export": bool(np.abs(saved["joint_pos"] - q[:, 7:]).max() == 0
                                      and np.abs(saved["joint_vel"] - v[:, 6:]).max() == 0),
                 "body_fk_export": max(fk_error.values()) < 2e-6, "action_coverage": action <= 20.,
                 "converter_gates": bool(row["converter_kinematic_candidate"])}
        report.update(gates=gates, geometry_qualified=all(gates.values()), reference=row["output"],
                      trace=row["kinematic_trace"], frames=201, duration_s=4., scale=scale,
                      target_mapping_error_m=mapping_error, target_mapping_rotation_error=rotation_error,
                      body_trace_error_m=body_error, body_trace_quaternion_error=body_quat_error,
                      IK_position_p95_m=residual_p95, foot_floor_min_m=float(feet.min()),
                      foot_support_gap_max_m=float(feet.max()), support_within_15mm_fraction=support_fraction,
                      self_penetration_max_m=collision, root_retarget_error_m=root_error,
                      root_quaternion_error=root_quat_error, qpos_export_error=state_error,
                      qvel_export_error=velocity_error, body_fk_export_errors=fk_error,
                      joint_velocity_max_rad_s=float(np.abs(velocity[:, assets.vadr]).max()),
                      reference_horizontal_displacement_m=float(np.linalg.norm(q[:, :2] - q[:1, :2], axis=1).max()),
                      max_required_action=action)
    return report


def worker(args):
    from contact_protocol import ContactCorrectedAssets
    from q1_retarget import BodyReference
    from runtime_guard import save

    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["reference_protocol"] != PROTOCOL or manifest["qualified_for_training"]
            or identity(manifest["body_manifest"]["path"]) != manifest["body_manifest"]):
        raise ValueError("Expected intact, unadmitted AMASS Q1 conversion")
    body_manifest = json.loads(Path(manifest["body_manifest"]["path"]).read_text())
    bodies = {row["clip"]: row for row in body_rows(body_manifest)}
    if ([row["clip"] for row in manifest["motions"]] != list(bodies)
            or any(identity(row["path"]) != row for row in manifest["sources"] + body_manifest["sources"])):
        raise ValueError("AMASS body set or source code identity changed")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("Q1 model/control identity differs from conversion")
    names = list(json.loads((HERE.parent / "supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"])
    reports = []
    for row in manifest["motions"]:
        original = bodies[row["clip"]]
        if row["body_reference"] != original["body_reference"] or row["source"] != original["source"]:
            raise ValueError("Q1 candidate provenance differs from the body receipt")
        result = check(assets, row, BodyReference.load(original["body_reference"]["path"], names))
        reports.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    qualified = [row for row in reports if row["geometry_qualified"]]
    report = {"status": "AMASS_INDEPENDENT_Q1_GEOMETRY_CHECK_COMPLETED_SPLITS_PENDING",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "reference_protocol": PROTOCOL,
              "retarget_manifest": identity(manifest_path), "body_manifest": manifest["body_manifest"],
              "clips": reports, "geometry_qualified_clips": [row["clip"] for row in qualified],
              "geometry_qualified_count": len(qualified), "geometry_qualified_duration_s": 4. * len(qualified),
              "qualified_for_training": False, "qualified_training_count": 0,
              "rejection_gate_counts": dict(collections.Counter(key for row in reports
                                            for key, passed in row.get("gates", {}).items() if not passed)),
              "new_physics_steps": 0, "new_optimizer_updates": 0, "source": identity(Path(__file__)),
              "storage_status": "LOCAL_ONLY", "limits": "Original 19 geometry/export thresholds retained, plus independent scale, mapping and root orientation. This does not repeat full SMPL-X mesh FK, reconcile HumanML aliases, assign splits, establish training admission or validate dynamics."}
    save(args.output / "reference_check.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "clips"}), flush=True)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new AMASS independent-check directory")
    budget = MemoryBoundBudget(StorageBudget(output, args.motions.resolve(strict=True),
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    os.nice(10)
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--root", str(root),
               "--motions", str(args.motions.resolve()), "--output", str(output),
               "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "reference_check.json", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
