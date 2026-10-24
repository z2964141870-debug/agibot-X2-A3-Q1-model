#!/usr/bin/env python3
"""Independently verify PHUMA native timing, Q1 exports and support geometry."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))
from contact_protocol import ContactCorrectedAssets
from supermimic.q1_assets import TRACKED, identity
from motion_io import decode_g1


PROTOCOL = "PHUMA_G1_30HZ_TO_Q1_50HZ_NATIVE_TIME_V1"


def foot_bottom(model, data, gids):
    lows = []
    for gid in gids:
        matrix = data.geom_xmat[gid].reshape(3, 3)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            low = data.geom_xpos[gid, 2] - model.geom_size[gid, 0]
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            low = data.geom_xpos[gid, 2] - np.abs(matrix[2]) @ model.geom_size[gid]
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            low = (model.mesh_vert[start:start + count] @ matrix[2] + data.geom_xpos[gid, 2]).min()
        else:
            raise ValueError("Unreviewed Q1 foot shape")
        lows.append(float(low))
    return min(lows)


def check(assets, row, scale, package):
    report = {"clip": row["clip"], "split": row["split"], "bucket": row["bucket"],
              "source_group": row["phuma_source"]["source_group"], "qualified_for_training_pilot": False}
    if "output" not in row:
        report["conversion_error"] = row.get("error")
        return report
    for key in ("output", "kinematic_trace"):
        if identity(row[key]["path"]) != row[key]:
            raise ValueError("Q1 reference or IK trace changed")
    source_row = row["phuma_source"]
    raw = package.read(source_row["member"])
    if len(raw) != source_row["bytes"] or hashlib.sha256(raw).hexdigest() != source_row["sha256"]:
        raise ValueError("Raw PHUMA member changed")
    source = decode_g1(raw)
    model, data = assets.model, mujoco.MjData(assets.model)
    mapping = json.loads((HERE / "retarget_mapping.json").read_text())["body_map"]
    with np.load(row["output"]["path"], allow_pickle=False) as saved, np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
        q, v = saved["qpos"].astype(float), saved["qvel"].astype(float)
        times = saved["time_s"]
        if q.shape != (len(times), 29) or v.shape != (len(times), 28) or len(times) < 2:
            raise ValueError("Incorrect Q1 reference dimensions")
        if saved["body_names"].tolist() != list(TRACKED) or float(saved["fps"]) != 50:
            raise ValueError("Q1 tracked body order or frequency differs")
        finite = all(np.isfinite(saved[key]).all() for key in saved.files if saved[key].dtype.kind in "fiu")
        before = trace["qpos_before_ground_origin"].copy()
        shift = row["diagnostics"]["global_ground_origin_shift_m"]
        before[:, 2] += shift
        state_error = float(np.abs(before - q).max())
        velocity = np.empty_like(v)
        for i in range(len(q) - 1):
            mujoco.mj_differentiatePos(model, velocity[i], .02, q[i], q[i + 1])
        velocity[-1] = velocity[-2]
        velocity_error = float(np.abs(velocity - v).max())
        trace_ids = [model.body(mapping[name]).id for name in trace["source_body_names"].tolist()]
        residuals, feet, collision = [], [], []
        fk_error = {name: 0. for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
        for frame in range(len(q)):
            data.qpos[:], data.qvel[:] = q[frame], v[frame]
            mujoco.mj_forward(model, data)
            velocities = np.empty((len(TRACKED), 6))
            for i, bid in enumerate(assets.body_ids):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, int(bid), velocities[i], 0)
            values = (data.xpos[assets.body_ids], data.xquat[assets.body_ids], velocities[:, 3:], velocities[:, :3])
            for key, value in zip(fk_error, values):
                delta = np.abs(saved[key][frame] - value)
                if key == "body_quat_w":
                    delta = np.minimum(delta.max(-1), np.abs(saved[key][frame] + value).max(-1))
                fk_error[key] = max(fk_error[key], float(delta.max()))
            target = trace["target_position_m_world"][frame].copy()
            target[:, 2] += shift
            residuals.append(np.linalg.norm(data.xpos[trace_ids] - target, axis=-1))
            feet.append(foot_bottom(model, data, assets.foot_geoms))
            collision.append(max([0.] + [-float(c.dist) for c in data.contact
                                        if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
        limits = model.jnt_range[assets.jids]
        excess = np.maximum(limits[:, 0] - q[:, 7:], 0) + np.maximum(q[:, 7:] - limits[:, 1], 0)
        source_times = np.arange(len(source["root_trans"])) / 30.
        source_xy = source["root_trans"][:, :2] - source["root_trans"][0, :2]
        expected_xy = scale * np.column_stack([np.interp(times, source_times, source_xy[:, i]) for i in range(2)])
        xy_error = float(np.max(np.abs(q[:, :2] - expected_xy)))
        expected_duration = (len(source["root_trans"]) - 1) / 30.
        duration_error = float(expected_duration - times[-1])
        residual_p95 = float(np.quantile(residuals, .95))
        feet = np.asarray(feet)
        support_fraction = float(np.mean(feet <= .015))
        collision_max = float(max(collision))
        reference_action = float(np.abs((q[:, 7:] - assets.q0) / assets.scale).max())
        gates = {
            "finite": bool(finite),
            "native_30_to_50hz": bool(row["source_hz"] == 30. and times[0] == 0.
                                      and np.max(np.abs(np.diff(times) - .02)) < 1e-10
                                      and -.000001 <= duration_error < .020001),
            "unit_quaternions": bool(np.abs(np.linalg.norm(q[:, 3:7], axis=-1) - 1).max() < 1e-6),
            "joint_limits": bool(excess.max() <= 1e-6),
            "joint_speed": bool(np.abs(velocity[:, assets.vadr]).max() <= 2.00002),
            "cartesian_residual": residual_p95 <= .05,
            "ground_origin_shift": abs(shift) <= .10,
            "foot_collision_floor": bool(feet.min() >= -.001),
            "support_within_15mm": support_fraction >= .9,
            "support_gap_at_most_30mm": bool(feet.max() <= .03),
            "self_penetration_at_most_3mm": collision_max <= .003,
            "root_above_floor": bool(q[:, 2].min() > .15),
            "root_displacement_preserved": xy_error < 1e-6,
            "qpos_export": state_error < 1e-6,
            "qvel_export": velocity_error < 3e-5,
            "joint_export": bool(np.abs(saved["joint_pos"] - q[:, 7:]).max() == 0
                                 and np.abs(saved["joint_vel"] - v[:, 6:]).max() == 0),
            "body_fk_export": max(fk_error.values()) < 2e-6,
            "action_coverage": reference_action <= 20.,
            "converter_gates": row["qualified_for_training_pilot"],
        }
        report.update({"reference": row["output"], "trace": row["kinematic_trace"],
                       "gates": gates, "qualified_for_training_pilot": all(gates.values()),
                       "frames": len(times), "duration_s": float(times[-1]), "duration_loss_s": duration_error,
                       "IK_position_p95_m": residual_p95, "foot_floor_min_m": float(feet.min()),
                       "foot_support_gap_p95_m": float(np.quantile(feet, .95)),
                       "foot_support_gap_max_m": float(feet.max()), "support_within_15mm_fraction": support_fraction,
                       "self_penetration_max_m": collision_max, "root_xy_retarget_error_m": xy_error,
                       "reference_horizontal_displacement_m": float(np.linalg.norm(q[:, :2] - q[:1, :2], axis=1).max()),
                       "joint_velocity_max_rad_s": float(np.abs(velocity[:, assets.vadr]).max()),
                       "qpos_export_error": state_error, "qvel_export_error": velocity_error,
                       "body_fk_export_errors": fk_error, "max_required_action": reference_action})
    return report


def verify(root, motions, output):
    if output.exists():
        raise FileExistsError(output)
    manifest_path = motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["reference_protocol"] != PROTOCOL:
        raise ValueError("A different reference protocol was supplied")
    assets = ContactCorrectedAssets(root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("Q1 model/control identities changed")
    for record in manifest["sources"]:
        if identity(record["path"]) != record:
            raise ValueError("Retargeting code or configuration changed")
    groups, reports = set(), []
    with zipfile.ZipFile(manifest["archive"]["path"]) as package:
        for row in manifest["motions"]:
            group = row["phuma_source"]["source_group"]
            if group in groups:
                raise ValueError("Source group repeated within references")
            groups.add(group)
            result = check(assets, row, manifest["scale"], package)
            reports.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    qualified = [row for row in reports if row["qualified_for_training_pilot"]]
    report = {"status": "INDEPENDENT_REFERENCE_CHECK_COMPLETED", "reference_protocol": PROTOCOL,
              "input_manifest": identity(manifest_path), "clips": reports,
              "qualified_clips": [row["clip"] for row in qualified],
              "qualified_split_counts": dict(collections.Counter(row["split"] for row in qualified)),
              "rejected_clips": [row["clip"] for row in reports if not row["qualified_for_training_pilot"]],
              "physics_steps": 0, "optimizer_updates": 0, "source": identity(__file__),
              "storage_status": "LOCAL_ONLY", "physics_tracking_validated": False,
              "limits": "Independent native-time/export and geometry checks; neither a dynamic tracker nor hardware validation."}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "qualified_split_counts": report["qualified_split_counts"],
                      "receipt": identity(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.root.resolve(), args.motions.resolve(), args.output.resolve())
