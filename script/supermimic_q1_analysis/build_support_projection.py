#!/usr/bin/env python3
"""Build a separate vertical-support reference candidate without integration."""

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

from inspect_reference_dynamics import (
    bottom_point, checked, configured_model, container_guard, describe, identity,
    inspect_clip,
)


SETTINGS = {"minimum_lower_foot_z_m": -.0005, "maximum_lower_foot_z_m": -.00005,
            "target_lower_foot_z_m": -.00025, "second_difference_weight": 10.,
            "solver_unit_m": .001, "maximum_root_correction_m": .03}
BODY_FIELDS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def support_shift(lower_foot, settings=SETTINGS):
    from scipy import optimize, sparse

    height = np.asarray(lower_foot, dtype=float)
    if height.ndim != 1 or len(height) < 5 or not np.isfinite(height).all():
        raise ValueError("At least five finite lower-foot heights required")
    low, high, target = (settings[key] for key in (
        "minimum_lower_foot_z_m", "maximum_lower_foot_z_m", "target_lower_foot_z_m"))
    unit, weight = settings["solver_unit_m"], settings["second_difference_weight"]
    if not np.isfinite([low, high, target, unit, weight]).all() or not low < target < high < 0 or unit <= 0 or weight < 0:
        raise ValueError("Invalid fixed support-projection settings")
    count = len(height)
    second = sparse.diags([np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)],
                          [0, 1, 2], shape=(count - 2, count), format="csr")
    matrix = sparse.vstack([sparse.eye(count, format="csr"), weight * second], format="csr")
    rhs = np.concatenate([(target - height) / unit, np.zeros(count - 2)])
    solved = optimize.lsq_linear(matrix, rhs, bounds=((low - height) / unit, (high - height) / unit),
                                 tol=1e-10, lsmr_tol=1e-12, max_iter=200, lsmr_maxiter=2000)
    shift = solved.x * unit
    if (not solved.success or not np.isfinite(shift).all()
            or np.min(height + shift) < low - 1e-10 or np.max(height + shift) > high + 1e-10
            or np.max(np.abs(shift)) > settings["maximum_root_correction_m"]):
        raise ValueError("Projection did not satisfy its fixed geometric/size limits")
    return shift, {"status": int(solved.status), "iterations": solved.nit,
                   "cost": float(solved.cost), "optimality": float(solved.optimality),
                   "second_difference_rms_m": float(np.sqrt(np.mean(np.diff(shift, n=2) ** 2)))}


def shifted_states(qpos, qvel, shift, dt):
    qpos, qvel, shift = np.asarray(qpos), np.asarray(qvel), np.asarray(shift)
    count = len(shift)
    if (qpos.shape != (count, 29) or qvel.shape != (count, 28) or shift.shape != (count,)
            or count < 5 or not np.isfinite(dt) or dt <= 0
            or not all(np.isfinite(value).all() for value in (qpos, qvel, shift))):
        raise ValueError("Finite Q1 states, correction and positive time step required")
    poses, velocities = qpos.astype(float).copy(), qvel.astype(float).copy()
    poses[:, 2] += shift
    velocities[:-1, 2] = np.diff(poses[:, 2]) / dt
    velocities[-1, 2] = velocities[-2, 2]
    return poses, velocities


def geometry(model, feet, floor, qpos, qvel, body_names, trace_ids, targets, mj, oom):
    data = mj.MjData(model)
    body_ids = [model.body(name).id for name in body_names]
    fields = {key: [] for key in BODY_FIELDS}
    gaps, contacts, collision_count, collision_depth, residuals = [], [], [], [], []
    foot_ids = set(map(int, np.concatenate(feet)))
    worst = {"depth_m": 0., "frame": None, "bodies": None}
    velocity = np.empty((len(body_ids), 6))
    for frame, (pose, speed) in enumerate(zip(qpos, qvel)):
        if frame % 128 == 0:
            container_guard(oom)
        data.qpos[:], data.qvel[:] = pose, speed
        mj.mj_forward(model, data)
        gaps.append([bottom_point(model, data, ids, mj)[0] for ids in feet])
        foot_contacts, self_contacts, depth = 0, 0, 0.
        for contact in data.contact:
            if contact.efc_address < 0:
                continue
            if ((contact.geom1 == floor and contact.geom2 in foot_ids)
                    or (contact.geom2 == floor and contact.geom1 in foot_ids)):
                foot_contacts += 1
            elif floor not in (contact.geom1, contact.geom2) and contact.dist < 0:
                self_contacts += 1
                depth = max(depth, -float(contact.dist))
                if -contact.dist > worst["depth_m"]:
                    worst = {"depth_m": -float(contact.dist), "frame": frame,
                             "bodies": [model.body(int(model.geom_bodyid[g])).name
                                        for g in (contact.geom1, contact.geom2)]}
        contacts.append(foot_contacts)
        collision_count.append(self_contacts)
        collision_depth.append(depth)
        fields["body_pos_w"].append(data.xpos[body_ids].copy())
        fields["body_quat_w"].append(data.xquat[body_ids].copy())
        for index, bid in enumerate(body_ids):
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_XBODY, int(bid), velocity[index], 0)
        fields["body_lin_vel_w"].append(velocity[:, 3:].copy())
        fields["body_ang_vel_w"].append(velocity[:, :3].copy())
        residuals.append(np.linalg.norm(data.xpos[trace_ids] - targets[frame], axis=1))
    if data.time != 0:
        raise RuntimeError("Unexpected integration")
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    arrays.update(foot_gap_m=np.asarray(gaps), active_foot_contacts=np.asarray(contacts),
                  self_contact_count=np.asarray(collision_count), self_contact_depth_m=np.asarray(collision_depth))
    return arrays, {"lower_foot_z_m": describe(np.min(gaps, axis=1)),
                    "active_foot_contact_frame_fraction": float(np.mean(np.asarray(contacts) > 0)),
                    "self_collision_frame_fraction": float(np.mean(np.asarray(collision_count) > 0)),
                    "worst_self_collision": worst,
                    "absolute_target_position_error_p95_m": float(np.quantile(residuals, .95))}


def build(root, admission_path, diagnostic_path, audit_path, output):
    root, output = root.resolve(), output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a fresh, independent Q1 candidate directory")
    baseline = container_guard(reserve=896 * 1024 ** 2)
    import mujoco as mj
    import scipy

    admission_id, diagnostic_id, audit_id = map(identity, (admission_path, diagnostic_path, audit_path))
    admission = json.loads(checked(root, admission_id).read_text())
    diagnostic = json.loads(checked(root, diagnostic_id).read_text())
    audit = json.loads(checked(root, audit_id).read_text())
    if (diagnostic["status"] != "REFERENCE_INVERSE_CONTACT_DIAGNOSTIC_COMPLETED"
            or diagnostic["admission"] != admission_id
            or audit["status"] != "REFERENCE_DYNAMICS_SAVED_SUMMARY_AUDIT_PASSED"
            or audit["diagnostic"] != diagnostic_id or len(audit["clips"]) != 3):
        raise ValueError("Matching completed reference diagnostic and saved-array audit required")
    manifest_path = checked(root, admission["motion_manifest"])
    manifest = json.loads(manifest_path.read_text())
    rows = [row for row in manifest["motions"] if row.get("qualified_for_training_pilot")]
    if ([row["clip"] for row in rows] != [row["clip"] for row in diagnostic["clips"]]
            or [row["output"] for row in rows] != [row["reference"] for row in diagnostic["clips"]]):
        raise ValueError("Candidate must cover the same three diagnosed references")
    mapping_path = root / "script/supermimic_for_q1/retarget_mapping.json"
    mapping = json.loads(mapping_path.read_text())["body_map"]
    inputs = [identity(__file__), identity(Path(__file__).with_name("inspect_reference_dynamics.py")),
              admission_id, diagnostic_id, audit_id, admission["motion_manifest"],
              admission["assets"]["mjcf"], identity(mapping_path), *admission["source_files"]]
    for row in rows:
        inputs.extend([row["output"], row["kinematic_trace"]])
    for item in inputs:
        checked(root, item)
    model, feet, floor, caps, meshes = configured_model(root, admission, mj)
    inputs.extend(meshes)
    container_guard(baseline["oom_kill"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "dynamics").mkdir()
    clips, dynamic_clips = [], []
    for row in rows:
        with np.load(row["output"]["path"], allow_pickle=False) as saved:
            source = {key: saved[key].copy() for key in saved.files}
        with np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
            targets = trace["target_position_m_world"].copy()
            targets[:, :, 2] += row["diagnostics"]["global_ground_origin_shift_m"]
            trace_ids = [model.body(mapping[name]).id for name in trace["source_body_names"].tolist()]
        times = source["time_s"]
        if (float(source["fps"]) != 50 or not np.allclose(np.diff(times), .02, atol=1e-10, rtol=0)
                or times[0] != 0):
            raise ValueError("Keep native registered 50 Hz timing")
        names = source["body_names"].tolist()
        if names != admission["assets"]["tracked_bodies"]:
            raise ValueError("Tracked-body contract mismatch")
        before_arrays, before = geometry(model, feet, floor, source["qpos"], source["qvel"], names,
                                         trace_ids, targets, mj, baseline["oom_kill"])
        shift, solver = support_shift(before_arrays["foot_gap_m"].min(axis=1))
        qpos, qvel = shifted_states(source["qpos"], source["qvel"], shift, .02)
        after_arrays, after = geometry(model, feet, floor, qpos, qvel, names, trace_ids, targets,
                                       mj, baseline["oom_kill"])
        if (not np.array_equal(before_arrays["self_contact_count"], after_arrays["self_contact_count"])
                or not np.allclose(before_arrays["self_contact_depth_m"], after_arrays["self_contact_depth_m"],
                                   rtol=0, atol=1e-10)):
            raise ValueError("Vertical translation unexpectedly changed robot self collisions")
        candidate = {**source, "qpos": qpos, "qvel": qvel,
                     **{key: after_arrays[key] for key in BODY_FIELDS}}
        path = output / (row["clip"] + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **candidate)
        correction = output / (row["clip"] + "_projection.npz")
        with correction.open("xb") as stream:
            np.savez_compressed(stream, root_height_shift_m=shift,
                                parent_foot_gap_m=before_arrays["foot_gap_m"],
                                candidate_foot_gap_m=after_arrays["foot_gap_m"],
                                parent_self_contact_count=before_arrays["self_contact_count"],
                                candidate_self_contact_count=after_arrays["self_contact_count"])
        result = {"clip": row["clip"], "parent": row["output"], "candidate": identity(path),
                  "projection_arrays": identity(correction), "frames": len(times),
                  "solver": solver, "root_correction_m": describe(shift), "before": before, "after": after,
                  "qualified_for_training_pilot": False,
                  "reason": "Separate candidate pending independent export audit and matched physics evaluation"}
        clips.append(result)
        dynamic_clips.append(inspect_clip(model, feet, floor, caps,
                                         {"clip": row["clip"], "output": result["candidate"]},
                                         output / "dynamics", mj, baseline["oom_kill"]))
        print(json.dumps(result, allow_nan=False), flush=True)
    for item in inputs:
        checked(root, item)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected Torch import")
    final_health = container_guard(baseline["oom_kill"])
    dynamics = {"status": "REFERENCE_INVERSE_CONTACT_DIAGNOSTIC_COMPLETED",
                "admission": admission_id, "active_joint_names": admission["assets"]["active_joint_names"],
                "effective_effort_nm": caps.tolist(), "contact_protocol": "Q1_FOOT_PRIORITY_V2",
                "friction": 1., "clips": dynamic_clips, "source": identity(__file__),
                "new_physics_steps": 0, "storage_status": "LOCAL_ONLY"}
    with (output / "dynamics.json").open("x") as stream:
        json.dump(dynamics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report = {"status": "SUPPORT_HEIGHT_CANDIDATE_BUILT_NOT_ADMITTED", "source": identity(__file__),
              "admission": admission_id, "parent_diagnostic": diagnostic_id,
              "parent_diagnostic_audit": audit_id, "settings": SETTINGS, "clips": clips,
              "candidate_dynamics": identity(output / "dynamics.json"), "inputs": inputs,
              "inputs_verified_before_after": len(inputs), "health_before": baseline, "health_after": final_health,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "versions": {"mujoco": mj.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "new_inverse_calls": sum(row["inverse_calls"] for row in dynamic_clips),
              "torch_imported": False, "physics_tracking_validated": False, "storage_status": "LOCAL_ONLY",
              "limits": "Only root height and its vertical velocity change. Native timing, horizontal path and joint states stay fixed. Soft-contact inverse residuals are diagnostic, not an optimized force allocation or tracking proof. Existing R03 inputs remain untouched; no candidate is admitted for training."}
    with (output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "receipt": identity(output / "receipt.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "admission", "diagnostic", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.admission, args.diagnostic, args.audit, args.output)


if __name__ == "__main__":
    main()
