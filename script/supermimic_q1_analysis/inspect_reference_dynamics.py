#!/usr/bin/env python3
"""Describe saved Q1 reference contact and inverse-force residuals, without stepping."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import xml.etree.ElementTree as ET

import numpy as np


FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
LIMITS = (
    "Inverse forces describe the exact saved pose/velocity and finite-difference acceleration "
    "under MuJoCo's soft contact model. They are not an optimized contact-force allocation, "
    "a proof that approximate tracking is impossible, or manufacturer hardware limits. "
    "The 1/10/30 mm proximity and 0.05 m/s slip thresholds are descriptive diagnostics. "
    "No reference, reward, control, training source, checkpoint or budget is changed."
)


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(root, expected):
    path = Path(expected["path"]).resolve()
    if not path.is_relative_to(root) or identity(path) != expected:
        raise ValueError(f"Input identity or project scope changed: {path}")
    return path


def derivatives(velocity, dt):
    velocity = np.asarray(velocity, dtype=np.float64)
    if (velocity.ndim != 2 or len(velocity) < 5 or not np.isfinite(dt)
            or dt <= 0 or not np.isfinite(velocity).all()):
        raise ValueError("Finite velocity matrix, positive dt and five frames required")
    return {
        "central3": (velocity[3:-1] - velocity[1:-3]) / (2 * dt),
        "central5": (-velocity[4:] + 8 * velocity[3:-1]
                     - 8 * velocity[1:-3] + velocity[:-4]) / (12 * dt),
    }


def describe(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def container_guard(baseline=None, reserve=128 * 1024 ** 2):
    base = Path("/sys/fs/cgroup/memory")
    current = int((base / "memory.usage_in_bytes").read_text())
    counters = dict(line.split() for line in (base / "memory.oom_control").read_text().splitlines())
    oom = int(counters["oom_kill"])
    if current + reserve >= 30 * 1024 ** 3 or baseline is not None and oom != baseline:
        raise RuntimeError("Diagnostic resource guard; training process is not signalled")
    return {"cgroup_memory_bytes": current, "oom_kill": oom}


def configured_model(root, admission, mj):
    assets = admission["assets"]
    if assets["contact_protocol"]["version"] != "Q1_FOOT_PRIORITY_V2":
        raise ValueError("Corrected contact admission required")
    xml = checked(root, assets["mjcf"])
    tree = ET.parse(xml)
    if tree.findall(".//include"):
        raise ValueError("Unregistered MJCF includes")
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_inputs = []
    for mesh in tree.findall("asset/mesh"):
        path = (xml.parent / meshdir / mesh.attrib["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("External mesh input")
        mesh_inputs.append(identity(path))
    model = mj.MjModel.from_xml_path(str(xml))
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError("Unexpected Q1 dimensions")
    if model.ntendon or model.opt.enableflags & mj.mjtEnableBit.mjENBL_INVDISCRETE:
        raise ValueError("This decomposition requires no tendons or discrete inverse mode")
    jids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_HINGE)
    names = [model.joint(int(j)).name for j in jids]
    qadr, vadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
    if names != assets["active_joint_names"] or not np.array_equal(vadr, np.arange(6, 28)):
        raise ValueError("Joint identity/layout mismatch")
    aids = []
    for jid in jids:
        found = np.flatnonzero((model.actuator_trnid[:, 0] == jid)
                              & (model.actuator_trntype == mj.mjtTrn.mjTRN_JOINT))
        if len(found) != 1:
            raise ValueError("Actuator mapping mismatch")
        aids.append(int(found[0]))
    aids = np.asarray(aids)
    caps = np.asarray(assets["effective_effort_nm"], dtype=float)
    if not np.allclose(model.actuator_gear[aids, 0], 1) or not np.allclose(
            model.actuator_ctrlrange[aids, 1] * assets["torque_fraction_of_model"], caps):
        raise ValueError("Effort contract mismatch")
    model.opt.timestep = assets["physics_dt"]
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    model.dof_damping[vadr] = 0
    floor = model.geom("floor").id
    if model.geom_type[floor] != mj.mjtGeom.mjGEOM_PLANE or abs(model.geom_pos[floor, 2]) > 1e-12:
        raise ValueError("Expected world-zero floor plane")
    feet = []
    for name in FOOT_BODIES:
        bid = model.body(name).id
        feet.append(np.flatnonzero((model.geom_bodyid == bid)
                                  & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))))
    foot_ids = np.concatenate(feet)
    if len(foot_ids) != len(assets["contact_protocol"]["foot_priorities"]):
        raise ValueError("Foot collision layout mismatch")
    model.geom_friction[foot_ids, 0] = 1
    model.geom_priority[foot_ids] = model.geom_priority[floor] + 1
    if model.geom_priority[foot_ids].tolist() != assets["contact_protocol"]["foot_priorities"]:
        raise ValueError("Contact priority mismatch")
    for i, aid in enumerate(aids):
        model.actuator_gainprm[aid, 0] = assets["kp"][i]
        model.actuator_biasprm[aid, :3] = [0, -assets["kp"][i], -assets["kd"][i]]
        model.actuator_biastype[aid] = mj.mjtBias.mjBIAS_AFFINE
        model.actuator_forcelimited[aid] = True
        model.actuator_forcerange[aid] = [-caps[i], caps[i]]
    if not np.array_equal(qadr, np.arange(7, 29)) or np.any(caps <= 0):
        raise ValueError("Invalid active positions or effort caps")
    return model, feet, floor, caps, mesh_inputs


def bottom_point(model, data, ids, mj):
    candidates = []
    for gid in ids:
        rotation = data.geom_xmat[gid].reshape(3, 3)
        center = data.geom_xpos[gid]
        kind = model.geom_type[gid]
        if kind == mj.mjtGeom.mjGEOM_SPHERE:
            point = center - np.array([0., 0., model.geom_size[gid, 0]])
        elif kind == mj.mjtGeom.mjGEOM_BOX:
            local = -np.where(rotation[2] >= 0, 1., -1.) * model.geom_size[gid]
            point = center + rotation @ local
        elif kind == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            point = center + rotation @ vertices[np.argmin(vertices @ rotation[2])]
        else:
            raise ValueError("Unsupported registered foot geometry")
        candidates.append((float(point[2]), point.copy(), int(model.geom_bodyid[gid])))
    return min(candidates, key=lambda row: row[0])


def inverse_with_check(model, data, acceleration, mj):
    data.qacc[:] = acceleration
    mj.mj_inverse(model, data)
    total = np.empty(model.nv)
    mj.mj_rne(model, data, 1, total)
    # RNE body forces exclude the joint armature term present in the mass matrix.
    total += model.dof_armature * data.qacc
    residual = total - data.qfrc_passive - data.qfrc_constraint
    error = float(np.max(np.abs(residual - data.qfrc_inverse)))
    if error > 1e-7 or not np.isfinite(data.qfrc_inverse).all() or data.time != 0:
        raise ValueError("Inverse-force decomposition, finite values or zero-time check failed")
    return data.qfrc_inverse.copy(), error


def summarize(arrays, caps, body_weight):
    gap, speed = arrays["foot_gap_m"], arrays["foot_tangent_speed_m_s"]
    report = {"analyzed_frames": len(gap), "foot_floor_contact_fraction": float(
        np.mean(arrays["active_foot_floor_contacts"] > 0)),
        "minimum_foot_gap_m": float(gap.min()), "lower_foot_gap_m": describe(gap.min(axis=1)),
        "proximity": {}, "inverse": {}}
    for distance in (.001, .01, .03):
        near = gap <= distance
        near_speed = speed[near]
        report["proximity"][str(distance)] = {
            "frame_any_foot_fraction": float(np.mean(near.any(axis=1))),
            "point_tangent_speed_m_s": describe(near_speed),
            "point_speed_above_0_05_fraction": float(np.mean(near_speed > .05)) if near_speed.size else None}
    for label in ("central3", "central5"):
        force = arrays[label + "_inverse"]
        ratio = np.abs(force[:, 6:]) / caps
        report["inverse"][label] = {
            "root_force_N": describe(np.linalg.norm(force[:, :3], axis=1)),
            "root_force_body_weights": describe(np.linalg.norm(force[:, :3], axis=1) / body_weight),
            "root_torque_Nm": describe(np.linalg.norm(force[:, 3:6], axis=1)),
            "active_effort_over_cap": describe(ratio),
            "frames_any_active_effort_over_cap_fraction": float(np.mean((ratio > 1).any(axis=1))),
            "per_joint_effort_over_cap_fraction": (ratio > 1).mean(axis=0).tolist(),
            "joint_acceleration_rad_s2": describe(np.abs(arrays[label + "_qacc"][:, 6:])),
            "inverse_rne_decomposition_max_error": float(arrays[label + "_decomposition_error"].max())}
    return report


def inspect_clip(model, feet, floor, caps, row, output, mj, oom):
    with np.load(row["output"]["path"], allow_pickle=False) as saved:
        qpos, qvel = saved["qpos"].astype(float), saved["qvel"].astype(float)
        times = saved["time_s"].astype(float)
    dt = float(np.diff(times).mean())
    if qpos.shape != (len(times), 29) or qvel.shape != (len(times), 28) or not np.isfinite(qpos).all():
        raise ValueError("Reference state layout/finite mismatch")
    if abs(dt - .02) > 1e-10 or not np.allclose(np.diff(times), dt, rtol=0, atol=1e-10):
        raise ValueError("Expected registered uniform 50 Hz reference")
    acceleration = derivatives(qvel, dt)
    indices = np.arange(2, len(times) - 2)
    count = len(indices)
    arrays = {"frame_index": indices, "time_s": times[indices],
              "foot_gap_m": np.empty((count, 2)), "foot_tangent_speed_m_s": np.empty((count, 2)),
              "active_foot_floor_contacts": np.empty(count, dtype=int)}
    for label, values in acceleration.items():
        arrays[label + "_qacc"] = values
        arrays[label + "_inverse"] = np.empty((count, 28))
        arrays[label + "_decomposition_error"] = np.empty(count)
    data = mj.MjData(model)
    flat_feet = set(map(int, np.concatenate(feet)))
    jac, jacrot = np.empty((3, model.nv)), np.empty((3, model.nv))
    for offset, frame in enumerate(indices):
        if offset % 128 == 0:
            container_guard(oom)
        data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
        mj.mj_forward(model, data)
        arrays["active_foot_floor_contacts"][offset] = sum(
            c.efc_address >= 0 and ((c.geom1 == floor and c.geom2 in flat_feet)
                                   or (c.geom2 == floor and c.geom1 in flat_feet))
            for c in data.contact)
        for side, ids in enumerate(feet):
            gap, point, bid = bottom_point(model, data, ids, mj)
            mj.mj_jac(model, data, jac, jacrot, point, bid)
            arrays["foot_gap_m"][offset, side] = gap
            arrays["foot_tangent_speed_m_s"][offset, side] = np.linalg.norm((jac @ qvel[frame])[:2])
        for label, values in acceleration.items():
            force, error = inverse_with_check(model, data, values[offset], mj)
            arrays[label + "_inverse"][offset] = force
            arrays[label + "_decomposition_error"][offset] = error
    target = output / (row["clip"] + ".npz")
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    body_weight = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
    return {"clip": row["clip"], "reference": row["output"], "source_frames": len(times),
            "excluded_boundary_frames": [0, 1, len(times) - 2, len(times) - 1],
            "dt_s": dt, "body_weight_N": body_weight, "raw_arrays": identity(target),
            **summarize(arrays, caps, body_weight), "inverse_calls": 2 * count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not output.is_relative_to(root / "data") or output.exists():
        raise ValueError("New project data output directory required")
    baseline = container_guard(reserve=384 * 1024 ** 2)
    import mujoco as mj
    admission_identity = identity(args.admission)
    admission = json.loads(checked(root, admission_identity).read_text())
    manifest_path = checked(root, admission["motion_manifest"])
    for item in admission["source_files"]:
        checked(root, item)
    manifest = json.loads(manifest_path.read_text())
    rows = [row for row in manifest["motions"] if row.get("qualified_for_training_pilot")]
    if [row["clip"] for row in rows] != ["001_walk_front_slow", "042_squat_shallow_repeated",
                                       "058_punch_front_alternating"]:
        raise ValueError("Unexpected qualified reference set")
    source_identity = identity(__file__)
    input_ids = [source_identity, admission_identity, admission["motion_manifest"], admission["assets"]["mjcf"],
                 *admission["source_files"], *(row["output"] for row in rows)]
    for item in input_ids:
        checked(root, item)
    model, feet, floor, caps, meshes = configured_model(root, admission, mj)
    input_ids += meshes
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for row in rows:
        result = inspect_clip(model, feet, floor, caps, row, output, mj, baseline["oom_kill"])
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    for item in input_ids:
        checked(root, item)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected Torch import")
    report = {"status": "REFERENCE_INVERSE_CONTACT_DIAGNOSTIC_COMPLETED",
              "source": source_identity, "admission": admission_identity,
              "inputs_verified_before_after": len(input_ids), "mesh_inputs": meshes,
              "versions": {"mujoco": mj.__version__, "numpy": np.__version__},
              "active_joint_names": admission["assets"]["active_joint_names"],
              "effective_effort_nm": caps.tolist(), "friction": 1.0,
              "contact_protocol": "Q1_FOOT_PRIORITY_V2", "clips": results,
              "acceleration_method": "central3/central5 derivatives of saved interval qvel; two boundary frames excluded per end",
              "health_before": baseline, "health_after": container_guard(baseline["oom_kill"]),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "torch_imported": False, "physics_tracking_validated": False,
              "storage_status": "LOCAL_ONLY", "limits": LIMITS}
    with (output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "receipt": identity(output / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
