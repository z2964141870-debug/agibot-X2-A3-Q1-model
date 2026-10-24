#!/usr/bin/env python3
"""Independently check P14 root translation, preserved anchors, FK and inverse arrays."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget


def support_point(model, data, ids, mj):
    candidates = []
    for gid in ids:
        rotation = data.geom_xmat[gid].reshape(3, 3)
        kind, size = model.geom_type[gid], model.geom_size[gid]
        if kind == mj.mjtGeom.mjGEOM_SPHERE:
            point = data.geom_xpos[gid] - np.array([0., 0., size[0]])
        elif kind == mj.mjtGeom.mjGEOM_BOX:
            local = np.where(rotation[2] >= 0, -size, size)
            point = data.geom_xpos[gid] + rotation.dot(local)
        elif kind == mj.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = model.mesh_vert[first:first + count].dot(rotation.T) + data.geom_xpos[gid]
            point = world[np.argmin(world[:, 2])]
        else:
            raise ValueError("Unsupported registered foot shape")
        candidates.append((point.copy(), int(model.geom_bodyid[gid])))
    return min(candidates, key=lambda item: item[0][2])


def audit(args):
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "supermimic_q1_phuma"))
    from retarget import ContactCorrectedAssets

    root, experiment = args.root.resolve(strict=True), args.experiment.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(experiment) or not experiment.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("A new audit file in the completed P14 experiment is required")
    receipt_path = experiment / "probe/receipt.json"
    report = json.loads(receipt_path.read_text())
    guard = json.loads((experiment / "guard.json").read_text())
    if (report["status"] != "TRAIN_ONLY_CONSTANT_ROOT_OFFSET_DIAGNOSTIC_COMPLETED"
            or guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"]
            or len(report["rows"]) != 8 or report["qualified_for_training"]):
        raise ValueError("Require a completed, reaped eight-group P14 diagnostic")
    budget = MemoryBoundBudget(StorageBudget(experiment,
        root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.check()
    inputs = [identity(receipt_path), identity(experiment / "guard.json"), *report["frozen_inputs"]]
    if any(identity(item["path"]) != item for item in inputs):
        raise ValueError("A fixed P14 input changed")
    assets = ContactCorrectedAssets(root)
    if assets.manifest() != report["q1_assets"]:
        raise ValueError("Q1 assets changed")
    model, data = assets.model, mj.MjData(assets.model)
    mapping = json.loads((here.parent / "supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"]
    names, bids = list(mapping), [model.body(name).id for name in mapping.values()]
    ankles = [names.index(side + "_ankle") for side in ("left", "right")]
    nonfeet = [i for i in range(len(names)) if i not in ankles]
    feet = [np.flatnonzero((model.geom_bodyid == model.body(side + "_ankle_roll_link").id)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))) for side in ("left", "right")]
    hips = [model.body(side + "_hip_pitch_link").id for side in ("left", "right")]
    radii = np.array([sum(np.linalg.norm(model.body(side + suffix).pos) for suffix in
        ("_hip_roll_link", "_hip_yaw_link", "_knee_link", "_ankle_pitch_link", "_ankle_roll_link"))
        for side in ("left", "right")])
    checks, frames, inverse_calls, rows, errors = 0, 0, 0, [], {}

    def equal(name, actual, expected, tolerance=1e-9):
        nonlocal checks
        error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f"Independent mismatch {name}: {error} > {tolerance}")
        errors[name] = max(errors.get(name, 0.), error)
        checks += 1

    for row in report["rows"]:
        if row["source_split"] != "train" or row["qualified_for_training"]:
            raise ValueError("Only the established train-only diagnostic is permitted")
        records = [row["prior_arrays"], row["prior_reach_arrays"]]
        if any(identity(item["path"]) != item for item in records):
            raise ValueError("An original array changed")
        inputs.extend(records)
        with np.load(records[0]["path"], allow_pickle=False) as saved:
            original = {key: saved[key].copy() for key in saved.files}
        with np.load(records[1]["path"], allow_pickle=False) as saved:
            reach = {key: saved[key].copy() for key in saved.files}
        old_q = original["qpos"]
        shift = float(original["common_ground_origin_shift_m"])
        root_rotation = Rotation.from_quat(old_q[:, 3:7], scalar_first=True).as_matrix()
        hip_centers = np.stack([old_q[:, :3] + root_rotation @ model.body(bid).pos for bid in hips], axis=1)
        targets = original["task_position_m"][:, ankles].copy()
        targets[:, :, 2] += shift
        difference = targets - hip_centers
        horizontal2 = np.sum(difference[:, :, :2] ** 2, axis=-1)
        extent = np.sqrt(np.maximum(0., radii ** 2 - horizontal2))
        lower = np.max(difference[:, :, 2] - extent, axis=1)
        upper = np.min(difference[:, :, 2] + extent, axis=1)
        valid = np.all(horizontal2 <= radii ** 2 + 1e-12, axis=1) & (lower <= upper + 1e-12)
        equal("prior_lower", lower, reach["requested_root_delta_lower_m"], 1e-10)
        equal("prior_upper", upper, reach["requested_root_delta_upper_m"], 1e-10)
        equal("prior_valid", valid, reach["requested_vertical_interval_valid"], 0)
        delta = min(0., float(upper.min()) - .005)
        admissible = bool(valid.all() and delta >= float(lower.max()) + .005 and abs(delta) <= .12)
        decision = row["decision"]
        equal("decision_lower", lower.max(), decision["common_lower_m"], 1e-10)
        equal("decision_upper", upper.min(), decision["common_upper_m"], 1e-10)
        equal("decision_delta", delta, decision["root_delta_z_m"], 1e-10)
        if (decision["status"] == "ACCEPTED_NECESSARY_BOUND_ONLY") != admissible:
            raise ValueError("Constant-offset admission disagrees with independently recomputed intervals")
        if row["status"] != "COMPLETED":
            if row["status"] == "REJECTED_NECESSARY_BOUND" and admissible:
                raise ValueError("An admissible necessary interval was incorrectly rejected")
            rows.append({"member": row["member"], "bound_rad_s": row["bound_rad_s"], "status": row["status"]})
            continue
        if not admissible:
            raise ValueError("An inadmissible interval reached IK")
        records = [row["arrays"], row["dynamics"]["raw_arrays"]]
        if any(identity(item["path"]) != item for item in records):
            raise ValueError("A P14 output changed")
        inputs.extend(records)
        with np.load(records[0]["path"], allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        with np.load(records[1]["path"], allow_pickle=False) as saved:
            dynamics = {key: saved[key].copy() for key in saved.files}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite P14 arrays")
        q, v, t = arrays["qpos"], arrays["qvel"], arrays["time_s"]
        expected_root = old_q[:, :7].copy()
        expected_root[:, 2] += delta
        equal("constant_translated_root", q[:, :7], expected_root, 1e-8)
        equal("original_root", arrays["original_root_qpos"], old_q[:, :7], 0)
        expected_root[:, 2] -= shift
        equal("solver_coordinate_root", arrays["requested_root_qpos"], expected_root, 1e-8)
        equal("saved_delta", arrays["root_delta_z_m"], delta, 1e-10)
        equal("time", t, original["time_s"], 0)
        equal("uniform_native_time", t, np.arange(len(t)) * .02, 1e-12)
        equal("world_foot_anchors", arrays["task_position_m"][:, ankles], original["task_position_m"][:, ankles], 0)
        shifted_tasks = original["task_position_m"].copy()
        shifted_tasks[:, nonfeet, 2] += delta
        equal("nonfeet_translated", arrays["task_position_m"], shifted_tasks, 1e-10)
        equal("original_task_positions", arrays["original_task_position_m"], original["task_position_m"], 0)
        equal("rotations", arrays["task_rotation_matrix"], original["task_rotation_matrix"], 0)
        equal("stance_labels", arrays["stance_mask"], original["stance_mask"], 0)
        equal("ground_origin", arrays["common_ground_origin_shift_m"], shift, 0)
        if (np.any(q[:, assets.qadr] < model.jnt_range[assets.jids, 0] - 1e-6)
                or np.any(q[:, assets.qadr] > model.jnt_range[assets.jids, 1] + 1e-6)):
            raise ValueError("P14 violated original joint position limits")
        velocity = np.empty_like(v)
        for frame in range(len(q) - 1):
            mj.mj_differentiatePos(model, velocity[frame], .02, q[frame], q[frame + 1])
        velocity[-1] = velocity[-2]
        equal("qvel", velocity, v, 1e-10)
        if np.abs(v[:, 6:]).max() > row["bound_rad_s"] + .00002:
            raise ValueError("The registered per-variant speed bound was exceeded")
        accelerations = {"central3": (v[3:-1] - v[1:-3]) / .04,
                         "central5": (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / .24}
        indices = np.arange(2, len(q) - 2)
        equal("inverse_indices", indices, dynamics["frame_index"], 0)
        for label in accelerations:
            equal("acceleration_" + label, accelerations[label], dynamics[label + "_qacc"], 1e-10)
        gaps, penetration, residual, nominal, speeds = [], [], [], [], []
        jac, jacrot = np.empty((3, model.nv)), np.empty((3, model.nv))
        for frame in range(len(q)):
            if frame % 100 == 0:
                budget.memory()
            data.qpos[:], data.qvel[:] = q[frame], v[frame]
            mj.mj_forward(model, data)
            points = [support_point(model, data, ids, mj) for ids in feet]
            gaps.append([point[0][2] for point in points])
            speed = []
            for point, body in points:
                mj.mj_jac(model, data, jac, jacrot, point, body)
                speed.append(float(np.sqrt(np.sum((jac @ v[frame])[:2] ** 2))))
            speeds.append(speed)
            penetration.append(max([0.] + [-float(c.dist) for c in data.contact
                if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
            expected, prior = arrays["task_position_m"][frame].copy(), original["task_position_m"][frame].copy()
            expected[:, 2] += shift
            prior[:, 2] += shift
            residual.append(data.xpos[bids] - expected)
            nominal.append(data.xpos[bids] - prior)
            if 2 <= frame < len(q) - 2:
                for label, values in accelerations.items():
                    data.qacc[:] = values[frame - 2]
                    mj.mj_inverse(model, data)
                    equal("fresh_inverse_" + label, data.qfrc_inverse,
                          dynamics[label + "_inverse"][frame - 2], 1e-7)
                    inverse_calls += 1
        gaps, penetration, residual, nominal, speeds = map(np.asarray, (gaps, penetration, residual, nominal, speeds))
        for name, actual in (("all_frame_foot_height_m", gaps), ("self_penetration_m", penetration),
                             ("task_residual_m", residual), ("nominal_residual_m", nominal)):
            equal("fresh_" + name, actual, arrays[name], 1e-9)
        equal("inverse_gaps", gaps[indices], dynamics["foot_gap_m"], 1e-9)
        equal("fresh_tangent_speeds", speeds[indices], dynamics["foot_tangent_speed_m_s"], 1e-9)
        low = gaps.min(axis=1)
        metrics = {"joint_speed_max_rad_s": np.abs(v[:, 6:]).max(), "support_within_15mm_fraction": (low <= .015).mean(),
            "support_gap_max_m": low.max(), "foot_floor_min_m": low.min(), "self_penetration_max_m": penetration.max(),
            "task_cartesian_residual_p95_m": np.quantile(np.linalg.norm(residual, axis=-1), .95),
            "nominal_cartesian_residual_p95_m": np.quantile(np.linalg.norm(nominal, axis=-1), .95)}
        for key, value in metrics.items():
            equal(key, value, row[key], 1e-9)
        subset = bool(low.min() >= -.001 and (low <= .015).mean() >= .9 and low.max() <= .03 and penetration.max() <= .003)
        equal("support_subset", subset, row["support_and_collision_subset_pass"], 0)
        equal("original_speed_gate", np.abs(v[:, 6:]).max() <= 2.00002, row["original_2rad_s_speed_gate"], 0)
        for distance in (.001, .01, .03):
            near = gaps[indices] <= distance
            entry = row["dynamics"]["proximity"][str(distance)]
            equal("near_fraction_" + str(distance), near.any(axis=1).mean(), entry["frame_any_foot_fraction"], 0)
            if near.any():
                equal("slip_fraction_" + str(distance), (speeds[indices][near] > .05).mean(), entry["point_speed_above_0_05_fraction"], 0)
        weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)
        for label in accelerations:
            force = dynamics[label + "_inverse"]
            entry = row["dynamics"]["inverse"][label]
            equal("root_force_" + label, np.median(np.linalg.norm(force[:, :3], axis=1) / weight), entry["root_force_body_weights"]["p50"])
            equal("effort_fraction_" + label, ((np.abs(force[:, 6:]) / assets.effort) > 1).any(axis=1).mean(), entry["frames_any_active_effort_over_cap_fraction"], 0)
        frames += len(q)
        rows.append({"member": row["member"], "bound_rad_s": row["bound_rad_s"],
                     "status": "SAVED_TRANSFORM_AND_FRESH_FK_INVERSE_MATCH", "frames": len(q)})
    if any(identity(item["path"]) != item for item in inputs):
        raise ValueError("A fixed input changed during independent checking")
    result = {"status": "INDEPENDENT_CONSTANT_ROOT_OFFSET_AUDIT_COMPLETED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "receipt": identity(receipt_path),
        "source": identity(__file__), "numeric_checks": checks, "max_abs_error_by_check": errors,
        "rows": rows, "fresh_fk_frames": frames, "fresh_inverse_calls": inverse_calls,
        "all_inputs_unchanged": True, "health_after": budget.check(), "storage_status": "LOCAL_ONLY",
        "new_physics_steps": 0, "new_optimizer_updates": 0, "new_training_references": 0,
        "limits": "Independent interval, target-transform, geometry, slip, derivative and inverse checks; no new IK or policy dynamics rollout. No reference admission."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    audit(parser.parse_args())
