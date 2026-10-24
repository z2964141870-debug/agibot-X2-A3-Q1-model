#!/usr/bin/env python3
"""Compare fixed train walking sources with stance targets and native IK avoidance."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

HERE = Path(__file__).resolve().parent
from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from probe_walking_retarget import select_training
from runtime_guard import StorageBudget, guarded_run
from walking_contact_targets import contact_targets, native_collision_limit, stance_mask


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fixed_selection(plan, previous, aliases, reserved):
    selected, proof = select_training(previous, aliases, reserved)
    expected = [{"row": row, "lineage": info} for row, info in selected]
    if (plan["role"] != "TRAIN_DIAGNOSTIC_ONLY_NO_REFERENCE_ADMISSION"
            or plan["selected"] != expected or plan["selection_proof"] != proof):
        raise ValueError("Use exactly the P11 train selection and held-out exclusions")
    return selected


def geometry(model, ids, mj, Rotation):
    kinds = {int(mj.mjtGeom.mjGEOM_SPHERE): "sphere", int(mj.mjtGeom.mjGEOM_BOX): "box",
             int(mj.mjtGeom.mjGEOM_MESH): "mesh"}
    result = []
    for gid in ids:
        entry = {"kind": kinds[int(model.geom_type[gid])], "position": model.geom_pos[gid].copy(),
                 "rotation": Rotation.from_quat(model.geom_quat[gid], scalar_first=True).as_matrix(),
                 "size": model.geom_size[gid].copy()}
        if entry["kind"] == "mesh":
            mesh = model.geom_dataid[gid]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            entry["vertices"] = model.mesh_vert[first:first + count].copy()
        result.append(entry)
    return result


def solve(solver, roots, positions, rotations, start, bound, collision, mj, mink, np, Rotation):
    configuration, model = solver.configuration, solver.sim.model
    configuration.update(start.copy())
    limit = native_collision_limit(mink, model, .02) if collision else None
    tasks = [task for _, task in solver.tasks] + [solver.posture]

    def targets(frame):
        q = configuration.data.qpos.copy()
        q[:7] = roots[frame]
        configuration.update(q)
        solver.root_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(Rotation.from_quat(q[3:7], scalar_first=True).as_matrix()), q[:3]))
        for index, task in solver.tasks:
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations[frame, index]), positions[frame, index]))

    targets(0)
    initial_limits = [solver.limits[0]] + ([limit] if limit else [])
    for _ in range(32):
        velocity = mink.solve_ik(configuration, tasks, .02, solver.config["solver"],
            damping=solver.config["damping"], safety_break=True, limits=initial_limits,
            constraints=[solver.root_task])
        configuration.integrate_inplace(velocity, .02)
        if np.linalg.norm(velocity) < 1e-5:
            break
    iterations = solver.config["iterations_per_frame"]
    dt = .02 / iterations
    if limit:
        limit.gain = .85 * dt
    limits = solver.limits + ([limit] if limit else [])
    output = []
    for frame in range(len(roots)):
        targets(frame)
        for _ in range(iterations):
            velocity = mink.solve_ik(configuration, tasks, dt, solver.config["solver"],
                damping=solver.config["damping"], safety_break=True, limits=limits,
                constraints=[solver.root_task])
            configuration.integrate_inplace(velocity, dt)
        q = configuration.data.qpos.copy()
        values = q[solver.sim.q_indices]
        if (not np.isfinite(q).all() or np.any(values < solver.sim.lower - 1e-6)
                or np.any(values > solver.sim.upper + 1e-6)):
            raise ValueError("Contact IK violated finite values or joint limits")
        output.append(q)
    output = np.asarray(output)
    if (np.abs(output[:, :7] - roots).max() > 1e-8
            or np.abs(np.diff(output[:, 7:], axis=0) / .02).max() > bound + .00002):
        raise ValueError("Contact IK changed root or violated the fixed speed bound")
    return output, len(limit.geom_id_pairs) if limit else 0


def worker(args):
    import numpy as np
    import mujoco as mj
    import mink
    from scipy.spatial.transform import Rotation

    if importlib.metadata.version("mink") != "1.2.0":
        raise ValueError("This diagnostic binds installed Mink 1.2.0 collision time scaling")
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    import retarget
    from prepare_expansion_batch import uncached_identity
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_analysis"))
    from analyze_phuma_support import pose_min_z
    from inspect_reference_dynamics import bottom_point, inspect_clip

    base = args.root / "data/supermimic_for_q1"
    prior = base / "PHUMA_P11_walking_velocity_probe_20261025/probe"
    plan, receipt = (json.loads((prior / name).read_text()) for name in ("selection.json", "receipt.json"))
    if (receipt["status"] != "TRAIN_ONLY_WALKING_RETARGET_SENSITIVITY_COMPLETED"
            or receipt["selection"] != identity(prior / "selection.json")):
        raise ValueError("Require completed, unchanged P11 diagnostic")
    previous_paths = [base / name for name in ("PHUMA_P02_scan_20261024/selection.json",
        "PHUMA_P04_selection_20261024.json", "PHUMA_P04_additional_selection_20261024.json")]
    previous = [json.loads(path.read_text()) for path in previous_paths]
    reserved_path = base / "PHUMA_P10_walking_selection_20261025.json"
    index = args.root / "data/references/humanml3d_metadata_9176e8fb_B_20261025/index.csv"
    with index.open(newline="") as stream:
        aliases = {row["new_name"]: row for row in csv.DictReader(stream)}
    reserved = json.loads(reserved_path.read_text())["walking_holdout"]["proof"]["reserved_directory_candidates"]
    selected = fixed_selection(plan, [row for data in previous for row in data["selected"]], aliases, reserved)
    frozen = [*receipt["frozen_inputs"], identity(prior / "selection.json"), identity(prior / "receipt.json"),
        *(row["arrays"] for row in receipt["rows"]), identity(__file__), identity(HERE / "walking_contact_targets.py"),
        identity(HERE.parent / "supermimic_q1_analysis/inspect_reference_dynamics.py")]
    frozen = list({record["path"]: record for record in frozen}.values())
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("A fixed P11 source, producer or diagnostic input changed")
    archive = previous[0]["archive"]
    if uncached_identity(archive["path"]) != archive:
        raise ValueError("PHUMA archive changed")
    output = args.output / "probe"
    output.mkdir()
    (output / "dynamics").mkdir()
    save_json(output / "selection.json", {"status": "P12_TRAIN_SELECTION_RECONFIRMED", "selected": plan["selected"],
        "prior_selection": identity(prior / "selection.json"), "role": "TRAIN_DIAGNOSTIC_ONLY_NO_REFERENCE_ADMISSION",
        "bounds_rad_s": [2., 8.], "variants": ["baseline", "stance", "stance_collision"],
        "stance_definition": "Mapped sole within 15mm of lower foot, isolated gaps closed; interior runs shorter than 4 frames removed. Walking-only hypothesis, not ground-truth contact.",
        "collision_time_scaling": "Mink1.2.0 gain=0.85*dt gives h=0.85*(distance-clearance), matching upstream displacement correction; 2mm wrist/trunk-leg clearance, 50mm detection",
        "source": identity(__file__)})
    assets = retarget.ContactCorrectedAssets(args.root)
    model, data = assets.model, mj.MjData(assets.model)
    feet = [np.array([gid for gid in assets.foot_geoms if model.geom_bodyid[gid]
                     == model.body(side + "_ankle_roll_link").id], dtype=int) for side in ("left", "right")]
    shapes = [geometry(model, ids, mj, Rotation) for ids in feet]
    source_model = mj.MjModel.from_xml_path(str(args.root / "data/references/phuma_source_2faf15b5/asset/humanoid_model/g1/custom.xml"))
    by_member = {row["member"]: row for row in receipt["rows"]}
    results = []
    with zipfile.ZipFile(archive["path"]) as package:
        for row, lineage in selected:
            raw = package.read(row["member"])
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Fixed train source bytes changed")
            reference, timing = retarget.source_reference(source_model, retarget.decode_g1(raw), list(retarget.BODY_MAP))
            ankles = [reference.names.index(side + "_ankle") for side in ("left", "right")]
            with np.load(by_member[row["member"]]["arrays"]["path"], allow_pickle=False) as saved:
                baseline = {bound: saved["qpos_bound" + str(bound)].copy() for bound in (2, 8)}
            per_clip = {"member": row["member"], "original_motion_key": lineage["original_motion_key"],
                        "source_split": row["split"], "timing": timing, "variants": []}
            for bound in (2, 8):
                solver, _ = retarget.grounded_solver(assets)
                solver.sim.config = dict(solver.sim.config, reference_velocity_limit_rad_s=bound)
                solver.limits[1] = mink.VelocityLimit(solver.sim.model,
                    {name: bound * (1 - 1e-6) for name in solver.sim.names})
                retarget.initialize_first_pose(solver, reference, receipt["scale"])
                _, original_q, targets, rotations, _ = solver.retarget(reference, receipt["scale"])
                repeat_error = float(np.abs(original_q - baseline[bound]).max())
                if repeat_error > 1e-8:
                    raise ValueError("Repeated baseline differs from immutable P11: " + str(repeat_error))
                lower = []
                for state in baseline[bound]:
                    data.qpos[:] = state
                    mj.mj_fwdPosition(model, data)
                    lower.append(retarget.support_height(model, data, assets.foot_geoms))
                floor = min(lower)
                desired_sole = np.asarray([[pose_min_z(shapes[side], targets[i, ankle], rotations[i, ankle])
                    for side, ankle in enumerate(ankles)] for i in range(len(reference.times))])
                mask = stance_mask(desired_sole)
                changed_p, changed_r, events = contact_targets(targets[:, ankles], rotations[:, ankles], mask,
                    shapes, floor, solver.robot_rotations[ankles])
                contact_p, contact_r = targets.copy(), rotations.copy()
                contact_p[:, ankles], contact_r[:, ankles] = changed_p, changed_r
                for variant in ("baseline", "stance", "stance_collision"):
                    tag = hashlib.sha256(row["member"].encode()).hexdigest()[:16] + "_v" + str(bound) + "_" + variant
                    result = {"variant": variant, "bound_rad_s": bound, "status": "FAILED",
                              "qualified_for_training": False, "baseline_repeat_max_abs": repeat_error}
                    try:
                        if variant == "baseline":
                            q, pair_count = original_q.copy(), 0
                            task_p, task_r = targets, rotations
                        else:
                            task_p, task_r = contact_p, contact_r
                            q, pair_count = solve(solver, original_q[:, :7], task_p, task_r,
                                original_q[0], bound, variant == "stance_collision", mj, mink, np, Rotation)
                        original_root = q[:, :7].copy()
                        q[:, 2] -= floor
                        qvel = np.empty((len(q), model.nv))
                        for frame in range(len(q) - 1):
                            mj.mj_differentiatePos(model, qvel[frame], .02, q[frame], q[frame + 1])
                        qvel[-1] = qvel[-2]
                        residual, nominal_residual, self_collision, foot_height = [], [], [], []
                        for frame, state in enumerate(q):
                            data.qpos[:] = state
                            mj.mj_fwdPosition(model, data)
                            shifted, nominal = task_p[frame].copy(), targets[frame].copy()
                            shifted[:, 2] -= floor
                            nominal[:, 2] -= floor
                            residual.append(data.xpos[solver.bodies] - shifted)
                            nominal_residual.append(data.xpos[solver.bodies] - nominal)
                            foot_height.append([bottom_point(model, data, ids, mj)[0] for ids in feet])
                            self_collision.append(max([0.] + [-float(c.dist) for c in data.contact
                                if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
                        path = output / (tag + ".npz")
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, time_s=reference.times, qpos=q, qvel=qvel,
                                task_position_m=task_p, task_rotation_matrix=task_r,
                                original_task_position_m=targets, original_task_rotation_matrix=rotations,
                                task_residual_m=np.asarray(residual), nominal_residual_m=np.asarray(nominal_residual),
                                self_penetration_m=np.asarray(self_collision), stance_mask=mask,
                                all_frame_foot_height_m=np.asarray(foot_height),
                                common_ground_origin_shift_m=-floor)
                        dynamics = inspect_clip(model, feet, model.geom("floor").id, assets.effort,
                            {"clip": tag, "output": identity(path)}, output / "dynamics", mj, args.expected_oom_kill)
                        gap = np.asarray(foot_height).min(axis=1)
                        result.update(status="COMPLETED", arrays=identity(path), dynamics=dynamics,
                            root_max_abs_m_or_quat=float(np.abs(original_root - baseline[bound][:, :7]).max()),
                            ground_origin_shift_m=-floor, collision_pair_count=pair_count,
                            inferred_stance_fraction=mask.mean(axis=0).tolist(),
                            stance_events=events if variant != "baseline" else [],
                            changed_ankle_target_max_m=float(np.linalg.norm(task_p[:, ankles] - targets[:, ankles], axis=-1).max()),
                            nominal_cartesian_residual_p95_m=float(np.quantile(np.linalg.norm(nominal_residual, axis=-1), .95)),
                            task_cartesian_residual_p95_m=float(np.quantile(np.linalg.norm(residual, axis=-1), .95)),
                            joint_speed_max_rad_s=float(np.abs(qvel[:, 6:]).max()),
                            self_penetration_max_m=float(max(self_collision)),
                            support_within_15mm_fraction=float(np.mean(gap <= .015)),
                            support_gap_max_m=float(gap.max()), foot_floor_min_m=float(gap.min()),
                            original_2rad_s_speed_gate=bool(np.abs(qvel[:, 6:]).max() <= 2.00002),
                            support_and_collision_subset_pass=bool(gap.min() >= -.001 and (gap <= .015).mean() >= .9
                                and gap.max() <= .03 and max(self_collision) <= .003))
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    per_clip["variants"].append(result)
                    print(json.dumps({"source": row["member"], **result}, allow_nan=False), flush=True)
            results.append(per_clip)
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("An immutable input changed during the contact diagnostic")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("No CUDA context is permitted")
    save_json(output / "receipt.json", {"status": "TRAIN_ONLY_WALKING_CONTACT_DIAGNOSTIC_COMPLETED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "selection": identity(output / "selection.json"),
        "rows": results, "frozen_inputs": frozen, "new_training_references": 0,
        "new_physics_steps": 0, "new_optimizer_updates": 0, "no_cuda_context": True,
        "versions": {"mink": importlib.metadata.version("mink"), "mujoco": mj.__version__},
        "storage_status": "LOCAL_ONLY", "limits": "Fixed train-only geometric and inverse-dynamics diagnosis; all variants remain ineligible. Root/time/position limits retained. 8rad/s variants fail the original speed gate. Inferred stance labels and target corrections are hypotheses, not contact truth or dynamic tracking."})
    print(json.dumps({"receipt": identity(output / "receipt.json")}), flush=True)


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new train-only diagnostic directory")
    budget = MemoryBoundBudget(StorageBudget(args.output, args.root / "data/references/phuma_source_2faf15b5",
        100 * 1024 ** 3, 128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stop = False

    def requested(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
               "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    result = guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.25, grace_s=5.,
        stop_requested=lambda: stop, progress_path=args.output / "probe", maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
