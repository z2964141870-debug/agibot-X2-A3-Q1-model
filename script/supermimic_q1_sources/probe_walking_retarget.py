#!/usr/bin/env python3
"""Measure train-only PHUMA walking retarget sensitivity without admitting data."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget, guarded_run
from select_walking_holdout import EXCLUDE, WALK, lineage


SEED = 2026102511
BOUNDS = (2., 4., 8.)


def select_training(previous, aliases, reserved_directories):
    blocked = set(reserved_directories)
    for row in previous:
        info = lineage(row, aliases)
        if row["split"] != "train" and info is not None:
            blocked.add(info["directory_subject_candidate"])
    eligible = []
    for row in previous:
        info = lineage(row, aliases)
        if (row["split"] == "train" and row["bucket"] == "locomotion" and info
                and info["directory_subject_candidate"] not in blocked
                and WALK.search(PurePosixPath(info["original_motion_key"]).name)
                and not EXCLUDE.search(PurePosixPath(info["original_motion_key"]).name)):
            eligible.append((row, info))
    selected, used, hashes = [], set(), set()
    for row, info in sorted(eligible, key=lambda pair: hashlib.sha256(
            (str(SEED) + "/" + pair[0]["member"]).encode()).hexdigest()):
        directory = info["directory_subject_candidate"]
        if directory not in used and row["sha256"] not in hashes:
            selected.append((row, info))
            used.add(directory)
            hashes.add(row["sha256"])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Need four train-only walking directories after all held-out exclusions")
    return selected, {"eligible_chunks": len(eligible), "excluded_directory_candidates": sorted(blocked)}


def worker(args):
    import numpy as np

    root, base, output = args.root, args.root / "data/supermimic_for_q1", args.output / "probe"
    previous_paths = [base / name for name in ("PHUMA_P02_scan_20261024/selection.json",
        "PHUMA_P04_selection_20261024.json", "PHUMA_P04_additional_selection_20261024.json")]
    reserved = base / "PHUMA_P10_walking_selection_20261025.json"
    index = root / "data/references/humanml3d_metadata_9176e8fb_B_20261025/index.csv"
    inputs = [identity(path) for path in [*previous_paths, reserved, index]]
    previous_selections = [json.loads(path.read_text()) for path in previous_paths]
    with index.open(newline="") as stream:
        aliases = {row["new_name"]: row for row in csv.DictReader(stream)}
    holdout_contract = json.loads(reserved.read_text())["walking_holdout"]
    heldout = holdout_contract["proof"]["reserved_directory_candidates"]
    for record in inputs[:3] + [inputs[-1]]:
        if record not in holdout_contract["inputs"]:
            raise ValueError("Use the exact previous selections and metadata bound by the walking reservation")
    selected, proof = select_training([row for data in previous_selections for row in data["selected"]], aliases, heldout)
    output.mkdir()
    plan = {"status": "TRAIN_ONLY_WALKING_RETARGET_PROBE_SELECTED", "seed": SEED,
            "selected": [{"row": row, "lineage": info} for row, info in selected],
            "bounds_rad_s": BOUNDS, "inputs": inputs, "selection_proof": proof,
            "role": "TRAIN_DIAGNOSTIC_ONLY_NO_REFERENCE_ADMISSION", "source": identity(__file__)}
    with (output / "selection.json").open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")

    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    import retarget
    import mink
    import mujoco
    from prepare_expansion_batch import uncached_identity
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_analysis"))
    from analyze_phuma_support import pose_min_z, curve_stats
    from scipy.spatial.transform import Rotation

    assets = retarget.ContactCorrectedAssets(root)
    vendor = root / "data/references/phuma_source_2faf15b5"
    receipt_path = vendor / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    urdf = root / "data/references/q1/model_20261023_P01/q1_v3/urdf/q1_v3.urdf"
    metadata = {joint.attrib["name"]: float(joint.find("limit").attrib["velocity"])
                for joint in ET.parse(urdf).findall("joint") if joint.find("limit") is not None}
    if not all(metadata[name] >= max(BOUNDS) for name in assets.names):
        raise ValueError("Diagnostic bounds exceed the provided model's velocity metadata")
    frozen = [*inputs, identity(urdf), identity(receipt_path), *receipt["files"],
              *[identity(path) for path in (Path(retarget.__file__),
                HERE.parent / "supermimic_q1_phuma/retarget_control.json",
                HERE.parent / "supermimic_q1_phuma/retarget_mapping.json",
                HERE.parent / "q1/q1_retarget.py", HERE.parent / "q1/q1_sim.py",
                HERE.parent / "supermimic_q1_analysis/analyze_phuma_support.py")]]
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("An existing input changed")
    archive = previous_selections[0]["archive"]
    if uncached_identity(archive["path"]) != archive:
        raise ValueError("The fixed PHUMA archive changed")
    source_model = mujoco.MjModel.from_xml_path(str(vendor / "asset/humanoid_model/g1/custom.xml"))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    target_data = mujoco.MjData(assets.model)
    target_data.qpos[assets.qadr] = assets.q0
    mujoco.mj_forward(assets.model, target_data)
    source_leg = np.linalg.norm(source_data.xpos[source_model.body("pelvis").id]
                               - source_data.xpos[source_model.body("left_ankle_roll_link").id])
    target_leg = np.linalg.norm(target_data.xpos[assets.model.body("pelvis").id]
                               - target_data.xpos[assets.model.body("left_ankle_roll_link").id])
    scale = float(target_leg / source_leg)
    source_feet = retarget.source_feet(source_model)
    body_names = list(retarget.BODY_MAP)
    ankle_indices = [body_names.index(side + "_ankle") for side in ("left", "right")]
    local_geometries = []
    kinds = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
             int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}
    for side in ("left", "right"):
        body = assets.model.body(side + "_ankle_roll_link").id
        geometries = []
        for gid in assets.foot_geoms:
            if assets.model.geom_bodyid[gid] != body:
                continue
            geom = {"kind": kinds[int(assets.model.geom_type[gid])],
                    "position": assets.model.geom_pos[gid].copy(),
                    "rotation": Rotation.from_quat(assets.model.geom_quat[gid], scalar_first=True).as_matrix(),
                    "size": assets.model.geom_size[gid].copy()}
            if geom["kind"] == "mesh":
                mid = assets.model.geom_dataid[gid]
                start, count = assets.model.mesh_vertadr[mid], assets.model.mesh_vertnum[mid]
                geom["vertices"] = assets.model.mesh_vert[start:start + count].copy()
            geometries.append(geom)
        local_geometries.append(geometries)

    results = []
    with zipfile.ZipFile(archive["path"]) as package:
        for row, info in selected:
            raw = package.read(row["member"])
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Selected train source changed")
            motion = retarget.decode_g1(raw)
            reference, timing = retarget.source_reference(source_model, motion, body_names)
            times, source_q = retarget.resample_qpos(source_model, motion)
            low_source = []
            for state in source_q:
                source_data.qpos[:] = state
                mujoco.mj_fwdPosition(source_model, source_data)
                low_source.append(retarget.support_height(source_model, source_data, source_feet))
            low_source = scale * (np.asarray(low_source) - min(low_source))
            arrays, variants, baseline = {"time_s": times, "source_fixed_grounded_m": low_source}, [], None
            for bound in BOUNDS:
                solver, floor = retarget.grounded_solver(assets)
                if solver.names != body_names:
                    raise ValueError("Source body order differs from the fixed mapping")
                solver.sim.config = dict(solver.sim.config, reference_velocity_limit_rad_s=bound)
                solver.limits[1] = mink.VelocityLimit(solver.sim.model,
                    {name: bound * (1 - 1e-6) for name in solver.sim.names})
                retarget.initialize_first_pose(solver, reference, scale)
                _, q, targets, rotations, errors = solver.retarget(reference, scale)
                desired, solved, penetration = [], [], []
                for frame, state in enumerate(q):
                    target_data.qpos[:] = state
                    mujoco.mj_fwdPosition(assets.model, target_data)
                    solved.append(retarget.support_height(assets.model, target_data, assets.foot_geoms))
                    desired.append(min(pose_min_z(local_geometries[side], targets[frame, index],
                                                  rotations[frame, index]) for side, index in enumerate(ankle_indices)))
                    penetration.append(max([0.] + [-float(contact.dist) for contact in target_data.contact
                        if assets.model.geom_bodyid[contact.geom1] and assets.model.geom_bodyid[contact.geom2]]))
                solved = np.asarray(solved)
                desired = np.asarray(desired)
                solved -= solved.min()
                desired -= desired.min()
                velocity = np.diff(q[:, 7:], axis=0) / .02
                if np.abs(velocity).max() > bound + .00002:
                    raise ValueError("Diagnostic solver violated its explicit velocity bound")
                target_identity = np.asarray(targets).copy()
                if baseline is None:
                    baseline = (q[:, :7].copy(), target_identity, rotations.copy())
                if not (np.allclose(q[:, :7], baseline[0], atol=1e-10, rtol=0) and np.array_equal(target_identity, baseline[1])
                        and np.array_equal(rotations, baseline[2])):
                    raise ValueError("Velocity comparison changed root trajectory or body targets")
                tag = str(int(bound))
                arrays.update({"qpos_bound" + tag: q, "solved_fixed_grounded_bound" + tag + "_m": solved,
                               "desired_fixed_grounded_bound" + tag + "_m": desired})
                variants.append({"bound_rad_s": bound, "root_and_tasks_identical": True,
                    "joint_speed_max_rad_s": float(np.abs(velocity).max()),
                    "joint_speed_near_bound_fraction": float(np.mean(np.abs(velocity) >= .99 * bound)),
                    "solved_fixed_grounded": curve_stats(solved),
                    "desired_fixed_grounded": curve_stats(desired),
                    "solved_desired_shape_rmse_m": float(np.sqrt(np.mean((solved - desired) ** 2))),
                    "mapped_source_shape_rmse_m": float(np.sqrt(np.mean((desired - low_source) ** 2))),
                    "cartesian_IK_residual_p95_m": float(np.quantile(np.linalg.norm(errors, axis=-1), .95)),
                    "self_penetration_max_m": float(max(penetration)), "qualified_for_training": False})
            path = output / (hashlib.sha256(row["member"].encode()).hexdigest()[:16] + ".npz")
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite diagnostic arrays")
            np.savez_compressed(path, **arrays)
            result = {"member": row["member"], "original_motion_key": info["original_motion_key"],
                      "source_split": row["split"], "timing": timing, "variants": variants,
                      "source_fixed_grounded": curve_stats(low_source), "arrays": identity(path)}
            results.append(result)
            print(json.dumps(result), flush=True)
    if any(identity(record["path"]) != record for record in frozen):
        raise ValueError("A frozen input changed during the probe")
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("Diagnostic must not create a CUDA context")
    result = {"status": "TRAIN_ONLY_WALKING_RETARGET_SENSITIVITY_COMPLETED",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "selection": identity(output / "selection.json"),
              "rows": results, "scale": scale, "model_velocity_metadata_rad_s": {name: metadata[name] for name in assets.names},
              "frozen_inputs": frozen, "source": identity(__file__), "new_training_references": 0,
              "new_physics_steps": 0, "new_optimizer_updates": 0, "no_cuda_context": True,
              "storage_status": "LOCAL_ONLY",
              "limits": "Training-source sensitivity, not admission or hardware capability validation. Only the in-memory IK speed bound differs; root/body targets, native time and position limits are fixed. Single constant foot-curve alignment is diagnostic and never exported as a reference. No held-out sample used and no current training input changed."}
    with (output / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": identity(output / "receipt.json")}), flush=True)


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new train-only diagnostic directory")
    budget = MemoryBoundBudget(StorageBudget(args.output, args.root / "data/references/phuma_source_2faf15b5",
        100 * 1024 ** 3, 64 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
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
