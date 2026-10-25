#!/usr/bin/env python3
"""Compare frozen BONES source, mapped foot poses and Q1 export clearance."""

import argparse
import json
from pathlib import Path
import signal
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "supermimic_q1_phuma"):
    sys.path.insert(0, str(HERE.parent / directory))
from contact_protocol import ContactCorrectedAssets
from verify_reference import foot_bottom

from bones_verify_reference import read_source, source_foot_ids
from bounded_archive_io import bounded_identity
from guard_phuma_batch import MemoryBoundBudget
from relocated_io import migration_view
from runtime_guard import StorageBudget, guarded_run, save


def feet(model, gids):
    result = {}
    for side in ("left", "right"):
        bid = model.body(side + "_ankle_roll_link").id
        members = []
        for gid in gids:
            body = int(model.geom_bodyid[gid])
            while body and body != bid:
                body = int(model.body_parentid[body])
            if body == bid:
                members.append(int(gid))
        if not members:
            raise ValueError("A foot has no collision geometry")
        result[side] = (bid, members)
    return result


def rigid_foot(model, data, bid, gids):
    rotation = data.xmat[bid].reshape(3, 3)
    shapes = []
    for gid in gids:
        local_position = rotation.T @ (data.geom_xpos[gid] - data.xpos[bid])
        local_rotation = rotation.T @ data.geom_xmat[gid].reshape(3, 3)
        kind = model.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            shapes.append(("sphere", local_position, float(model.geom_size[gid, 0])))
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            points = signs * model.geom_size[gid]
            shapes.append(("points", points @ local_rotation.T + local_position, 0.))
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            points = model.mesh_vert[start:start + count].astype(float)
            shapes.append(("points", points @ local_rotation.T + local_position, 0.))
        else:
            raise ValueError("Unreviewed rigid foot collision shape")
    return shapes


def rigid_bottom(shapes, position, rotation):
    lows = []
    for kind, local, radius in shapes:
        if kind == "sphere":
            lows.append(float(position[2] + rotation[2] @ local - radius))
        else:
            lows.append(float((local @ rotation[2] + position[2]).min()))
    return min(lows)


def clearance(values):
    values = np.asarray(values, dtype=float)
    low = values.min(axis=1)
    return {"frames": len(low), "minimum_m": float(low.min()), "maximum_m": float(low.max()),
            "median_m": float(np.median(low)), "p95_m": float(np.quantile(low, .95)),
            "legacy_within_15mm_fraction": float(np.mean(low <= .015)),
            "within_minus1_to_15mm_fraction": float(np.mean((low >= -.001) & (low <= .015))),
            "penetration_over_1mm_fraction": float(np.mean(low < -.001)),
            "both_feet_above_30mm_fraction": float(np.mean(low > .03))}


def trajectory(model, qpos, groups):
    data = mujoco.MjData(model)
    values, pairs = [], {}
    for frame, state in enumerate(qpos):
        data.qpos[:] = state
        mujoco.mj_fwdPosition(model, data)
        values.append([foot_bottom(model, data, gids) for _, gids in groups.values()])
        for contact in data.contact:
            a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if a and b and contact.dist < 0:
                names = tuple(sorted((model.body(a).name, model.body(b).name)))
                if -contact.dist > pairs.get(names, {}).get("depth_m", 0.):
                    pairs[names] = {"bodies": names, "depth_m": -float(contact.dist), "frame": frame}
    worst = sorted(pairs.values(), key=lambda item: item["depth_m"], reverse=True)[:6]
    return np.asarray(values), worst


def budget(args):
    return MemoryBoundBudget(StorageBudget(args.output, args.root / "data/references/bones_seed/archives",
        100 * 1024 ** 3, 12 * 1024 ** 3, 250 * 1024 ** 3),
        int(28.75 * 1024 ** 3), args.expected_oom_kill)


def worker(args):
    import relocated_io
    relocated_io.identity = bounded_identity
    view = migration_view(args.root)
    manifest_path = args.batch / "retarget/manifest.json"
    check_path = args.batch / "reference_check.json"
    manifest, checked = json.loads(manifest_path.read_text()), json.loads(check_path.read_text())
    rows = manifest["motions"]
    if (manifest["status"] != "BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING"
            or checked["status"] != "BONES_INDEPENDENT_REFERENCE_CHECK_COMPLETED"
            or not 1 <= len(rows) <= 16 or any(row["split"] != "train" for row in rows)
            or [row["clip"] for row in rows] != [row["clip"] for row in checked["clips"]]):
        raise ValueError("Use the completed frozen train-only BONES batch")
    frozen = [bounded_identity(manifest_path), bounded_identity(check_path), manifest["source_model"],
              *manifest["sources"], *(item for row in rows for item in
                (row["source_csv"], row["output"], row["kinematic_trace"]))]
    versions = [bounded_identity(path) for path in (Path(__file__).resolve(),
        HERE / "bones_verify_reference.py", HERE.parent / "supermimic_q1_phuma/verify_reference.py")]
    if any(bounded_identity(item["path"]) != item for item in frozen):
        raise ValueError("A frozen real BONES input changed")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("The Q1 asset identity changed")
    source_model = mujoco.MjModel.from_xml_path(manifest["source_model"]["path"])
    source_feet = feet(source_model, source_foot_ids(source_model))
    target_feet = feet(assets.model, assets.foot_geoms)
    data = mujoco.MjData(assets.model)
    data.qpos[assets.qadr] = assets.q0
    mujoco.mj_fwdPosition(assets.model, data)
    shapes = {side: rigid_foot(assets.model, data, bid, gids)
              for side, (bid, gids) in target_feet.items()}
    reports = []
    for row, original in zip(rows, checked["clips"]):
        budget(args).check()
        times, native, fingerprint = read_source(Path(row["source_csv"]["path"]).read_bytes(),
            source_model, row["bones_source"]["move_duration_frames"])
        if fingerprint != original["normalized_native_numeric_sha256"]:
            raise ValueError("The independently decoded native identity changed")
        native_low, native_pairs = trajectory(source_model, native, source_feet)
        with np.load(row["output"]["path"], allow_pickle=False) as saved, \
                np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
            query, q = saved["time_s"], saved["qpos"].astype(float)
            sampled = np.empty((len(query), 36))
            for column in (*range(3), *range(7, 36)):
                sampled[:, column] = np.interp(query, times, native[:, column])
            sampled[:, 3:7] = Slerp(times, Rotation.from_quat(native[:, [4, 5, 6, 3]]))(query).as_quat()[:, [3, 0, 1, 2]]
            source_low, _ = trajectory(source_model, sampled, source_feet)
            before_low, _ = trajectory(assets.model, trace["qpos_before_ground_origin"], target_feet)
            final_low, final_pairs = trajectory(assets.model, q, target_feet)
            body_names = trace["source_body_names"].tolist()
            mapped, closure = [], 0.
            for frame, state in enumerate(q):
                data.qpos[:] = state
                mujoco.mj_fwdPosition(assets.model, data)
                lows = []
                for side, (bid, gids) in target_feet.items():
                    actual = rigid_bottom(shapes[side], data.xpos[bid], data.xmat[bid].reshape(3, 3))
                    closure = max(closure, abs(actual - foot_bottom(assets.model, data, gids)))
                    index = body_names.index(side + "_ankle")
                    lows.append(rigid_bottom(shapes[side], trace["target_position_m_world"][frame, index],
                                             trace["target_rotation_matrix"][frame, index]))
                mapped.append(lows)
            shift = row["diagnostics"]["global_ground_origin_shift_m"]
            exported = clearance(final_low)
            delta = max(abs(exported["minimum_m"] - original["foot_floor_min_m"]),
                        abs(exported["maximum_m"] - original["foot_support_gap_max_m"]),
                        abs(exported["legacy_within_15mm_fraction"] - original["support_within_15mm_fraction"]))
            shift_error = float(np.abs(before_low + shift - final_low).max())
            if closure > 1e-10 or delta > 1e-10 or shift_error > 2e-7:
                raise ValueError("Foot geometry closure, original metric or export shift differs")
            mapped = np.asarray(mapped)
            arrays = args.output / (row["clip"] + ".npz")
            np.savez_compressed(arrays, native_time_s=times, native_g1_foot_bottom_m=native_low,
                target_time_s=query, sampled_g1_foot_bottom_m=source_low,
                mapped_q1_foot_bottom_m=mapped, q1_before_origin_foot_bottom_m=before_low,
                q1_export_foot_bottom_m=final_low)
            report = {"clip": row["clip"], "source_group": row["bones_source"]["source_group"],
                "source_filename": row["bones_source"]["filename"], "split": "train",
                "g1_native_120hz": clearance(native_low), "g1_sampled_50hz": clearance(source_low),
                "mapped_q1_foot_pose": clearance(mapped), "q1_before_origin": clearance(before_low),
                "q1_export": exported, "global_origin_shift_m": shift,
                "rigid_geometry_closure_max_error_m": closure, "original_metric_max_error": delta,
                "global_shift_max_error_m": shift_error,
                "mapped_to_ik_foot_bottom_max_error_m": float(np.abs(mapped - before_low).max()),
                "g1_native_deepest_self_collision_pairs": native_pairs,
                "q1_export_deepest_self_collision_pairs": final_pairs, "arrays": bounded_identity(arrays)}
            reports.append(report)
            save(args.output / "progress.json", {"completed": len(reports), "requested": len(rows)})
    if any(bounded_identity(item["path"]) != item for item in frozen + versions):
        raise ValueError("Inputs or diagnostic code changed during the comparison")
    save(args.output / "summary.json", {"status": "BONES_TRAIN_SUPPORT_COMPARISON_COMPLETED",
        "inputs": frozen, "sources": versions, "clips": reports, "relocation": view.evidence(),
        "physics_steps": 0, "optimizer_updates": 0, "new_training_references": 0,
        "storage_after": budget(args).check(), "storage_status": "LOCAL_ONLY",
        "limits": "Read-only FK and rigid foot-goal diagnostics on eight frozen train sources. No IK optimization, dynamics tracking, new admission or proof of exact BONES generation XML identity."})
    print(json.dumps({"status": "BONES_TRAIN_SUPPORT_COMPARISON_COMPLETED", "clips": len(reports)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "batch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.root, args.batch, args.output = args.root.resolve(strict=True), args.batch.resolve(strict=True), args.output.resolve()
    base = args.root / "data/supermimic_for_q1"
    if (not args.batch.is_relative_to(base) or not args.output.is_relative_to(base)
            or args.output == base or args.output.is_relative_to(args.batch)):
        raise ValueError("Use an independent diagnostic output in the unified Q1 root")
    if args.worker:
        worker(args)
        return 0
    budget(args).memory()
    args.output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
        "--root", str(args.root), "--batch", str(args.batch), "--output", str(args.output),
        "--expected-oom-kill", str(args.expected_oom_kill)]
    result = guarded_run(command, args.root, budget(args), args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), stop_requested=lambda: requested,
        progress_path=args.output / "progress.json", maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
