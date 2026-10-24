#!/usr/bin/env python3
"""Separate mapped foot targets and rigid-link IK conflicts in fixed AMASS clips."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for directory in ("supermimic_for_q1", "q1", "supermimic_q1_phuma", "supermimic_q1_analysis"):
    sys.path.insert(0, str(HERE.parent / directory))

from analyze_phuma_support import curve_stats, pose_min_z
from prepare_amass_body import identity


def rigid_pair_bound(first, second, actual_first, actual_second, length):
    separation = np.linalg.norm(np.asarray(first) - second, axis=-1)
    observed_squared = np.sum((np.asarray(first) - actual_first) ** 2
                              + (np.asarray(second) - actual_second) ** 2, axis=-1)
    return (separation - length) ** 2 / 2., observed_squared


def worker(args):
    import mujoco
    from scipy.spatial.transform import Rotation
    from contact_protocol import ContactCorrectedAssets
    from runtime_guard import save
    from verify_reference import foot_bottom

    manifest_path, checked_path = args.motions / "manifest.json", args.checks / "reference_check.json"
    manifest, checked = (json.loads(path.read_text()) for path in (manifest_path, checked_path))
    if (checked["retarget_manifest"] != identity(manifest_path)
            or manifest["qualified_for_training"] or len(manifest["motions"]) != 4
            or any(row["split"] != "UNASSIGNED_DIAGNOSTIC_ONLY" for row in manifest["motions"])):
        raise ValueError("Use the four fixed, unadmitted B001 diagnostic clips")
    records = [identity(manifest_path), identity(checked_path), *manifest["sources"]]
    if any(identity(row["path"]) != row for row in records):
        raise ValueError("The completed AMASS conversion or source code changed")
    old = {row["clip"]: row for row in checked["clips"]}
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("Q1 asset contract changed")
    model, data = assets.model, mujoco.MjData(assets.model)
    mapping_path = HERE.parent / "supermimic_q1_phuma/retarget_mapping.json"
    mapping = json.loads(mapping_path.read_text())["body_map"]
    names = list(mapping)
    ids = [model.body(mapping[name]).id for name in names]
    geometries, foot_ids = [], []
    for side in ("left", "right"):
        ankle = model.body(side + "_ankle_roll_link").id
        selected = [int(gid) for gid in assets.foot_geoms if model.geom_bodyid[gid] == ankle]
        foot_ids.append(selected)
        local = []
        kinds = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                 int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}
        for gid in selected:
            geom = {"kind": kinds[int(model.geom_type[gid])], "position": model.geom_pos[gid].copy(),
                    "rotation": Rotation.from_quat(model.geom_quat[gid], scalar_first=True).as_matrix(),
                    "size": model.geom_size[gid].copy()}
            if geom["kind"] == "mesh":
                mesh = model.geom_dataid[gid]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                geom["vertices"] = model.mesh_vert[start:start + count].copy()
            local.append(geom)
        geometries.append(local)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in manifest["motions"]:
        for key in ("body_reference", "output", "kinematic_trace"):
            if identity(row[key]["path"]) != row[key]:
                raise ValueError("A diagnostic body, reference or trace changed")
        with np.load(row["output"]["path"], allow_pickle=False) as saved, np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
            targets = trace["target_position_m_world"].copy()
            targets[:, :, 2] += row["diagnostics"]["global_ground_origin_shift_m"]
            actual, desired_foot, solved_foot, rigid_error = [], [], [], 0.
            for frame, q in enumerate(saved["qpos"]):
                data.qpos[:] = q
                mujoco.mj_fwdPosition(model, data)
                actual.append(data.xpos[ids].copy())
                desired, solved = [], []
                for side, side_name in enumerate(("left", "right")):
                    index = names.index(side_name + "_ankle")
                    rotation = trace["target_rotation_matrix"][frame, index]
                    desired.append(pose_min_z(geometries[side], targets[frame, index], rotation))
                    low = foot_bottom(model, data, foot_ids[side])
                    solved.append(low)
                    body = ids[index]
                    rigid = pose_min_z(geometries[side], data.xpos[body], data.xmat[body].reshape(3, 3))
                    rigid_error = max(rigid_error, abs(low - rigid))
                desired_foot.append(desired)
                solved_foot.append(solved)
            actual, desired_foot, solved_foot = map(np.asarray, (actual, desired_foot, solved_foot))
            error = np.linalg.norm(actual - targets, axis=-1)
            desired_curve, solved_curve = desired_foot.min(-1), solved_foot.min(-1)
            desired_grounded = desired_curve - desired_curve.min()
            solved_grounded = solved_curve - solved_curve.min()
            conflicts = []
            for side in ("left", "right"):
                elbow, wrist = (names.index(side + "_" + part) for part in ("elbow", "wrist"))
                if (ids[elbow] == ids[wrist] or model.body_parentid[ids[wrist]] != ids[elbow]
                        or model.body_jntnum[ids[wrist]] != 0):
                    raise ValueError("Expected the actual distinct wrist body fixed to its elbow")
                length = float(np.linalg.norm(model.body_pos[ids[wrist]]))
                if np.abs(np.linalg.norm(actual[:, elbow] - actual[:, wrist], axis=-1) - length).max() > 1e-9:
                    raise ValueError("Solved elbow-to-wrist distance is not rigid")
                bound, observed = rigid_pair_bound(targets[:, elbow], targets[:, wrist],
                                                    actual[:, elbow], actual[:, wrist], length)
                if np.min(observed - bound) < -1e-10:
                    raise ValueError("Rigid-pair least-squares bound failed")
                conflicts.append({"side": side, "robot_elbow_body": mapping[side + "_elbow"],
                                  "robot_wrist_body": mapping[side + "_wrist"], "robot_fixed_distance_m": length,
                                  "target_separation_p95_m": float(np.quantile(np.linalg.norm(targets[:, elbow] - targets[:, wrist], axis=-1), .95)),
                                  "minimum_pair_mean_squared_residual_m2": float(bound.mean()),
                                  "observed_pair_mean_squared_residual_m2": float(observed.mean())})
            old_row = old[row["clip"]]
            checks = {"rigid_query_matches_mujoco": rigid_error < 1e-9,
                      "original_support_fraction_reproduced": float(np.mean(solved_curve <= .015)) == old_row["support_within_15mm_fraction"],
                      "original_max_gap_reproduced": abs(float(solved_curve.max()) - old_row["foot_support_gap_max_m"]) < 1e-9,
                      "original_global_residual_reproduced": abs(float(np.quantile(error, .95)) - old_row["IK_position_p95_m"]) < 1e-9,
                      "source_trace_body_order": trace["source_body_names"].tolist() == names}
            if not all(checks.values()):
                raise ValueError("Prior AMASS geometric evidence did not reproduce")
            path = args.output / (row["clip"] + ".npz")
            np.savez_compressed(path, time_s=saved["time_s"], body_names=np.asarray(names),
                                residual_norm_m=error, desired_foot_z_m=desired_foot, solved_foot_z_m=solved_foot,
                                desired_constant_grounded_m=desired_grounded, solved_constant_grounded_m=solved_grounded)
            result = {"clip": row["clip"], "source": row["source"]["relative_path"], "checks": checks,
                      "per_body_p95_residual_m": {name: float(np.quantile(error[:, i], .95)) for i, name in enumerate(names)},
                      "desired_original_shift_m": curve_stats(desired_curve),
                      "desired_constant_grounded_m": curve_stats(desired_grounded),
                      "solved_existing_reference_m": curve_stats(solved_curve),
                      "solved_constant_grounded_m": curve_stats(solved_grounded),
                      "solved_desired_curve_rmse_m": float(np.sqrt(np.mean((solved_grounded - desired_grounded) ** 2))),
                      "rigid_link_conflicts": conflicts, "rigid_query_max_error_m": rigid_error,
                      "arrays": identity(path)}
            rows.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    if any(identity(row["path"]) != row for row in records):
        raise ValueError("Diagnostic inputs or prior source code changed")
    report = {"status": "AMASS_FOUR_CLIP_MAPPING_DIAGNOSIS_COMPLETED", "captured_utc": datetime.now(timezone.utc).isoformat(),
              "retarget_manifest": records[0], "reference_check": records[1], "mapping": identity(mapping_path),
              "rows": rows, "source": identity(Path(__file__)),
              "geometry_helper": identity(HERE.parent / "supermimic_q1_analysis/analyze_phuma_support.py"),
              "new_ik_solves": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "references_modified": False, "storage_status": "LOCAL_ONLY",
              "limits": "Four purposive unassigned diagnostic clips, not an independent test. Foot target and solved geometry only; no original human sole/contact validation. Constant grounding is one diagnostic translation per entire curve, never a changed reference."}
    save(args.output / "receipt.json", report)


def guarded(args):
    from guard_phuma_batch import MemoryBoundBudget
    from runtime_guard import StorageBudget, guarded_run

    root, output = args.root.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new AMASS diagnostic directory")
    budget = MemoryBoundBudget(StorageBudget(output, args.motions.resolve(strict=True),
                                            100 * 1024 ** 3, 512 * 1024 ** 2, 512 * 1024 ** 2),
                               int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    output.mkdir()
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    os.nice(10)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--root", str(root),
               "--motions", str(args.motions.resolve()), "--checks", str(args.checks.resolve()),
               "--output", str(output / "diagnosis"), "--expected-oom-kill", str(args.expected_oom_kill)]
    record = guarded_run(command, root, budget, output / "guard.json", root / "logs" / (output.name + ".log"),
                         check_interval_s=.25, grace_s=5., stop_requested=lambda: requested,
                         progress_path=output / "diagnosis", maximum_idle_s=900.)
    print(json.dumps(record, allow_nan=False), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        raise SystemExit(guarded(args))
