#!/usr/bin/env python3
"""Compare source, mapped-target and solved PHUMA foot geometry without integration."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def pose_min_z(geometries, position, rotation):
    lows = []
    for geom in geometries:
        center = position + rotation @ geom["position"]
        orientation = rotation @ geom["rotation"]
        if geom["kind"] == "sphere":
            low = center[2] - geom["size"][0]
        elif geom["kind"] == "box":
            low = center[2] - np.abs(orientation[2]) @ geom["size"]
        elif geom["kind"] == "mesh":
            low = center[2] + np.min(geom["vertices"] @ orientation[2])
        else:
            raise ValueError("Unsupported foot geometry: " + geom["kind"])
        lows.append(low)
    if not lows:
        raise ValueError("A foot must have collision geometry")
    return float(min(lows))


def curve_stats(values):
    values = np.asarray(values)
    return {"min_m": float(values.min()), "max_m": float(values.max()),
            "p95_m": float(np.quantile(values, .95)),
            "fraction_at_most_15mm": float(np.mean(values <= .015)),
            "fraction_at_most_30mm": float(np.mean(values <= .03))}


def analyze(args):
    cg = Path("/sys/fs/cgroup/memory")
    memory = []

    def health():
        usage = int((cg / "memory.usage_in_bytes").read_text())
        memory.append(usage)
        if usage >= 30 * 1024 ** 3 - 256 * 1024 ** 2:
            raise RuntimeError("Diagnostic memory headroom exhausted; training guard unchanged")
        return dict(line.split() for line in (cg / "memory.oom_control").read_text().splitlines())

    baseline = health()
    if 30 * 1024 ** 3 - memory[-1] < 1024 ** 3:
        raise RuntimeError("Need at least 1 GiB diagnostic headroom")
    phuma = args.root / "script/supermimic_q1_phuma"
    sys.path.insert(0, str(phuma))
    import mujoco
    from scipy.spatial.transform import Rotation
    from retarget import BODY_MAP, decode_g1, resample_qpos, source_feet
    from verify_reference import foot_bottom
    from contact_protocol import ContactCorrectedAssets

    manifest_path = args.motions / "manifest.json"
    checks_path = args.motions / "independent_check.json"
    manifest = json.loads(manifest_path.read_text())
    checks = json.loads(checks_path.read_text())
    spec = json.loads(args.training_spec.read_text())
    if (manifest["reference_protocol"] != "PHUMA_G1_30HZ_TO_Q1_50HZ_NATIVE_TIME_V2"
            or manifest["pilot"] or checks["input_manifest"] != identity(manifest_path)):
        raise ValueError("Require the unchanged full P03 conversion and its independent checks")
    for record in manifest["sources"] + [manifest["source_model"], manifest["source_receipt"]]:
        if identity(record["path"]) != record:
            raise ValueError("Bound source changed: " + record["path"])
    if Path(spec["motions"]) != args.motions.resolve():
        raise ValueError("Training spec belongs to other references")
    pinned = json.loads(Path(manifest["source_receipt"]["path"]).read_text())
    for record in pinned["files"] + spec["frozen_files"]:
        if identity(record["path"]) != record:
            raise ValueError("A pinned source, Q1 asset or training input changed")
    by_name = {row["clip"]: row for row in manifest["motions"]}
    independent = {row["clip"]: row for row in checks["clips"]}
    selected = [by_name[name] for name in args.clips]
    if not selected or len(set(args.clips)) != len(selected) or any(row["split"] != "train" for row in selected):
        raise ValueError("Use unique, explicitly selected training clips only")
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["q1_assets"]:
        raise ValueError("Q1 asset contract differs from conversion")
    source_model = mujoco.MjModel.from_xml_path(manifest["source_model"]["path"])
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(assets.model)
    health()
    source_ids = source_feet(source_model)
    source_sides, target_sides, local_geometries = [], [], []
    source_bodies, target_bodies = [], []
    for side in ("left", "right"):
        source_body = source_model.body(side + "_ankle_roll_link").id
        target_body = assets.model.body(side + "_ankle_roll_link").id
        source_bodies.append(source_body)
        target_bodies.append(target_body)
        members = []
        for gid in source_ids:
            body = int(source_model.geom_bodyid[gid])
            while body and body != source_body:
                body = int(source_model.body_parentid[body])
            if body == source_body:
                members.append(gid)
        source_sides.append(members)
        ids = [int(gid) for gid in assets.foot_geoms if assets.model.geom_bodyid[gid] == target_body]
        target_sides.append(ids)
        geometries = []
        kinds = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                 int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}
        for gid in ids:
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
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with zipfile.ZipFile(manifest["archive"]["path"]) as archive:
        for row in selected:
            name = row["clip"]
            for record in (row["output"], row["kinematic_trace"]):
                if identity(record["path"]) != record:
                    raise ValueError("A diagnostic reference/trace changed")
            raw = archive.read(row["phuma_source"]["member"])
            if len(raw) != row["phuma_source"]["bytes"] or hashlib.sha256(raw).hexdigest() != row["phuma_source"]["sha256"]:
                raise ValueError("Selected original PHUMA member differs")
            times, source_qpos = resample_qpos(source_model, decode_g1(raw))
            with np.load(row["output"]["path"], allow_pickle=False) as saved, np.load(row["kinematic_trace"]["path"], allow_pickle=False) as trace:
                if not np.array_equal(times, saved["time_s"]):
                    raise ValueError("Source and Q1 time axes differ")
                names = trace["source_body_names"].tolist()
                source_trace_ids = [source_model.body(BODY_MAP[key]).id for key in names]
                target_indices = [names.index(side + "_ankle") for side in ("left", "right")]
                source_low, target_low, desired_low, z_error = [], [], [], []
                rigid_error = source_fk_error = 0.
                shift = row["diagnostics"]["global_ground_origin_shift_m"]
                for frame, state in enumerate(source_qpos):
                    source_data.qpos[:] = state
                    mujoco.mj_fwdPosition(source_model, source_data)
                    source_fk_error = max(source_fk_error, float(np.abs(source_data.xpos[source_trace_ids] - trace["source_position_m_world"][frame]).max()))
                    target_data.qpos[:], target_data.qvel[:] = saved["qpos"][frame], saved["qvel"][frame]
                    mujoco.mj_fwdPosition(assets.model, target_data)
                    source_low.append([foot_bottom(source_model, source_data, ids) for ids in source_sides])
                    actual, desired, error = [], [], []
                    for side, (body, index) in enumerate(zip(target_bodies, target_indices)):
                        low = foot_bottom(assets.model, target_data, target_sides[side])
                        rigid = pose_min_z(local_geometries[side], target_data.xpos[body], target_data.xmat[body].reshape(3, 3))
                        rigid_error = max(rigid_error, abs(low - rigid))
                        target_position = trace["target_position_m_world"][frame, index].copy()
                        target_position[2] += shift
                        desired.append(pose_min_z(local_geometries[side], target_position, trace["target_rotation_matrix"][frame, index]))
                        actual.append(low)
                        error.append(float(target_data.xpos[body, 2] - target_position[2]))
                    target_low.append(actual)
                    desired_low.append(desired)
                    z_error.append(error)
                    if frame % 50 == 0:
                        if health()["oom_kill"] != baseline["oom_kill"]:
                            raise RuntimeError("New OOM during diagnostic")
            source_low, target_low, desired_low = map(np.asarray, (source_low, target_low, desired_low))
            source_curve, target_curve, desired_curve = (values.min(axis=1) for values in (source_low, target_low, desired_low))
            source_equivalent = manifest["scale"] * (source_curve - source_curve.min())
            desired_aligned = desired_curve - desired_curve.min()
            solved_aligned = target_curve - target_curve.min()
            recomputed = curve_stats(target_curve)
            old = independent[name]
            checks_passed = {"source_fk_matches_original_trace": source_fk_error < 1e-9,
                             "rigid_geometry_matches_mujoco": rigid_error < 1e-9,
                             "previous_screen_support_fraction_reproduced": recomputed["fraction_at_most_15mm"] == old["support_within_15mm_fraction"],
                             "previous_screen_max_gap_reproduced": abs(recomputed["max_m"] - old["foot_support_gap_max_m"]) < 1e-9}
            if not all(checks_passed.values()):
                raise ValueError("D02 geometric reconciliation failed: " + name)
            arrays = args.output / (name + ".npz")
            np.savez_compressed(arrays, time_s=times, source_foot_z_m=source_low,
                                solved_foot_z_m=target_low, desired_foot_z_m=desired_low,
                                ankle_task_vertical_error_m=np.asarray(z_error),
                                source_equivalent_grounded_m=source_equivalent,
                                desired_constant_grounded_m=desired_aligned, solved_constant_grounded_m=solved_aligned)
            result = {"clip": name, "split": row["split"], "bucket": row["bucket"],
                      "originally_qualified": old["qualified_for_training_pilot"],
                      "failed_gates": [key for key, passed in old["gates"].items() if not passed],
                      "source_original_world_m": curve_stats(source_curve),
                      "source_equivalent_constant_grounded_m": curve_stats(source_equivalent),
                      "desired_with_existing_shift_m": curve_stats(desired_curve),
                      "desired_constant_grounded_m": curve_stats(desired_aligned),
                      "solved_existing_reference_m": recomputed,
                      "solved_constant_grounded_m": curve_stats(solved_aligned),
                      "mapped_source_shape_rmse_m": float(np.sqrt(np.mean((desired_aligned - source_equivalent) ** 2))),
                      "solved_mapped_shape_rmse_m": float(np.sqrt(np.mean((solved_aligned - desired_aligned) ** 2))),
                      "ankle_task_vertical_error_abs_p95_m": float(np.quantile(np.abs(z_error), .95)),
                      "source_fk_max_abs_m": source_fk_error, "rigid_geometry_max_error_m": rigid_error,
                      "checks": checks_passed, "arrays": identity(arrays)}
            rows.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    final_health = health()
    if final_health["oom_kill"] != baseline["oom_kill"]:
        raise RuntimeError("New OOM during diagnostic")
    report = {"status": "PHUMA_SUPPORT_MAPPING_DIAGNOSIS_COMPLETED", "code_commit": args.code_commit,
              "training_spec": identity(args.training_spec), "training_code_commit": spec["code_commit"],
              "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "manifest": identity(manifest_path), "independent_screen": identity(checks_path),
              "selected_training_clips": args.clips, "scale": manifest["scale"], "rows": rows,
              "max_sampled_cgroup_bytes": max(memory), "oom_kill": int(final_health["oom_kill"]),
              "new_physics_integrations": 0, "new_IK_solves": 0, "new_policy_calls": 0, "new_optimizer_updates": 0,
              "references_modified": False, "source": identity(__file__), "storage_status": "LOCAL_ONLY",
              "limits": "Purposive training-only sample. Desired rigid foot poses need not be jointly reachable. "
                        "Constant grounding is per-curve diagnostic alignment, never a per-frame projection or a changed reference. "
                        "Source-equivalent lengths are scaled Q1 meters. This is not dynamic feasibility or policy evaluation."}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "clips": len(rows), "receipt": identity(args.output / "receipt.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--training-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--code-commit", required=True)
    analyze(parser.parse_args())
