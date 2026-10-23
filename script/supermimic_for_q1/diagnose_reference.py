#!/usr/bin/env python3
"""Inspect preserved Q1 IK outputs without changing data or stepping physics."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from prepare_motion import SOURCE_BODIES, SOURCE_XML_SHA, source_reference, support_height
from supermimic.q1_assets import Q1Assets, identity


def summary(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def foot_geoms(model, ankles):
    roots = {model.body(name).id for name in ankles}
    ids = []
    for gid in range(model.ngeom):
        body = int(model.geom_bodyid[gid])
        while body and body not in roots:
            body = int(model.body_parentid[body])
        if body in roots and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            ids.append(gid)
    return np.asarray(ids)


def canonical(model, joint_values=None):
    data = mujoco.MjData(model)
    if joint_values is not None:
        data.qpos[7:] = joint_values
    mujoco.mj_forward(model, data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--source-vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = Q1Assets(args.root)
    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_xml = args.source_vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml"
    if identity(source_xml)["sha256"] != SOURCE_XML_SHA:
        raise ValueError("Source model changed")
    source = mujoco.MjModel.from_xml_path(str(source_xml))
    source_data, target_data = canonical(source), canonical(assets.model, assets.q0)
    source_feet = foot_geoms(source, ("left_ankle_roll_Link", "right_ankle_roll_Link"))
    names = list(SOURCE_BODIES)
    mapping = json.loads((Path(__file__).parent / "retarget_mapping.json").read_text())["body_map"]
    source_ids = np.array([source.body(SOURCE_BODIES[n]).id for n in names])
    target_ids = np.array([assets.model.body(mapping[n]).id for n in names])
    report = {"status": "DIAGNOSTIC_ONLY", "physics_steps": 0, "optimizer_updates": 0,
              "input_manifest": identity(manifest_path), "source_xml": identity(source_xml),
              "q1_assets": assets.manifest(), "source_foot_geometry": [],
              "source_neutral_root": source_data.qpos[:7].tolist(),
              "q1_neutral_root": target_data.qpos[:7].tolist(),
              "q1_neutral_foot_min_z": support_height(assets.model, target_data, assets.foot_geoms),
              "limb_lengths": {}, "clips": [], "storage_status": "LOCAL_ONLY"}
    for gid in source_feet:
        report["source_foot_geometry"].append({"body": source.body(int(source.geom_bodyid[gid])).name,
                                               "type": int(source.geom_type[gid]),
                                               "position": source_data.geom_xpos[gid].tolist(),
                                               "size": source.geom_size[gid].tolist()})
    try:
        report["source_neutral_foot_min_z"] = support_height(source, source_data, source_feet)
    except ValueError as exc:
        report["source_foot_support_error"] = str(exc)
    for side in ("left", "right"):
        for proximal, distal in (("hip", "knee"), ("knee", "ankle"),
                                  ("shoulder", "elbow"), ("elbow", "wrist")):
            i, j = [names.index(side + "_" + n) for n in (proximal, distal)]
            a = float(np.linalg.norm(source_data.xpos[source_ids[i]] - source_data.xpos[source_ids[j]]))
            b = float(np.linalg.norm(target_data.xpos[target_ids[i]] - target_data.xpos[target_ids[j]]))
            report["limb_lengths"][side + "_" + proximal + "_" + distal] = {
                "source_m": a, "q1_m": b, "ratio": b / a}
    for item in manifest["motions"]:
        trace_path = Path(item["kinematic_trace"]["path"])
        if identity(trace_path)["sha256"] != item["kinematic_trace"]["sha256"]:
            raise ValueError("Preserved IK trace changed")
        with np.load(trace_path, allow_pickle=False) as trace:
            trace_names = trace["source_body_names"].tolist()
            qpos = trace["qpos_before_ground_origin"]
            targets = trace["target_position_m_world"]
            errors = trace["position_error_m"]
            positions = trace["source_position_m_world"]
            root_index = trace_names.index("pelvis")
            ids = np.array([assets.model.body(mapping[n]).id for n in trace_names])
            state = mujoco.MjData(assets.model)
            feet, recomputed = [], []
            for q in qpos:
                state.qpos[:] = q
                mujoco.mj_forward(assets.model, state)
                feet.append(support_height(assets.model, state, assets.foot_geoms))
                recomputed.append(state.xpos[ids] - targets[len(feet) - 1])
            reference, _ = source_reference(source, Path(item["source"]["path"]), trace_names, 120.)
            expected_root = target_data.qpos[:3] + manifest["scale"] * (
                reference.positions[:, root_index] - reference.calibration_positions[root_index])
            entry = {"clip": item["clip"], "trace": identity(trace_path),
                     "source_root_z_m": summary(positions[:, root_index, 2]),
                     "source_root_offset_from_neutral_m": summary(positions[:, root_index, 2] - source_data.qpos[2]),
                     "target_root_z_before_floor_m": summary(qpos[:, 2]),
                     "foot_min_z_before_floor_m": summary(feet),
                     "root_formula_max_abs": float(np.abs(qpos[:, :3] - expected_root).max()),
                     "stored_residual_max_abs_difference": float(np.abs(np.asarray(recomputed) - errors).max()),
                     "per_body_error_m": {n: summary(np.linalg.norm(errors[:, i], axis=-1))
                                          for i, n in enumerate(trace_names)},
                     "initial_joint_position_rad": dict(zip(assets.names, qpos[0, 7:].tolist())),
                     "joint_velocity_peak_rad_s": dict(zip(assets.names, (np.abs(np.diff(qpos[:, 7:], axis=0)) / .02).max(0).tolist()))}
            report["clips"].append(entry)
            print(json.dumps(entry, allow_nan=False), flush=True)
    report["sources"] = [identity(Path(__file__)), identity(Path(__file__).parent / "prepare_motion.py")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": identity(args.output),
                      "source_neutral_root": report["source_neutral_root"],
                      "q1_neutral_root": report["q1_neutral_root"],
                      "source_foot_min_z": report.get("source_neutral_foot_min_z"),
                      "q1_foot_min_z": report["q1_neutral_foot_min_z"],
                      "limb_lengths": report["limb_lengths"]}), flush=True)


if __name__ == "__main__":
    main()
