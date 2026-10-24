#!/usr/bin/env python3
"""Compare a local G1 model with a pinned official XML for source FK readiness."""

import argparse
import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from inventory_amass import identity


def compare(candidate, official, mesh_dir, seed=20261024, frames=64):
    reference_xml = ET.parse(official).getroot()
    reference_xml.find("compiler").set("meshdir", str(mesh_dir.resolve(strict=True)))
    models = [mujoco.MjModel.from_xml_path(str(candidate)),
              mujoco.MjModel.from_xml_string(ET.tostring(reference_xml, encoding="unicode"))]
    names = lambda model, kind, count: [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    body_names = [names(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody) for m in models]
    joint_names = [names(m, mujoco.mjtObj.mjOBJ_JOINT, m.njnt) for m in models]
    dimensions = [{key: int(getattr(m, key)) for key in ("nq", "nv", "nu", "nbody", "njnt")}
                  for m in models]
    layout = {"dimensions": dimensions[0] == dimensions[1],
              "body_names": body_names[0] == body_names[1],
              "joint_names": joint_names[0] == joint_names[1]}
    differences = {}
    for key in ("body_parentid", "body_jntnum", "jnt_bodyid", "jnt_type", "jnt_qposadr",
                "jnt_dofadr", "jnt_limited", "body_pos", "body_quat", "jnt_pos", "jnt_axis",
                "jnt_range", "qpos0"):
        a, b = (np.asarray(getattr(m, key)) for m in models)
        differences[key] = (float(np.max(np.abs(a.astype(float) - b.astype(float))))
                            if a.shape == b.shape else None)

    def feet(model):
        ankles = {model.body(side + "_ankle_roll_link").id for side in ("left", "right")}
        records = []
        for gid in range(model.ngeom):
            body = int(model.geom_bodyid[gid])
            while body and body not in ankles:
                body = int(model.body_parentid[body])
            if body in ankles and (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                records.append({"body": model.body(int(model.geom_bodyid[gid])).name,
                                "type": int(model.geom_type[gid]),
                                "position": model.geom_pos[gid].tolist(),
                                "quaternion": model.geom_quat[gid].tolist(),
                                "size": model.geom_size[gid].tolist(),
                                "contype": int(model.geom_contype[gid]),
                                "conaffinity": int(model.geom_conaffinity[gid])})
        return records

    foot_records = [feet(m) for m in models]
    foot_equal = bool(foot_records[0]) and foot_records[0] == foot_records[1]
    fk = {"frames": 0, "position_max_abs_m": None, "quaternion_sign_invariant_max_abs": None}
    if all(layout.values()):
        if (models[0].nq != 36 or models[0].nv != 35
                or models[0].jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
                or any(models[0].jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE)):
            raise ValueError("Expected a free base followed by 29 hinges")
        rng = np.random.default_rng(seed)
        states = np.tile(models[0].qpos0, (frames, 1))
        states[:, :3] += rng.normal(scale=.15, size=(frames, 3))
        states[:, 3:7] = Rotation.random(frames, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
        lo = np.maximum(models[0].jnt_range[1:, 0], models[1].jnt_range[1:, 0])
        hi = np.minimum(models[0].jnt_range[1:, 1], models[1].jnt_range[1:, 1])
        if np.any(lo >= hi):
            raise ValueError("Official and candidate joint limits have no common interval")
        states[:, 7:] = rng.uniform(lo, hi, size=(frames, 29))
        data = [mujoco.MjData(m) for m in models]
        position_error, quaternion_error = 0., 0.
        for state in states:
            for model, current in zip(models, data):
                current.qpos[:] = state
                mujoco.mj_kinematics(model, current)
            position_error = max(position_error, float(np.abs(data[0].xpos - data[1].xpos).max()))
            quaternion_error = max(quaternion_error, float(np.minimum(
                np.abs(data[0].xquat - data[1].xquat).max(-1),
                np.abs(data[0].xquat + data[1].xquat).max(-1)).max()))
        fk = {"frames": frames, "position_max_abs_m": position_error,
              "quaternion_sign_invariant_max_abs": quaternion_error}
    passed = (all(layout.values()) and all(v is not None and v <= 1e-12 for v in differences.values())
              and foot_equal and fk["frames"] == frames and fk["position_max_abs_m"] <= 1e-12
              and fk["quaternion_sign_invariant_max_abs"] <= 1e-12)
    return {"status": "OFFICIAL_G1_KINEMATICS_EQUIVALENT" if passed else "OFFICIAL_G1_KINEMATICS_DIFFER",
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "candidate": identity(candidate), "official_xml": identity(official),
            "official_compilation_mesh_directory": str(mesh_dir.resolve()),
            "dimensions": dimensions, "layout_equal": layout,
            "compiled_array_max_abs_differences": differences,
            "foot_collision_geometry_equal": foot_equal, "foot_collision_geometry": foot_records,
            "seed": seed, "random_pose_fk": fk, "code": identity(Path(__file__)),
            "physics_steps": 0, "optimizer_updates": 0, "bones_generation_model_identity_proven": False,
            "limits": "Official Unitree G1 equivalence only. Existing local meshes were reused to compile the official XML, not independently validated as official mesh bytes. This does not establish the BONES generation model, original FBX hierarchy or retarget quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.candidate.resolve(), args.official.resolve(), args.mesh_dir.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
