"""Measure sole geometry in saved Q1 poses; no integration, policy or optimization."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from prepare_q1_gait_review import FIXED, identity, write_csv


def foot_geometry(model, side):
    import mujoco
    body = model.body(side + "_ankle_roll_link").id
    geometries = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind = int(model.geom_type[geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            geometries.append((geom, "mesh", model.mesh_vert[start:start+count].copy()))
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            geometries.append((geom, "sphere", float(model.geom_size[geom, 0])))
        else:
            raise ValueError("Unsupported active foot collision geometry: " + str(kind))
    if not geometries or not any(kind == "mesh" for _, kind, _ in geometries):
        raise ValueError("Expected actual Q1 collision mesh and corner spheres")
    return body, geometries


def measure(model, data, poses, geometry):
    import mujoco
    import numpy as np

    n = len(poses)
    lower = np.empty((n, 2))
    points = np.empty((n, 2, 3))
    body_p, body_r = np.empty((n, 2, 3)), np.empty((n, 2, 3, 3))
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for foot, (body, shapes) in enumerate(geometry):
            candidates = []
            for geom, kind, value in shapes:
                rotation = data.geom_xmat[geom].reshape(3, 3)
                if kind == "mesh":
                    candidates.append(value @ rotation.T + data.geom_xpos[geom])
                else:
                    bottom = data.geom_xpos[geom].copy()
                    bottom[2] -= value
                    candidates.append(bottom[None])
            candidates = np.concatenate(candidates)
            point = candidates[candidates[:, 2].argmin()]
            lower[frame, foot], points[frame, foot] = point[2], point
            body_p[frame, foot] = data.xpos[body]
            body_r[frame, foot] = data.xmat[body].reshape(3, 3)
    local_point = np.einsum("tfji,tfj->tfi", body_r, points - body_p)
    left = np.maximum(np.arange(n) - 1, 0)
    right = np.minimum(np.arange(n) + 1, n - 1)
    # Hold each selected material point fixed in the foot for its centered difference.
    before = body_p[left] + np.einsum("tfij,tfj->tfi", body_r[left], local_point)
    after = body_p[right] + np.einsum("tfij,tfj->tfi", body_r[right], local_point)
    velocity = (after - before) / ((right-left)[:, None, None] * .02)
    return lower, np.linalg.norm(velocity[..., :2], axis=-1)


def audit(args):
    import mujoco
    import numpy as np
    import torch

    runtime = args.scene.parents[1]
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat

    if args.output.exists():
        raise FileExistsError(args.output)
    trial = args.trace.parent
    if json.loads((trial / "status.json").read_text())["status"] != "COMPLETED":
        raise ValueError("Read completed saved evidence only")
    summary = json.loads(args.trace.with_suffix(".json").read_text())
    if identity(args.trace) != summary["trace"]:
        raise ValueError("Saved trace identity changed")
    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"]
    if len(rows) != 330:
        raise ValueError("Original full330 library required")
    for row in rows:
        if identity(Path(row["path"]))["sha256"] != row["sha256"]:
            raise ValueError("Original Q1 reference changed")
    with np.load(args.trace, allow_pickle=False) as data:
        reference_steps = data["reference_steps"].copy()
        states = {name: data["states"][:int(data["steps"][index]), index].copy()
                  for index, row in enumerate(rows) if (name := row["name"]) in FIXED}
    torch.set_num_threads(8)
    kin = Q1Kinematics(args.scene, "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    if not np.array_equal(reference_steps, (motion.clip_len-1).cpu().numpy()):
        raise ValueError("Reference frame alignment differs from the original saved evaluation")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    geometry = [foot_geometry(model, side) for side in ("left", "right")]
    plane = model.geom("floor").id
    if (int(model.geom_type[plane]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
            or not np.allclose(model.geom_pos[plane], 0.)
            or not np.allclose(model.geom_quat[plane], [1., 0., 0., 0.])):
        raise ValueError("This diagnostic assumes the actual horizontal floor at z0")
    results, curves = [], []
    for index, row in enumerate(rows):
        count = int(reference_steps[index])
        if count <= 1:
            raise ValueError("Unexpected reference length")
        # Reference arrays after an early physics failure may contain reset frames.
        # Reconstruct the original immutable clip instead of reading that suffix.
        first = int(motion.clip_start[index])
        ref = motion.ref_state(torch.arange(first+1, first+count+1))
        poses = torch.cat((ref[0], mat_to_quat(ref[1]), ref[4]), -1).cpu().numpy()
        cases = [("reference", poses)]
        if row["name"] in states:
            state = states[row["name"]]
            cases.append(("physical_baseline", np.concatenate((state[:, :7], state[:, 13:35]), -1)))
        for label, pose in cases:
            height, speed = measure(model, data, pose, geometry)
            for foot, side in enumerate(("left", "right")):
                near = height[:, foot] <= .005
                values = dict(clip=row["name"], split=row["split"], group=row["group"], case=label,
                    side=side, frames=len(pose), lowest_height_m=float(height[:, foot].min()),
                    sole_height_p05_m=float(np.quantile(height[:, foot], .05)),
                    sole_height_p95_m=float(np.quantile(height[:, foot], .95)),
                    below_minus3mm_fraction=float((height[:, foot] < -.003).mean()),
                    near_floor5mm_fraction=float(near.mean()),
                    near_floor_material_speed_mps=float(speed[near, foot].mean()) if near.any() else None,
                    near_floor_speed_above20cms_fraction=float((speed[near, foot] > .2).mean()) if near.any() else None)
                results.append(values)
                if row["name"] in FIXED:
                    curves.extend(dict(clip=row["name"], case=label, side=side, seconds=(frame+1)*.02,
                                       sole_height_m=float(h), lowest_material_speed_mps=float(v))
                                  for frame, (h, v) in enumerate(zip(height[:, foot], speed[:, foot])))
        if index % 50 == 0:
            print(json.dumps(dict(reference_clips_processed=index+1)), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "per_clip_sole.csv", results)
    write_csv(args.output / "fixed_sole_timeseries.csv", curves)
    value = dict(status="SAVED_POSE_SOLE_GEOMETRY_AUDITED", captured_utc=datetime.now(timezone.utc).isoformat(),
        sources=[identity(path) for path in (args.trace, args.library, args.scene, Path(__file__))],
        reference_clips=330, physical_fixed_clips=len(states), rows=len(results), fixed_rows=len(curves),
        reference_csv_sha_verified=330, reference_frames_from_original_library=True,
        geometry={side: [model.geom(geom).name for geom, _, _ in shapes]
                  for side, (_, shapes) in zip(("left", "right"), geometry)},
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        limits=["Exact lowest active mesh vertex or sphere surface vs actual floor; no contact forces or dynamics computed.",
                "Near-ground is a geometric proxy, not a contact label, support proof or data quality gate.",
                "Material-point velocity uses centered20ms saved pose differences; no point-switch displacement jumps.",
                "Only reference poses are measured across330; physical comparison is the fixed3 baseline clips.",
                "This does not isolate past learning degradation or authorize changing references or training."])
    with (args.output / "summary.json").open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("trace", "library", "scene", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    audit(parser.parse_args())
