"""Project existing SOMA poses onto the actual V3 collision constraints on CPU."""

import argparse
import json
import multiprocessing
from pathlib import Path
import sys
import time

from common import FIXED, identity, require, save
from prepare import read_yaml

_worker = None


def initialize(config):
    global _worker
    sys.path.insert(0, config["reference_ik_tools"])
    sys.path.insert(0, str(Path(config["adapted_source"]) / "src"))
    import mink
    import mujoco
    import numpy as np
    from supermimic.filter_motions import CollisionGeometry
    geometry = CollisionGeometry(Path(config["xml"]))
    model = geometry.model
    configuration = mink.Configuration(model)
    pairs = [([int(a)], [int(b)]) for a, b in geometry.pairs]
    plane = [i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
    require(len(plane) == 1, "Expected the frozen horizontal floor")
    pairs.append((geometry.geoms.tolist(), plane))
    collision = mink.CollisionAvoidanceLimit(model, pairs, gain=.85,
        minimum_distance_from_collisions=.002, collision_detection_distance=.05)
    posture = mink.PostureTask(model, cost=1.)
    smooth = mink.PostureTask(model, cost=.02)
    pelvis = mink.FrameTask("pelvis", "body", position_cost=[2., 2., .1], orientation_cost=3.)
    feet = [mink.FrameTask(s + "_ankle_roll_link", "body", position_cost=[.5, .5, 2.],
                           orientation_cost=.2) for s in ("left", "right")]
    limits = [mink.ConfigurationLimit(model), collision]
    _worker = dict(mink=mink, mujoco=mujoco, np=np, geometry=geometry, model=model,
        configuration=configuration, posture=posture, smooth=smooth, pelvis=pelvis,
        feet=feet, limits=limits, config=config)


def job(item):
    relative, source, destination = item
    w = _worker
    np, mink, mj = w["np"], w["mink"], w["mujoco"]
    model, geom, cfg = w["model"], w["geometry"], w["configuration"]
    before = identity(source)
    started = time.monotonic()
    with np.load(source, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    names = original["joint_names"].tolist()
    bodies = original["body_names"].tolist()
    root = bodies.index("pelvis")
    native = [model.joint(i).name for i in range(1, 23)]
    order = [names.index(n) for n in native]
    reference = np.concatenate((original["body_pos_w"][:, root],
        original["body_quat_w"][:, root], original["joint_pos"][:, order]), axis=1).astype(float)
    require(float(original["fps"]) == 50 and np.isfinite(reference).all(), "Bad source reference")
    result, iterations, failures = [], [], []
    worst_ground, worst_self = 0., 0.
    previous = reference[0].copy()
    for frame, pose in enumerate(reference):
        target = pose.copy()
        target[3:7] /= np.linalg.norm(target[3:7])
        target[7:] = np.clip(target[7:], model.jnt_range[1:, 0] + 1e-7,
                             model.jnt_range[1:, 1] - 1e-7)
        cfg.update(target)
        w["posture"].set_target(target)
        w["smooth"].set_target(previous)
        w["pelvis"].set_target(cfg.get_transform_frame_to_world("pelvis", "body"))
        geom.pose(target)
        lower = geom.lower_bounds()
        for foot, side in zip(w["feet"], ("left", "right")):
            body = model.body(side + "_ankle_roll_link").id
            mask = model.geom_bodyid[geom.geoms] == body
            lift = max(0., .002 - float(lower[mask].min()))
            transform = cfg.get_transform_frame_to_world(side + "_ankle_roll_link", "body")
            foot.set_target(mink.SE3.from_rotation_and_translation(
                transform.rotation(), transform.translation() + np.array([0., 0., lift])))
        succeeded = False
        ground = max(0., -float(geom.lower_bounds().min()))
        penetration = geom.penetration().maximum_excess
        iteration = -1
        try:
            for iteration in range(16):
                velocity = mink.solve_ik(cfg, [w["posture"], w["smooth"], w["pelvis"], *w["feet"]],
                    .02, solver="daqp", damping=1e-5, limits=w["limits"], safety_break=False)
                cfg.integrate_inplace(velocity, .02)
                geom.pose(cfg.q)
                ground = max(0., -float(geom.lower_bounds().min()))
                penetration = geom.penetration().maximum_excess
                if ground <= .001 and penetration <= .003:
                    succeeded = True
                    break
            require(np.isfinite(cfg.q).all(), "Nonfinite IK result")
        except Exception as error:
            failures.append(dict(frame=frame, error=f"{type(error).__name__}: {error}"))
        if not succeeded:
            failures.append(dict(frame=frame, reason="collision_projection_did_not_converge"))
        corrected = cfg.q.copy()
        # Large deviations invalidate the candidate rather than manufacturing a new motion.
        if (np.max(np.abs(corrected[:2] - pose[:2])) > .05 or
                np.max(np.abs(corrected[7:] - pose[7:])) > .5 or
                abs(corrected[2] - pose[2]) > .15):
            failures.append(dict(frame=frame, reason="source_deviation_exceeds_frozen_bound"))
        result.append(corrected)
        previous = corrected
        iterations.append(iteration + 1)
        worst_ground, worst_self = max(worst_ground, ground), max(worst_self, penetration)
    repaired = np.asarray(result)
    positions, rotations = [], []
    for pose in repaired:
        geom.pose(pose)
        positions.append(np.stack([geom.data.xpos[model.body(n).id] for n in bodies]))
        rotations.append(np.stack([geom.data.xquat[model.body(n).id] for n in bodies]))
    positions, rotations = np.asarray(positions), np.asarray(rotations)
    joints = repaired[:, 7:][:, [native.index(n) for n in names]]
    left, right = np.maximum(np.arange(len(repaired)) - 1, 0), np.minimum(np.arange(len(repaired)) + 1, len(repaired) - 1)
    dt = (right - left) * .02
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_quat(rotations[..., [1, 2, 3, 0]].reshape(-1, 4))
    rv = rot.as_quat().reshape(rotations.shape)
    angular = (Rotation.from_quat(rv[right].reshape(-1, 4)) *
               Rotation.from_quat(rv[left].reshape(-1, 4)).inv()).as_rotvec().reshape(*positions.shape)
    original.update(joint_pos=joints.astype(np.float32),
        joint_vel=((joints[right] - joints[left]) / dt[:, None]).astype(np.float32),
        body_pos_w=positions.astype(np.float32), body_quat_w=rotations.astype(np.float32),
        body_lin_vel_w=((positions[right] - positions[left]) / dt[:, None, None]).astype(np.float32),
        body_ang_vel_w=(angular / dt[:, None, None]).astype(np.float32),
        v3_collision_repair="Mink1.4.0_actual_V3_geometry", repair_source_sha256=before["sha256"])
    original["reference_repair_status"] = "projected" if not failures else "review"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **original)
    require(identity(source) == before, "Original motion changed during repair")
    row = dict(path=relative, status="PROJECTED" if not failures else "REPAIR_REVIEW_REQUIRED",
        source=before, output=identity(destination), frames=len(reference), failures=failures,
        joint_change_rmse_rad=float(np.sqrt(np.mean((repaired[:, 7:] - reference[:, 7:])**2))),
        root_z_change_min_m=float((repaired[:, 2] - reference[:, 2]).min()),
        root_z_change_max_m=float((repaired[:, 2] - reference[:, 2]).max()),
        root_xy_max_change_m=float(np.max(np.abs(repaired[:, :2] - reference[:, :2]))),
        max_remaining_ground_penetration_m=worst_ground, max_remaining_self_excess_m=worst_self,
        qp_iterations=sum(iterations), seconds=time.monotonic() - started,
        new_dynamics_steps=0, new_policy_optimizer_updates=0)
    save(destination.with_suffix(".repair.json"), row)
    return row


def run(config, output, scope, workers):
    output.mkdir(parents=True, exist_ok=False)
    subset = read_yaml(config["q1_subset"], config["morph_python"])
    clips = [r for r in subset["clips"] if scope == "all" or Path(r).stem in FIXED]
    require(len(clips) == (559 if scope == "all" else 3), "Frozen reference selection changed")
    jobs = [(r, str(Path(subset["root"]) / r), str(output / "references" / r)) for r in clips]
    rows = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=initialize, initargs=(config,)) as pool:
        for row in pool.imap_unordered(job, jobs):
            rows.append(row)
            print(json.dumps(dict(completed=len(rows), total=len(clips), path=row["path"],
                status=row["status"], seconds=row["seconds"])), flush=True)
            save(output / "progress.json", dict(completed=len(rows), total=len(clips)))
    subset.update(root=str(output / "references"), clips=clips)
    save(output / "q1_paired.yaml", subset)
    save(output / "result.json", dict(status="COMPLETED", scope=scope, rows=rows,
        projected=sum(r["status"] == "PROJECTED" for r in rows),
        repair_is_not_screening_acceptance=True, source_subset=identity(config["q1_subset"]),
        original_screening_thresholds_unchanged=True, new_dynamics_steps=0, new_policy_optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("fixed", "all"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output.resolve(), args.scope, args.workers)
