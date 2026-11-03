"""Tracking and actual collision-surface diagnostics for native saved states."""

import csv
from pathlib import Path
import re

from contracts import require


def tracking(actual, reference, present, steps):
    import numpy as np
    actual, reference, present = map(np.asarray, (actual, reference, present))
    require(actual.ndim == reference.ndim == 2 and actual.shape[1] == reference.shape[1] == 269
            and present.shape == (29,) and present.dtype == np.bool_ and present.any()
            and type(steps) is int and 0 < steps < min(len(actual), len(reference)),
            "Invalid native tracking prefix or joint mask")
    a, b = actual[1:steps + 1], reference[1:steps + 1]
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite native trace")
    pa, pb = (x[:, 58:100].reshape(-1, 14, 3) for x in (a, b))
    return dict(joint_rmse_rad=float(np.sqrt(np.mean((a[:, :29][:, present] - b[:, :29][:, present]) ** 2))),
        body_relative_rmse_m=float(np.sqrt(np.mean(
            ((pa[:, 1:] - pa[:, :1]) - (pb[:, 1:] - pb[:, :1])) ** 2))))


def support_point(kind, size, direction, vertices=None):
    """Support of the actual MuJoCo collision shape in a local unit direction."""
    import numpy as np
    size, direction = np.asarray(size), np.asarray(direction)
    require(size.shape == (3,) and np.isfinite(size).all() and (size >= 0).all()
            and direction.shape == (3,) and np.isfinite(direction).all()
            and np.isclose(np.linalg.norm(direction), 1.), "Expected a finite unit support direction")
    if kind == "mesh":
        vertices = np.asarray(vertices)
        require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices)
                and np.isfinite(vertices).all(), "Missing or nonfinite mesh vertices")
        return vertices[(vertices @ direction).argmax()].copy()
    if kind == "sphere":
        return size[0] * direction
    if kind == "box":
        return size * np.sign(direction)
    if kind == "ellipsoid":
        require((size > 0).all(), "Degenerate ellipsoid")
        return size ** 2 * direction / np.linalg.norm(size * direction)
    if kind == "capsule":
        result = size[0] * direction
        result[2] += size[1] * np.sign(direction[2])
        return result
    if kind == "cylinder":
        result = np.zeros(3)
        norm = np.linalg.norm(direction[:2])
        if norm:
            result[:2] = size[0] * direction[:2] / norm
        result[2] = size[1] * np.sign(direction[2])
        return result
    raise RuntimeError("Unsupported active foot collision shape: " + str(kind))


def foot_shapes(model, spec):
    import mujoco
    kinds = {int(getattr(mujoco.mjtGeom, "mjGEOM_" + name.upper())): name
             for name in ("sphere", "box", "ellipsoid", "capsule", "cylinder", "mesh")}
    groups = [[], []]
    for geom in range(model.ngeom):
        name = (model.geom(geom).name or "").rsplit("/", 1)[-1]
        if not re.fullmatch(spec.foot_geom_regex, name):
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        require(name.startswith(("left_", "right_")), "Foot side is not explicit")
        kind = kinds.get(int(model.geom_type[geom]))
        require(kind is not None, "Unsupported active foot geometry")
        vertices = None
        if kind == "mesh":
            mesh = int(model.geom_dataid[geom])
            first, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[first:first + count].copy()
        groups[0 if name.startswith("left_") else 1].append(
            (geom, kind, model.geom_size[geom].copy(), vertices))
    require(all(groups), "Both real collision feet are required")
    return groups


def reference_poses(model, spec, states):
    import mujoco
    import numpy as np
    import torch
    require(states.ndim == 2 and states.shape[1] == 269, "Expected canonical269 reference")
    require(np.isfinite(states).all(), "Nonfinite canonical reference")
    joints = {}
    for joint in range(model.njnt):
        name = (model.joint(joint).name or "").rsplit("/", 1)[-1]
        require(name not in joints, "Ambiguous compiled joint name")
        joints[name] = joint
    free = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
    require(len(free) == 1, "Expected one free robot and a static floor")
    address = int(model.jnt_qposadr[free[0]])
    root_name = (model.body(int(model.jnt_bodyid[free[0]])).name or "").rsplit("/", 1)[-1]
    require(root_name == spec.root_body, "The free joint is not the declared robot root")
    addresses = np.asarray([model.jnt_qposadr[joints[name]] for name in spec.joint_names])
    require(all(int(model.jnt_type[joints[name]]) == int(mujoco.mjtJoint.mjJNT_HINGE)
                for name in spec.joint_names), "Expected scalar native robot hinges")
    q = spec.translator.to_native(torch.as_tensor(states[:, :29].copy()),
                                 torch.as_tensor(model.qpos0[addresses].copy())).numpy()
    poses = np.repeat(model.qpos0[None], len(states), axis=0)
    poses[:, address:address + 3] = states[:, 58:61]
    poses[:, address + 3:address + 7] = states[:, 100:104]
    poses[:, addresses] = q
    require(np.isfinite(poses).all() and np.allclose(
        np.linalg.norm(poses[:, address + 3:address + 7], axis=1), 1., atol=1e-5),
        "Nonfinite native reference qpos or invalid WXYZ orientation")
    return poses, address


def sole_series(model, spec, poses, states):
    import mujoco
    import numpy as np
    require(len(poses) == len(states) and poses.shape[1] == model.nq, "Sole trace dimensions differ")
    groups, data = foot_shapes(model, spec), mujoco.MjData(model)
    height, speed = np.empty((len(poses), 2)), np.empty((len(poses), 2))
    slots = [spec.tracked_bodies.index(side + "_ankle_roll_link") for side in ("left", "right")]
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for foot, shapes in enumerate(groups):
            candidates = []
            for geom, kind, size, vertices in shapes:
                rotation = data.geom_xmat[geom].reshape(3, 3)
                point = support_point(kind, size, -rotation[2], vertices)
                candidates.append((data.geom_xpos[geom] + rotation @ point, int(model.geom_bodyid[geom])))
            point, body = min(candidates, key=lambda item: item[0][2])
            name = (model.body(body).name or "").rsplit("/", 1)[-1]
            require(name == spec.tracked_bodies[slots[foot]], "Foot collision body differs from velocity slot")
            linear = states[frame, 156:198].reshape(14, 3)[slots[foot]]
            angular = states[frame, 198:240].reshape(14, 3)[slots[foot]]
            velocity = linear + np.cross(angular, point - data.xpos[body])
            height[frame, foot], speed[frame, foot] = point[2], np.linalg.norm(velocity[:2])
    return height, speed


def summarize(model, spec, actual, reference, poses, target_poses, force):
    import numpy as np
    height, speed = sole_series(model, spec, poses, actual)
    target_height, _ = sole_series(model, spec, target_poses[:len(actual)], reference[:len(actual)])
    require(len(actual) >= 2 and force.shape == height.shape and np.isfinite(force).all()
            and (force >= 0).all(), "Missing physical contact-force evidence")
    quat = actual[:, 100:156].reshape(-1, 14, 4)[:, 7]
    target_quat = reference[:len(actual), 100:156].reshape(-1, 14, 4)[:, 7]
    def tilt(q):
        return np.arccos(np.clip(1. - 2. * (q[:, 1] ** 2 + q[:, 2] ** 2), -1., 1.))
    angles, target_angles = tilt(quat), tilt(target_quat)
    upper = [i for i, name in enumerate(spec.joint_names) if "shoulder" in name or "elbow" in name]
    slots = [i for i, native in enumerate(spec.translator.perm) if native in upper]
    events = sorted((int(i), foot) for foot in (0, 1)
        for i in np.flatnonzero((height[1:, foot] > .01) & (height[:-1, foot] <= .01)) + 1)
    loaded = force[1:] >= 5.
    result = dict(sole_p95_m=np.quantile(height[1:], .95, axis=0).tolist(),
        reference_sole_p95_m=np.quantile(target_height[1:], .95, axis=0).tolist(),
        lift_events_above_10mm=len(events),
        alternating_lift_transitions=sum(a[1] != b[1] for a, b in zip(events, events[1:])),
        loaded_samples=loaded.sum(axis=0).tolist(),
        loaded_slip_mean_m_s=[float(speed[1:, side][loaded[:, side]].mean())
                             if loaded[:, side].any() else None for side in (0, 1)],
        torso_tilt_rms_rad=float(np.sqrt(np.mean(angles[1:] ** 2))),
        tilt_error_rms_rad=float(np.sqrt(np.mean((angles[1:] - target_angles[1:]) ** 2))),
        upper_joint_rmse_rad=float(np.sqrt(np.mean(
            (actual[1:, :29][:, slots] - reference[1:len(actual), :29][:, slots]) ** 2))),
        upper_joint_peak_to_peak_rad=np.ptp(actual[:, :29][:, slots], axis=0).tolist(),
        upper_reference_peak_to_peak_rad=np.ptp(reference[:len(actual), :29][:, slots], axis=0).tolist(),
        root_xy_final_error_m=float(np.linalg.norm(actual[-1, 58:60] - reference[len(actual)-1, 58:60])),
        load_time_note="native post-control forward solver normal forces; endpoint samples, not continuous support",
        slip_note="instantaneous velocity of the lowest actual collision-surface material point")
    return result, dict(sole_height=height, reference_sole_height=target_height,
                       material_point_slip=speed, load=force, tilt=angles), slots


def write_series(output, actual, reference, series, slots, names):
    with (Path(output) / "foot_posture.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time_s", "left_sole_m", "right_sole_m", "left_reference_sole_m",
            "right_reference_sole_m", "left_load_n", "right_load_n", "left_slip_m_s", "right_slip_m_s", "torso_tilt_rad"))
        writer.writerows((i, .02*i, *series["sole_height"][i], *series["reference_sole_height"][i],
            *series["load"][i], *series["material_point_slip"][i], series["tilt"][i]) for i in range(len(actual)))
    with (Path(output) / "upper_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s"] + [stage+"_"+names[i] for stage in ("actual", "reference") for i in slots])
        writer.writerows([i, .02*i] + actual[i, slots].tolist() + reference[i, slots].tolist()
                         for i in range(len(actual)))
