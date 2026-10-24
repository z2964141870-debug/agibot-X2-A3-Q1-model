"""Train-only stance targets for an isolated Q1 IK diagnostic."""

import numpy as np


def runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("A one-dimensional contact mask is required")
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def stance_mask(sole_height, tolerance=.015, minimum_frames=4):
    height = np.asarray(sole_height, dtype=float)
    if (height.ndim != 2 or height.shape[1] != 2 or len(height) < 5
            or not np.isfinite(height).all() or tolerance <= 0 or minimum_frames < 1):
        raise ValueError("Finite bilateral foot heights and positive fixed thresholds required")
    mask = height - height.min(axis=1, keepdims=True) <= tolerance
    for side in range(2):
        values = mask[:, side]
        # Close isolated label dropouts, then discard short interior contact runs.
        values[1:-1] |= values[:-2] & values[2:]
        for start, end in runs(values):
            if end - start < minimum_frames and start > 0 and end < len(values):
                values[start:end] = False
    return mask


def contact_targets(positions, rotations, mask, feet, floor, level_rotations):
    from scipy.interpolate import BPoly
    from scipy.spatial.transform import Rotation, Slerp

    p, r = np.asarray(positions, float), np.asarray(rotations, float)
    contact = np.asarray(mask, bool)
    if (p.ndim != 3 or p.shape[1:] != (2, 3) or r.shape != (*p.shape[:2], 3, 3)
            or contact.shape != p.shape[:2] or not np.isfinite(p).all()
            or not np.isfinite(r).all() or not np.isfinite(floor)
            or len(feet) != 2 or len(level_rotations) != 2):
        raise ValueError("Contact target shapes, geometry and finite values must agree")
    from analyze_phuma_support import pose_min_z
    from q1_retarget import heading_rotation

    output, orientation, events = p.copy(), r.copy(), []
    velocity = np.gradient(p, .02, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, .02, axis=0, edge_order=2)
    for side in range(2):
        intervals = runs(contact[:, side])
        if not intervals:
            raise ValueError("Each foot needs at least one inferred stance interval")
        side_events = []
        for start, end in intervals:
            middle = (start + end - 1) // 2
            level = level_rotations[side]
            anchor_rotation = heading_rotation(r[middle, side]) @ heading_rotation(level).T @ level
            anchor = np.median(p[start:end, side], axis=0)
            anchor[2] = floor - pose_min_z(feet[side], np.zeros(3), anchor_rotation)
            output[start:end, side] = anchor
            orientation[start:end, side] = anchor_rotation
            side_events.append({"start_frame": start, "end_frame_exclusive": end,
                                "anchor_m": anchor.tolist(), "rotation": anchor_rotation.tolist()})
        first, last = intervals[0][0], intervals[-1][1] - 1
        output[:first, side] += output[first, side] - p[first, side]
        output[last + 1:, side] += output[last, side] - p[last, side]
        for (left_start, left_end), (right_start, right_end) in zip(intervals, intervals[1:]):
            left, right = left_end - 1, right_start
            boundary = [[output[index, side] - p[index, side], -velocity[index, side],
                         -acceleration[index, side]] for index in (left, right)]
            polynomial = BPoly.from_derivatives([left * .02, right * .02], boundary)
            gap = np.arange(left + 1, right)
            output[gap, side] = p[gap, side] + polynomial(gap * .02)
            offsets = [orientation[index, side] @ r[index, side].T for index in (left, right)]
            interpolation = Slerp([left, right], Rotation.from_matrix(offsets))
            orientation[gap, side] = interpolation(gap).as_matrix() @ r[gap, side]
        for indices, edge in ((np.arange(first), first), (np.arange(last + 1, len(p)), last)):
            orientation[indices, side] = (orientation[edge, side] @ r[edge, side].T) @ r[indices, side]
        events.append(side_events)
    if not np.isfinite(output).all() or not np.isfinite(orientation).all():
        raise ValueError("Nonfinite stance interpolation")
    return output, orientation, events


def native_collision_limit(mink, model, dt):
    wrists, obstacles = [], []
    obstacle_names = {"pelvis", "torso_link", "left_hip_pitch_link", "right_hip_pitch_link",
                      "left_hip_roll_link", "right_hip_roll_link", "left_hip_yaw_link",
                      "right_hip_yaw_link", "left_knee_link", "right_knee_link"}
    for gid in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[gid])).name
        if body in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            wrists.append(gid)
        if body in obstacle_names:
            obstacles.append(gid)
    if not wrists or not obstacles:
        raise ValueError("Expected registered wrist and trunk/leg geometry")
    # Mink 1.2.0 divides this displacement bound by dt; match the upstream correction.
    limit = mink.CollisionAvoidanceLimit(model, [(wrists, obstacles)], gain=.85 * dt,
        minimum_distance_from_collisions=.002, collision_detection_distance=.05)
    if not limit.geom_id_pairs:
        raise ValueError("Native collision filtering removed every diagnostic pair")
    return limit
