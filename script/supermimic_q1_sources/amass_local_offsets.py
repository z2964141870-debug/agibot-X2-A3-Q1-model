"""Body-local calibration offsets with native-clock Mink IK for Q1 diagnostics."""

import numpy as np


def local_offset_targets(body, robot_positions, robot_rotations, scale):
    from scipy.spatial.transform import Rotation
    from q1_retarget import heading_rotation

    root = body.names.index("pelvis")
    calibration = Rotation.from_quat(body.calibration_orientations, scalar_first=True).as_matrix()
    source_rotations = Rotation.from_quat(body.orientations.reshape(-1, 4), scalar_first=True).as_matrix()
    source_rotations = source_rotations.reshape(len(body.times), len(body.names), 3, 3)
    alignment = robot_rotations[root] @ calibration[root].T
    heading = heading_rotation(robot_rotations[root]) @ heading_rotation(calibration[root]).T
    rotations = alignment @ source_rotations @ calibration.transpose(0, 2, 1) @ alignment.T @ robot_rotations
    roots = robot_positions[root] + scale * (body.positions[:, root] - body.calibration_positions[root]) @ heading.T
    rest_difference = (robot_positions - robot_positions[root]
                       - scale * (body.calibration_positions - body.calibration_positions[root]) @ heading.T)
    local_offsets = np.einsum("bij,bj->bi", robot_rotations.transpose(0, 2, 1), rest_difference)
    relative = scale * (body.positions - body.positions[:, root:root + 1]) @ heading.T
    positions = roots[:, None] + relative + np.einsum("fbij,bj->fbi", rotations, local_offsets)
    return positions, rotations, roots


def independent_local_offsets(body, robot_positions, robot_rotations, scale):
    from scipy.spatial.transform import Rotation

    root = body.names.index("pelvis")
    calibration = Rotation.from_quat(body.calibration_orientations, scalar_first=True).as_matrix()
    source_rotations = Rotation.from_quat(body.orientations.reshape(-1, 4), scalar_first=True).as_matrix()
    source_rotations = source_rotations.reshape(len(body.times), len(body.names), 3, 3)
    align = robot_rotations[root] @ calibration[root].T
    yaw = lambda matrix: np.arctan2(matrix[1, 0], matrix[0, 0])
    heading = Rotation.from_euler("z", yaw(robot_rotations[root]) - yaw(calibration[root])).as_matrix()
    roots = robot_positions[root] + scale * (body.positions[:, root] - body.calibration_positions[root]) @ heading.T
    points, rotations = np.empty_like(body.positions), np.empty_like(source_rotations)
    for index in range(len(body.names)):
        scaled_rest = scale * heading @ (body.calibration_positions[index] - body.calibration_positions[root])
        offset = robot_rotations[index].T @ (robot_positions[index] - robot_positions[root] - scaled_rest)
        for frame in range(len(body.times)):
            rotation = align @ source_rotations[frame, index] @ calibration[index].T @ align.T @ robot_rotations[index]
            source_relative = scale * heading @ (body.positions[frame, index] - body.positions[frame, root])
            points[frame, index] = roots[frame] + source_relative + rotation @ offset
            rotations[frame, index] = rotation
    quaternions = Rotation.from_matrix(rotations[:, root]).as_quat(scalar_first=True)
    return points, rotations, roots, quaternions


def fit_and_solve(solver, source, targets, rotations, roots):
    import mink
    import mujoco
    from scipy.spatial.transform import Rotation

    iterations = solver.config["iterations_per_frame"]
    if iterations != 4 or not np.allclose(np.diff(source.times), .02, rtol=0, atol=1e-12):
        raise ValueError("Keep the established four IK iterations and real 50 Hz clock")
    initial_iterations = 0
    qposes, errors = [], []
    for frame in range(len(source.times)):
        q = solver.configuration.data.qpos.copy()
        q[:3] = roots[frame]
        q[3:7] = Rotation.from_matrix(rotations[frame, solver.root]).as_quat(scalar_first=True)
        solver.configuration.update(q)
        solver.root_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rotations[frame, solver.root]), roots[frame]))
        for index, task in solver.tasks:
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations[frame, index]), targets[frame, index]))
        if frame == 0:
            for initial_iterations in range(1, 33):
                velocity = mink.solve_ik(solver.configuration, [task for _, task in solver.tasks] + [solver.posture],
                                         .02, solver.config["solver"], damping=solver.config["damping"],
                                         safety_break=True, limits=[solver.limits[0]], constraints=[solver.root_task])
                solver.configuration.integrate_inplace(velocity, .02)
                if np.linalg.norm(velocity) < 1e-5:
                    break
        for _ in range(iterations):
            velocity = mink.solve_ik(solver.configuration, [task for _, task in solver.tasks] + [solver.posture],
                                     .02 / iterations, solver.config["solver"], damping=solver.config["damping"],
                                     safety_break=True, limits=solver.limits, constraints=[solver.root_task])
            solver.configuration.integrate_inplace(velocity, .02 / iterations)
        q = solver.configuration.data.qpos.copy()
        joints = q[solver.sim.q_indices]
        if (not np.isfinite(q).all() or np.any(joints < solver.sim.lower - 1e-6)
                or np.any(joints > solver.sim.upper + 1e-6)):
            raise ValueError("Local-offset IK violated the existing joint limit contract")
        q[solver.sim.q_indices] = np.clip(joints, solver.sim.lower, solver.sim.upper)
        solver.configuration.update(q)
        mujoco.mj_fwdPosition(solver.sim.model, solver.configuration.data)
        qposes.append(q)
        errors.append(solver.configuration.data.xpos[solver.bodies] - targets[frame])
    return np.asarray(qposes), np.asarray(errors), initial_iterations
