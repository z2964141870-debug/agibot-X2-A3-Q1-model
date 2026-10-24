#!/usr/bin/env python3
"""P12C: velocity-qualified stance, native floor bounds and prioritized foot tasks."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

import probe_walking_contact as engine
import probe_walking_contact_velocity as velocity_adapter


def floor_limit(mink, model, dt):
    floor = model.geom("floor").id
    feet = [gid for gid in range(model.ngeom)
        if model.body(int(model.geom_bodyid[gid])).name in ("left_ankle_roll_link", "right_ankle_roll_link")
        and (model.geom_contype[gid] or model.geom_conaffinity[gid])]
    limit = mink.CollisionAvoidanceLimit(model, [(feet, [floor])], gain=.85 * dt,
        minimum_distance_from_collisions=0., collision_detection_distance=.05)
    if len(limit.geom_id_pairs) != len(feet) or not feet:
        raise ValueError("Every registered foot collision shape must have a native floor constraint")
    return limit


def solve_grounded(solver, roots, positions, rotations, start, bound, collision,
                   mj, mink, np, Rotation, floor):
    model, configuration = solver.sim.model, solver.configuration
    model.geom_pos[model.geom("floor").id, 2] = floor
    for index, task in solver.tasks:
        if solver.names[index] in ("left_ankle", "right_ankle"):
            task.set_position_cost(200.)
            task.set_orientation_cost(10.)
    tasks = [task for _, task in solver.tasks] + [solver.posture]
    extra = [floor_limit(mink, model, .02)]
    if collision:
        extra.append(engine.native_collision_limit(mink, model, .02))
    configuration.update(start.copy())

    def targets(frame):
        q = configuration.data.qpos.copy()
        q[:7] = roots[frame]
        configuration.update(q)
        solver.root_task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(Rotation.from_quat(q[3:7], scalar_first=True).as_matrix()), q[:3]))
        for index, task in solver.tasks:
            task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotations[frame, index]), positions[frame, index]))

    def update(dt, limits):
        for limit in extra:
            limit.gain = .85 * dt
        vector = mink.solve_ik(configuration, tasks, dt, solver.config["solver"],
            damping=solver.config["damping"], safety_break=True, limits=limits,
            constraints=[solver.root_task])
        configuration.integrate_inplace(vector, dt)
        return vector

    targets(0)
    for _ in range(32):
        if np.linalg.norm(update(.02, [solver.limits[0], *extra])) < 1e-5:
            break
    iterations, result = solver.config["iterations_per_frame"], []
    for frame in range(len(roots)):
        targets(frame)
        for _ in range(iterations):
            update(.02 / iterations, [*solver.limits, *extra])
        q = configuration.data.qpos.copy()
        joints = q[solver.sim.q_indices]
        if (not np.isfinite(q).all() or np.any(joints < solver.sim.lower - 1e-6)
                or np.any(joints > solver.sim.upper + 1e-6)):
            raise ValueError("Ground-constrained IK violated finite or position limits")
        result.append(q)
    result = np.asarray(result)
    if (np.abs(result[:, :7] - roots).max() > 1e-8
            or np.abs(np.diff(result[:, 7:], axis=0) / .02).max() > bound + .00002):
        raise ValueError("Ground-constrained IK changed root or violated the speed bound")
    return result, sum(len(limit.geom_id_pairs) for limit in extra)


def worker(args):
    fixed = [engine.identity(__file__), engine.identity(Path(__file__).with_name("test_walking_contact_floor.py"))]
    original_target, original_solve, original_save = engine.contact_targets, engine.solve, engine.save_json
    context = {}

    def capture_floor(positions, rotations, mask, feet, floor, level_rotations):
        context["floor"] = floor
        return original_target(positions, rotations, mask, feet, floor, level_rotations)

    def grounded(*arguments):
        return solve_grounded(*arguments, floor=context["floor"])

    def save_effective(path, value):
        if any(engine.identity(item["path"]) != item for item in fixed):
            raise ValueError("The P12C adapter changed during execution")
        value = dict(value)
        value["effective_floor_adapter"] = {"protocol": "P12C_NATIVE_FOOT_FLOOR_AND_PRIORITIZED_ANKLES",
            "floor_clearance_m": 0., "detection_distance_m": .05,
            "foot_position_cost": 200., "foot_orientation_cost": 10.,
            "floor_coordinate": "Same per-bound baseline constant origin as ground_origin_shift_m; no root projection",
            "stance_solver_includes_floor": True, "collision_solver_also_includes_original_wrist_limit": True,
            "source": fixed[0], "controls": fixed[1], "runtime_solve_hook_verified": engine.solve is grounded,
            "limits": "Native first-order IK distance bounds; final geometry and inverse costs must still pass independently."}
        if "frozen_inputs" in value:
            value["frozen_inputs"] = [*value["frozen_inputs"], *fixed]
        original_save(path, value)

    engine.contact_targets, engine.solve, engine.save_json = capture_floor, grounded, save_effective
    try:
        velocity_adapter.worker(args)
    finally:
        engine.contact_targets, engine.solve, engine.save_json = original_target, original_solve, original_save


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new P12C diagnostic directory")
    budget = engine.MemoryBoundBudget(engine.StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stop = False

    def requested(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
        "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    record = engine.guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.25, grace_s=5.,
        stop_requested=lambda: stop, progress_path=args.output / "probe", maximum_idle_s=900.)
    print(json.dumps(record), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
