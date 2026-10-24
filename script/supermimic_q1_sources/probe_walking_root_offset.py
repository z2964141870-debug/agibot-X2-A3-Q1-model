#!/usr/bin/env python3
"""P14: constant root/upper-body translation with fixed P12C world foot targets."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget, guarded_run


HERE = Path(__file__).resolve().parent
MARGIN_M = .005
MAX_OFFSET_M = .12


def constant_offset(lower, upper, valid):
    lower, upper, valid = np.asarray(lower, float), np.asarray(upper, float), np.asarray(valid)
    if (lower.ndim != 1 or not len(lower) or upper.shape != lower.shape or valid.shape != lower.shape
            or valid.dtype != np.bool_ or not np.isfinite(lower).all() or not np.isfinite(upper).all()):
        raise ValueError("Finite matching interval arrays and boolean validity required")
    lo, hi = float(lower.max()), float(upper.min())
    delta = min(0., hi - MARGIN_M)
    reasons = []
    if not valid.all():
        reasons.append("A_FRAME_HAS_NO_NECESSARY_VERTICAL_INTERVAL")
    if delta < lo + MARGIN_M:
        reasons.append("NO_COMMON_INTERVAL_WITH_REGISTERED_MARGIN")
    if abs(delta) > MAX_OFFSET_M:
        reasons.append("CONSTANT_OFFSET_EXCEEDS_REGISTERED_120MM_CAP")
    return {"status": "REJECTED" if reasons else "ACCEPTED_NECESSARY_BOUND_ONLY",
            "common_lower_m": lo, "common_upper_m": hi, "root_delta_z_m": delta,
            "margin_m": MARGIN_M, "maximum_offset_m": MAX_OFFSET_M, "reasons": reasons}


def translate_targets(roots, positions, names, delta):
    roots, positions = np.asarray(roots, float), np.asarray(positions, float)
    if (roots.ndim != 2 or roots.shape[1] != 7 or positions.shape != (len(roots), len(names), 3)
            or len(set(names)) != len(names) or not np.isfinite(roots).all()
            or not np.isfinite(positions).all() or not np.isfinite(delta) or not -MAX_OFFSET_M <= delta <= 0):
        raise ValueError("Finite root/task arrays and the bounded nonpositive offset required")
    ankles = [names.index(side + "_ankle") for side in ("left", "right")]
    nonfeet = [i for i in range(len(names)) if i not in ankles]
    changed_roots, changed_positions = roots.copy(), positions.copy()
    changed_roots[:, 2] += delta
    changed_positions[:, nonfeet, 2] += delta
    return changed_roots, changed_positions


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def budget_for(args):
    return MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)


def worker(args):
    import mujoco as mj
    import mink
    from scipy.spatial.transform import Rotation

    from probe_walking_contact_floor import solve_grounded

    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    import retarget
    sys.path.insert(0, str(HERE.parent / "supermimic_q1_analysis"))
    from inspect_reference_dynamics import bottom_point, inspect_clip

    if importlib.metadata.version("mink") != "1.2.0":
        raise ValueError("Require the previously controlled Mink collision-time convention")
    base = args.root / "data/supermimic_for_q1"
    prior = base / "PHUMA_P12C_walking_floor_probe_20261025"
    reach = base / "PHUMA_P13_walking_reach_20261025"
    paths = [prior / "probe/receipt.json", prior / "independent_audit.json", prior / "guard.json",
             reach / "diagnosis/receipt.json", reach / "independent_audit.json", reach / "guard.json"]
    p12, a12, g12, p13, a13, g13 = [json.loads(path.read_text()) for path in paths]
    if (p12["status"] != "TRAIN_ONLY_WALKING_CONTACT_DIAGNOSTIC_COMPLETED"
            or a12["status"] != "INDEPENDENT_WALKING_CONTACT_AUDIT_COMPLETED"
            or a12["receipt"] != identity(paths[0])
            or p13["status"] != "TRAIN_ONLY_WALKING_REACH_BOUNDS_COMPLETED"
            or a13["status"] != "INDEPENDENT_WALKING_REACH_AUDIT_COMPLETED"
            or a13["receipt"] != identity(paths[3]) or p13["prior"] != identity(paths[0])
            or any(g["status"] != "WORKER_COMPLETED" or not g["cleanup"]["worker_reaped"] for g in (g12, g13))
            or len(p12["rows"]) != 4 or any(row["source_split"] != "train" for row in p12["rows"])):
        raise ValueError("Require the completed, independently checked train-only P12C and P13")
    selected = [row for row in p13["rows"] if row["variant"] == "stance_collision"]
    if (len(selected) != 8 or any(row["source_split"] != "train" for row in selected)
            or {(row["member"], row["bound_rad_s"]) for row in selected}
            != {(row["member"], bound) for row in p12["rows"] for bound in (2, 8)}):
        raise ValueError("Require exactly the four established train clips and both registered bounds")
    frozen = [*(identity(path) for path in paths), *p13["frozen_inputs"],
        identity(__file__), identity(HERE / "test_walking_root_offset.py"),
        identity(HERE / "audit_walking_root_offset.py"),
        identity(HERE / "guard_phuma_batch.py"), identity(HERE / "runtime_guard.py")]
    for row in selected:
        frozen.extend([row["input"], row["arrays"]])
    frozen = list({row["path"]: row for row in frozen}.values())
    if any(identity(row["path"]) != row for row in frozen):
        raise ValueError("A bound input changed before the constant-offset probe")
    budget = budget_for(args)
    budget.check()
    output = args.output / "probe"
    output.mkdir()
    (output / "dynamics").mkdir()
    assets = retarget.ContactCorrectedAssets(args.root)
    model, data = assets.model, mj.MjData(assets.model)
    names = list(retarget.BODY_MAP)
    feet = [np.array([gid for gid in assets.foot_geoms
        if model.geom_bodyid[gid] == model.body(side + "_ankle_roll_link").id]) for side in ("left", "right")]
    old_by_key = {(row["member"], result["bound_rad_s"]): result
        for row in p12["rows"] for result in row["variants"] if result["variant"] == "stance_collision"}
    rows, solved_frames = [], 0
    for source in selected:
        budget.check()
        with np.load(source["arrays"]["path"], allow_pickle=False) as saved:
            decision = constant_offset(saved["requested_root_delta_lower_m"],
                saved["requested_root_delta_upper_m"], saved["requested_vertical_interval_valid"])
        old = old_by_key[(source["member"], source["bound_rad_s"])]
        if source["input"] != old["arrays"] or old["status"] != "COMPLETED":
            raise ValueError("The P13 reach array must refer to the exact P12C collision variant")
        row = {key: source[key] for key in ("member", "original_motion_key", "source_split", "bound_rad_s")}
        row.update(variant="constant_root_z_stance_collision", decision=decision,
                   prior_arrays=source["input"], prior_reach_arrays=source["arrays"],
                   qualified_for_training=False, comparison={key: old[key] for key in
                   ("support_within_15mm_fraction", "support_gap_max_m", "foot_floor_min_m",
                    "self_penetration_max_m", "joint_speed_max_rad_s", "nominal_cartesian_residual_p95_m")})
        if decision["status"] == "REJECTED":
            row["status"] = "REJECTED_NECESSARY_BOUND"
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        try:
            with np.load(source["input"]["path"], allow_pickle=False) as saved:
                old_q, times, positions, rotations, mask = [saved[key].copy() for key in
                    ("qpos", "time_s", "task_position_m", "task_rotation_matrix", "stance_mask")]
                shift = float(saved["common_ground_origin_shift_m"])
            if not np.allclose(times, np.arange(len(times)) * .02, rtol=0, atol=1e-12):
                raise ValueError("The original native-time axis changed")
            roots = old_q[:, :7].copy()
            roots[:, 2] -= shift
            delta = decision["root_delta_z_m"]
            changed_roots, changed_positions = translate_targets(roots, positions, names, delta)
            solver, _ = retarget.grounded_solver(assets)
            bound = source["bound_rad_s"]
            solver.sim.config = dict(solver.sim.config, reference_velocity_limit_rad_s=bound)
            solver.limits[1] = mink.VelocityLimit(solver.sim.model,
                {name: bound * (1 - 1e-6) for name in solver.sim.names})
            start = old_q[0].copy()
            start[:7] = changed_roots[0]
            q, pairs = solve_grounded(solver, changed_roots, changed_positions, rotations,
                start, bound, True, mj, mink, np, Rotation, floor=-shift)
            q[:, 2] += shift
            velocity = np.empty((len(q), model.nv))
            for frame in range(len(q) - 1):
                mj.mj_differentiatePos(model, velocity[frame], .02, q[frame], q[frame + 1])
            velocity[-1] = velocity[-2]
            gap, penetration, residual, nominal = [], [], [], []
            for frame, state in enumerate(q):
                if frame % 100 == 0:
                    budget.memory()
                data.qpos[:], data.qvel[:] = state, velocity[frame]
                mj.mj_forward(model, data)
                expected, original = changed_positions[frame].copy(), positions[frame].copy()
                expected[:, 2] += shift
                original[:, 2] += shift
                residual.append(data.xpos[solver.bodies] - expected)
                nominal.append(data.xpos[solver.bodies] - original)
                gap.append([bottom_point(model, data, ids, mj)[0] for ids in feet])
                penetration.append(max([0.] + [-float(c.dist) for c in data.contact
                    if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
            gap, penetration = np.asarray(gap), np.asarray(penetration)
            path = output / Path(source["input"]["path"]).name
            with path.open("xb") as stream:
                np.savez_compressed(stream, time_s=times, qpos=q, qvel=velocity,
                    task_position_m=changed_positions, task_rotation_matrix=rotations,
                    original_task_position_m=positions, original_root_qpos=old_q[:, :7],
                    requested_root_qpos=changed_roots, root_delta_z_m=delta,
                    common_ground_origin_shift_m=shift, stance_mask=mask,
                    task_residual_m=np.asarray(residual), nominal_residual_m=np.asarray(nominal),
                    all_frame_foot_height_m=gap, self_penetration_m=penetration)
            dynamics = inspect_clip(model, feet, model.geom("floor").id, assets.effort,
                {"clip": path.stem, "output": identity(path)}, output / "dynamics", mj, args.expected_oom_kill)
            low = gap.min(axis=1)
            row.update(status="COMPLETED", arrays=identity(path), frames=len(q), dynamics=dynamics,
                collision_pair_count=pairs, joint_speed_max_rad_s=float(np.abs(velocity[:, 6:]).max()),
                nominal_cartesian_residual_p95_m=float(np.quantile(np.linalg.norm(nominal, axis=-1), .95)),
                task_cartesian_residual_p95_m=float(np.quantile(np.linalg.norm(residual, axis=-1), .95)),
                self_penetration_max_m=float(penetration.max()), support_within_15mm_fraction=float(np.mean(low <= .015)),
                support_gap_max_m=float(low.max()), foot_floor_min_m=float(low.min()),
                original_2rad_s_speed_gate=bool(np.abs(velocity[:, 6:]).max() <= 2.00002),
                support_and_collision_subset_pass=bool(low.min() >= -.001 and (low <= .015).mean() >= .9
                    and low.max() <= .03 and penetration.max() <= .003))
            solved_frames += len(q)
        except Exception as exc:
            row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    if any(identity(row["path"]) != row for row in frozen):
        raise ValueError("A bound input changed during the constant-offset probe")
    health = budget.check()
    save_json(output / "receipt.json", {"status": "TRAIN_ONLY_CONSTANT_ROOT_OFFSET_DIAGNOSTIC_COMPLETED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "frozen_inputs": frozen,
        "q1_assets": assets.manifest(), "prior_contact_receipt": identity(paths[0]),
        "prior_reach_receipt": identity(paths[3]), "rows": rows, "solved_frames": solved_frames,
        "new_physics_steps": 0, "new_optimizer_updates": 0, "new_training_references": 0,
        "qualified_for_training": False, "health_after": health, "storage_status": "LOCAL_ONLY",
        "protocol": {"margin_m": MARGIN_M, "maximum_offset_m": MAX_OFFSET_M,
            "root_xy_orientation_and_vertical_shape_preserved": True, "foot_world_targets_preserved": True,
            "nonfoot_tasks_translate_with_root": True, "original_time_preserved": True},
        "limits": "Necessary reach bound only. New IK references are diagnostic, not admitted. Four support/collision checks are only a subset of the original nineteen gates; 8rad/s has no original speed admission. Inverse residuals are not policy tracking or proof of dynamics feasibility."})
    print(json.dumps({"receipt": identity(output / "receipt.json")}), flush=True)


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new P14 diagnostic directory")
    budget = budget_for(args)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
        "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    record = guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.25, grace_s=5.,
        stop_requested=lambda: stopped, progress_path=args.output / "probe", maximum_idle_s=900.)
    print(json.dumps(record), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
