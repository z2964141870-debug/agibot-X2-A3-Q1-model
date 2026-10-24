#!/usr/bin/env python3
"""Read-only P13 leg-reach bounds for the completed train-only P12C arrays."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget, guarded_run


SIDES = ("left", "right")


def leg_chain(model, side):
    import mujoco as mj

    hip = model.body(side + "_hip_pitch_link").id
    ankle = model.body(side + "_ankle_roll_link").id
    if model.body_parentid[hip] != model.body("pelvis").id:
        raise ValueError("The hip anchor must be attached directly to the fixed pelvis")
    bodies, body = [], ankle
    while body != hip:
        if body == 0:
            raise ValueError("The ankle must descend from the named hip")
        bodies.append(body)
        body = int(model.body_parentid[body])
    for body in [hip, *bodies]:
        first, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
        for joint in range(first, first + count):
            if (model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE
                    or np.max(np.abs(model.jnt_pos[joint])) > 1e-12):
                raise ValueError("This conservative bound requires hinges at body origins")
    records = [{"body": model.body(body).name, "offset_m": model.body_pos[body].tolist(),
                "length_m": float(np.linalg.norm(model.body_pos[body]))} for body in bodies]
    return hip, ankle, sum(row["length_m"] for row in records), records


def reach_bounds(target, hip, radius):
    target, hip = np.asarray(target, float), np.asarray(hip, float)
    if (target.ndim != 2 or target.shape[1] != 3 or hip.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(hip).all()
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError("Finite matching point arrays and a positive reach radius required")
    difference = target - hip
    horizontal2 = np.sum(difference[:, :2] ** 2, axis=1)
    vertical = np.sqrt(np.maximum(0., radius ** 2 - horizontal2))
    return {"excess_m": np.linalg.norm(difference, axis=1) - radius,
            "horizontal_feasible": horizontal2 <= radius ** 2 + 1e-12,
            "root_delta_lower_m": difference[:, 2] - vertical,
            "root_delta_upper_m": difference[:, 2] + vertical}


def statistics(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {"count": int(values.size), "min": float(values.min()),
            "p50": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker(args):
    import mujoco as mj

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supermimic_q1_phuma"))
    from retarget import ContactCorrectedAssets

    prior = args.root / "data/supermimic_for_q1/PHUMA_P12C_walking_floor_probe_20261025"
    receipt_path, audit_path = prior / "probe/receipt.json", prior / "independent_audit.json"
    receipt, audit, guard = [json.loads(path.read_text()) for path in
                            (receipt_path, audit_path, prior / "guard.json")]
    if (receipt["status"] != "TRAIN_ONLY_WALKING_CONTACT_DIAGNOSTIC_COMPLETED"
            or audit["status"] != "INDEPENDENT_WALKING_CONTACT_AUDIT_COMPLETED"
            or audit["receipt"] != identity(receipt_path)
            or guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"]
            or len(receipt["rows"]) != 4 or any(row["source_split"] != "train" for row in receipt["rows"])):
        raise ValueError("Require the original completed, independently checked train-only P12C")
    frozen = [identity(receipt_path), identity(audit_path), identity(prior / "guard.json"),
              *receipt["frozen_inputs"], identity(__file__),
              identity(Path(__file__).with_name("test_diagnose_walking_reach.py"))]
    frozen = list({row["path"]: row for row in frozen}.values())
    if any(identity(row["path"]) != row for row in frozen):
        raise ValueError("A bound input changed before the reach diagnosis")
    budget = MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        32 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.check()
    output = args.output / "diagnosis"
    output.mkdir()
    assets = ContactCorrectedAssets(args.root)
    model, data = assets.model, mj.MjData(assets.model)
    chains = [leg_chain(model, side) for side in SIDES]
    mapping_path = args.root / "script/supermimic_q1_phuma/retarget_mapping.json"
    names = list(json.loads(mapping_path.read_text())["body_map"])
    inputs, results, evaluated, bound_checks = [], [], 0, 0
    for clip in receipt["rows"]:
        for result in clip["variants"]:
            if result["status"] != "COMPLETED":
                raise ValueError("All P12C variants must be complete")
            record = result["arrays"]
            if identity(record["path"]) != record:
                raise ValueError("Saved P12C array bytes changed")
            inputs.append(record)
            with np.load(record["path"], allow_pickle=False) as saved:
                q, mask = saved["qpos"].copy(), saved["stance_mask"].copy()
                shift = float(saved["common_ground_origin_shift_m"])
                targets = {label: saved[field].copy() for label, field in
                           (("original", "original_task_position_m"), ("requested", "task_position_m"))}
            if q.shape != (len(q), model.nq) or not all(np.isfinite(v).all() for v in [q, *targets.values()]):
                raise ValueError("Finite arrays with the registered Q1 dimensions required")
            centers, actual = [], []
            for frame, state in enumerate(q):
                if frame % 100 == 0:
                    budget.memory()
                data.qpos[:] = state
                mj.mj_kinematics(model, data)
                centers.append([data.xpos[hip].copy() for hip, _, _, _ in chains])
                actual.append([data.xpos[ankle].copy() for _, ankle, _, _ in chains])
            centers, actual = np.asarray(centers), np.asarray(actual)
            radii = np.array([row[2] for row in chains])
            if np.max(np.linalg.norm(actual - centers, axis=-1) - radii) > 1e-10:
                raise ValueError("The proposed chain bound excludes an actual Q1 FK pose")
            bound_checks += len(q) * 2
            arrays = {"hip_center_m": centers, "actual_ankle_m": actual, "stance_mask": mask}
            mode_rows = []
            for label, points in targets.items():
                points[:, :, 2] += shift
                ankle_targets = points[:, [names.index(side + "_ankle") for side in SIDES]]
                side_bounds = [reach_bounds(ankle_targets[:, side], centers[:, side], chains[side][2])
                               for side in range(2)]
                excess = np.stack([row["excess_m"] for row in side_bounds], axis=1)
                if np.any(np.linalg.norm(actual - ankle_targets, axis=-1) + 1e-10 < np.maximum(0., excess)):
                    raise ValueError("Observed ankle error is below the geometric lower bound")
                lower = np.stack([row["root_delta_lower_m"] for row in side_bounds], axis=1).max(axis=1)
                upper = np.stack([row["root_delta_upper_m"] for row in side_bounds], axis=1).min(axis=1)
                horizontal = np.stack([row["horizontal_feasible"] for row in side_bounds], axis=1).all(axis=1)
                valid = horizontal & (lower <= upper + 1e-12)
                arrays.update({label + "_excess_m": excess, label + "_root_delta_lower_m": lower,
                               label + "_root_delta_upper_m": upper, label + "_vertical_interval_valid": valid})
                mode_rows.append({"target": label,
                    "sides": [{"side": side, "outside_reach_fraction": float(np.mean(excess[:, i] > 1e-8)),
                        "excess_m": statistics(excess[:, i]),
                        "stance_frames": int(mask[:, i].sum()),
                        "stance_excess_m": statistics(excess[mask[:, i].astype(bool), i])} for i, side in enumerate(SIDES)],
                    "either_ankle_outside_fraction": float(np.mean((excess > 1e-8).any(axis=1))),
                    "horizontal_impossible_fraction": float(np.mean(~horizontal)),
                    "no_common_vertical_interval_fraction": float(np.mean(~valid)),
                    "necessary_lowering_m_on_valid_frames": statistics(np.maximum(0., -upper[valid]))})
            path = output / Path(record["path"]).name
            np.savez_compressed(path, **arrays)
            entry = {"member": clip["member"], "original_motion_key": clip["original_motion_key"],
                     "source_split": "train", "variant": result["variant"], "bound_rad_s": result["bound_rad_s"],
                     "frames": len(q), "input": record, "arrays": identity(path), "modes": mode_rows}
            results.append(entry)
            evaluated += len(q)
            print(json.dumps({key: entry[key] for key in ("original_motion_key", "variant", "bound_rad_s", "frames")}), flush=True)
    if any(identity(row["path"]) != row for row in [*frozen, *inputs]):
        raise ValueError("A bound input changed during the reach diagnosis")
    budget.check()
    report = {"status": "TRAIN_ONLY_WALKING_REACH_BOUNDS_COMPLETED", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "prior": frozen[0], "independent_prior_audit": frozen[1], "source": identity(__file__),
        "frozen_inputs": frozen, "q1_assets": assets.manifest(),
        "chains": [{"side": side, "maximum_radius_m": chain[2], "segments": chain[3]}
                   for side, chain in zip(SIDES, chains)],
        "rows": results, "evaluated_frames": evaluated, "actual_fk_chain_checks": bound_checks,
        "new_ik_solves": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
        "new_training_references": 0, "qualified_for_training": False, "storage_status": "LOCAL_ONLY",
        "limits": "Triangle-inequality outer bound, ignoring hinge ranges and foot orientation. An outside target is impossible at the fixed root; an inside target or nonempty root-Z interval does not prove feasibility, contact, dynamics, or policy effectiveness."}
    save_json(output / "receipt.json", report)
    print(json.dumps({"receipt": identity(output / "receipt.json")}), flush=True)


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new P13 diagnostic directory")
    budget = MemoryBoundBudget(StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        32 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.check()
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
        stop_requested=lambda: stopped, progress_path=args.output / "diagnosis", maximum_idle_s=900.)
    print(json.dumps(record), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
