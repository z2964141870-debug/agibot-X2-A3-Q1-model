#!/usr/bin/env python3
"""Recompute P13 bounds from root transforms and saved P12C targets, independently."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from guard_phuma_batch import MemoryBoundBudget
from prepare_amass_body import identity
from runtime_guard import StorageBudget


def audit(args):
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    root = args.root.resolve(strict=True)
    experiment = root / "data/supermimic_for_q1/PHUMA_P13_walking_reach_20261025"
    receipt_path = experiment / "diagnosis/receipt.json"
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(experiment):
        raise ValueError("Use a new audit file inside the completed P13 experiment")
    receipt_id = identity(receipt_path)
    report = json.loads(receipt_path.read_text())
    guard = json.loads((experiment / "guard.json").read_text())
    if (report["status"] != "TRAIN_ONLY_WALKING_REACH_BOUNDS_COMPLETED"
            or guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"]):
        raise ValueError("Require a completed and reaped P13")
    budget = MemoryBoundBudget(StorageBudget(experiment,
        root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        32 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.check()
    inputs = [receipt_id, *report["frozen_inputs"], report["q1_assets"]["mjcf"]]
    if any(identity(row["path"]) != row for row in inputs):
        raise ValueError("A fixed input changed")
    model = mj.MjModel.from_xml_path(report["q1_assets"]["mjcf"]["path"])
    data = mj.MjData(model)
    names = list(json.loads((root / "script/supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"])
    sides, radii, checks, frames = ("left", "right"), [], 0, 0
    for side in sides:
        labels = [side + suffix for suffix in ("_hip_roll_link", "_hip_yaw_link", "_knee_link",
                                               "_ankle_pitch_link", "_ankle_roll_link")]
        radius = sum(float(np.sqrt(np.dot(model.body(label).pos, model.body(label).pos))) for label in labels)
        radii.append(radius)
        if abs(radius - report["chains"][sides.index(side)]["maximum_radius_m"]) > 1e-12:
            raise ValueError("Independent segment-radius mismatch")
        checks += 1

    def equal(actual, expected, tolerance=1e-10):
        nonlocal checks
        if not np.allclose(actual, expected, rtol=0, atol=tolerance):
            raise ValueError("Independent reach array/statistic mismatch")
        checks += 1

    def stats(actual, expected):
        equal(actual.size, expected["count"], 0)
        if actual.size:
            for value, key in zip((actual.min(), np.median(actual), np.quantile(actual, .95), actual.max()),
                                  ("min", "p50", "p95", "max")):
                equal(value, expected[key])
        elif any(expected[key] is not None for key in ("min", "p50", "p95", "max")):
            raise ValueError("Empty statistics must remain null")

    for row in report["rows"]:
        if row["source_split"] != "train":
            raise ValueError("Only the fixed train diagnosis is allowed")
        for record in (row["input"], row["arrays"]):
            if identity(record["path"]) != record:
                raise ValueError("A saved array changed")
            inputs.append(record)
        with np.load(row["input"]["path"], allow_pickle=False) as saved:
            q = saved["qpos"].copy()
            targets = [saved[key].copy() for key in ("original_task_position_m", "task_position_m")]
            shift, mask = float(saved["common_ground_origin_shift_m"]), saved["stance_mask"].astype(bool)
        with np.load(row["arrays"]["path"], allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        rotations = Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
        hips = np.stack([q[:, :3] + rotations @ model.body(side + "_hip_pitch_link").pos for side in sides], axis=1)
        equal(hips, arrays["hip_center_m"])
        for frame, state in enumerate(q):
            if frame % 100 == 0:
                budget.memory()
            data.qpos[:] = state
            mj.mj_kinematics(model, data)
            equal(data.xpos[[model.body(side + "_ankle_roll_link").id for side in sides]], arrays["actual_ankle_m"][frame])
        for points, mode in zip(targets, row["modes"]):
            label = mode["target"]
            ankle = points[:, [names.index(side + "_ankle") for side in sides]].copy()
            ankle[:, :, 2] += shift
            delta = ankle - hips
            distance = np.sqrt(np.einsum("fsc,fsc->fs", delta, delta))
            excess = distance - radii
            equal(excess, arrays[label + "_excess_m"])
            horizontal2 = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2
            extent = np.sqrt(np.maximum(0, np.square(radii) - horizontal2))
            lower = np.max(delta[:, :, 2] - extent, axis=1)
            upper = np.min(delta[:, :, 2] + extent, axis=1)
            horizontal = np.all(horizontal2 <= np.square(radii) + 1e-12, axis=1)
            valid = horizontal & (lower <= upper + 1e-12)
            for actual, key in ((lower, "_root_delta_lower_m"), (upper, "_root_delta_upper_m"),
                                (valid, "_vertical_interval_valid")):
                equal(actual, arrays[label + key])
            equal(np.mean((excess > 1e-8).any(axis=1)), mode["either_ankle_outside_fraction"])
            equal(np.mean(~horizontal), mode["horizontal_impossible_fraction"])
            equal(np.mean(~valid), mode["no_common_vertical_interval_fraction"])
            stats(np.maximum(0., -upper[valid]), mode["necessary_lowering_m_on_valid_frames"])
            for i, details in enumerate(mode["sides"]):
                equal(np.mean(excess[:, i] > 1e-8), details["outside_reach_fraction"])
                equal(mask[:, i].sum(), details["stance_frames"], 0)
                stats(excess[:, i], details["excess_m"])
                stats(excess[mask[:, i], i], details["stance_excess_m"])
        frames += len(q)
    if any(identity(row["path"]) != row for row in inputs):
        raise ValueError("An input changed during independent checking")
    health = budget.check()
    result = {"status": "INDEPENDENT_WALKING_REACH_AUDIT_COMPLETED", "captured_utc": datetime.now(timezone.utc).isoformat(),
              "receipt": receipt_id, "source": identity(__file__), "numeric_checks": checks,
              "fresh_fk_frames": frames, "rows": len(report["rows"]), "all_inputs_unchanged": True,
              "root_transform_hip_centers_recomputed": True, "source_geometry_radius_recomputed": True,
              "new_ik_solves": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
              "health_after": health, "storage_status": "LOCAL_ONLY",
              "limits": "Independent saved-array, analytic transform, radius, interval and statistic checks; no proof that inner targets or root-Z corrections are dynamically feasible."}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    audit(parser.parse_args())
