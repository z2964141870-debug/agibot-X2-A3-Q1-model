#!/usr/bin/env python3
"""Independently recompute saved P12 geometry, native derivatives and inverse forces."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
from guard_phuma_batch import MemoryBoundBudget
from runtime_guard import StorageBudget


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def audit(args):
    import mujoco as mj

    sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))
    from prepare_walking_holdout import load_verifier
    verifier = load_verifier()
    root, receipt_path, output = args.root.resolve(strict=True), args.receipt.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("A new independent audit file is required")
    budget = MemoryBoundBudget(StorageBudget(receipt_path.parent.parent,
        root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.check()
    receipt_id = identity(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    guard = json.loads((receipt_path.parent.parent / "guard.json").read_text())
    if (receipt["status"] != "TRAIN_ONLY_WALKING_CONTACT_DIAGNOSTIC_COMPLETED"
            or guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"]):
        raise ValueError("Only a completed and reaped diagnostic can be audited")
    inputs = [receipt_id, *receipt["frozen_inputs"]]
    if any(identity(item["path"]) != item for item in inputs):
        raise ValueError("A bound input changed before independent checking")
    assets = verifier.ContactCorrectedAssets(root)
    model, data = assets.model, mj.MjData(assets.model)
    feet = [np.flatnonzero((model.geom_bodyid == model.body(side + "_ankle_roll_link").id)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))) for side in ("left", "right")]
    mapping = json.loads((HERE.parent / "supermimic_q1_phuma/retarget_mapping.json").read_text())["body_map"]
    bids = [model.body(name).id for name in mapping.values()]
    nonfeet = [i for i, name in enumerate(mapping) if name not in ("left_ankle", "right_ankle")]
    rows, maxima, checks, inverse_calls = [], {}, 0, 0
    old_path = next(Path(item["path"]) for item in inputs
        if item["path"].endswith("PHUMA_P11_walking_velocity_probe_20261025/probe/receipt.json"))
    old = {row["member"]: row for row in json.loads(old_path.read_text())["rows"]}

    def equal(name, actual, expected, tolerance=1e-8):
        nonlocal checks
        error = float(np.abs(np.asarray(actual) - np.asarray(expected)).max())
        maxima[name] = max(maxima.get(name, 0.), error)
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f"Independent mismatch {name}: {error} > {tolerance}")
        checks += 1

    for clip in receipt["rows"]:
        if clip["source_split"] != "train":
            raise ValueError("A held-out source entered the train-only diagnostic")
        with np.load(old[clip["member"]]["arrays"]["path"], allow_pickle=False) as saved:
            original_roots = {bound: saved["qpos_bound" + str(bound)][:, :7].copy() for bound in (2, 8)}
        for result in clip["variants"]:
            if result["status"] != "COMPLETED":
                rows.append({"member": clip["member"], "variant": result["variant"], "status": "NOT_COMPLETED_NO_AUDIT"})
                continue
            records = [result["arrays"], result["dynamics"]["raw_arrays"]]
            if any(identity(item["path"]) != item for item in records):
                raise ValueError("Diagnostic array bytes changed")
            inputs.extend(records)
            with np.load(records[0]["path"], allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            with np.load(records[1]["path"], allow_pickle=False) as saved:
                dynamics = {name: saved[name].copy() for name in saved.files}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Nonfinite saved array")
            q, v, t = arrays["qpos"], arrays["qvel"], arrays["time_s"]
            expected_root = original_roots[result["bound_rad_s"]].copy()
            expected_root[:, 2] += float(arrays["common_ground_origin_shift_m"])
            equal("original_root_preserved", q[:, :7], expected_root, 1e-8)
            equal("time", t, np.arange(len(t)) * .02, 1e-12)
            equal("nonfoot_tasks", arrays["task_position_m"][:, nonfeet],
                  arrays["original_task_position_m"][:, nonfeet], 0)
            equal("nonfoot_rotations", arrays["task_rotation_matrix"][:, nonfeet],
                  arrays["original_task_rotation_matrix"][:, nonfeet], 0)
            velocity = np.empty_like(v)
            for frame in range(len(q) - 1):
                mj.mj_differentiatePos(model, velocity[frame], .02, q[frame], q[frame + 1])
            velocity[-1] = velocity[-2]
            equal("qvel", velocity, v, 1e-10)
            qacc = {"central3": (v[3:-1] - v[1:-3]) / .04,
                    "central5": (-v[4:] + 8 * v[3:-1] - 8 * v[1:-3] + v[:-4]) / .24}
            indices = np.arange(2, len(q) - 2)
            equal("inverse_frame_indices", indices, dynamics["frame_index"], 0)
            for label in qacc:
                equal("qacc_" + label, qacc[label], dynamics[label + "_qacc"], 1e-10)
            gaps, penetration, residual, nominal = [], [], [], []
            for frame in range(len(q)):
                if frame % 100 == 0:
                    budget.memory()
                data.qpos[:], data.qvel[:] = q[frame], v[frame]
                mj.mj_forward(model, data)
                gaps.append([verifier.foot_bottom(model, data, ids) for ids in feet])
                penetration.append(max([0.] + [-float(c.dist) for c in data.contact
                    if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]]))
                shift = float(arrays["common_ground_origin_shift_m"])
                expected = arrays["task_position_m"][frame].copy()
                original = arrays["original_task_position_m"][frame].copy()
                expected[:, 2] += shift
                original[:, 2] += shift
                residual.append(data.xpos[bids] - expected)
                nominal.append(data.xpos[bids] - original)
                if 2 <= frame < len(q) - 2:
                    offset = frame - 2
                    for label in qacc:
                        data.qacc[:] = qacc[label][offset]
                        mj.mj_inverse(model, data)
                        equal("fresh_inverse_" + label, data.qfrc_inverse,
                              dynamics[label + "_inverse"][offset], 1e-7)
                        inverse_calls += 1
            gaps = np.asarray(gaps)
            equal("fresh_foot_heights", gaps, arrays["all_frame_foot_height_m"], 1e-9)
            equal("fresh_self_penetration", penetration, arrays["self_penetration_m"], 1e-9)
            equal("fresh_task_residual", residual, arrays["task_residual_m"], 1e-9)
            equal("fresh_nominal_residual", nominal, arrays["nominal_residual_m"], 1e-9)
            equal("inverse_foot_heights", gaps[indices], dynamics["foot_gap_m"], 1e-9)
            low = gaps.min(axis=1)
            for name, actual in {"support_within_15mm_fraction": np.mean(low <= .015),
                "support_gap_max_m": low.max(), "foot_floor_min_m": low.min(),
                "self_penetration_max_m": max(penetration), "joint_speed_max_rad_s": np.abs(v[:, 6:]).max(),
                "nominal_cartesian_residual_p95_m": np.quantile(np.linalg.norm(nominal, axis=-1), .95)}.items():
                equal(name, actual, result[name], 1e-9)
            weight = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
            force = dynamics["central3_inverse"]
            equal("inverse_root_median", np.median(np.linalg.norm(force[:, :3], axis=1) / weight),
                result["dynamics"]["inverse"]["central3"]["root_force_body_weights"]["p50"], 1e-10)
            equal("inverse_effort_fraction", ((np.abs(force[:, 6:]) / assets.effort) > 1).any(axis=1).mean(),
                result["dynamics"]["inverse"]["central3"]["frames_any_active_effort_over_cap_fraction"], 0)
            rows.append({"member": clip["member"], "variant": result["variant"], "bound_rad_s": result["bound_rad_s"],
                         "status": "SAVED_ARRAYS_AND_FRESH_FK_INVERSE_MATCH", "frames": len(q)})
    if any(identity(item["path"]) != item for item in inputs):
        raise ValueError("A bound input changed during independent checking")
    budget.check()
    report = {"status": "INDEPENDENT_WALKING_CONTACT_AUDIT_COMPLETED", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "receipt": receipt_id, "rows": rows, "numeric_checks": checks, "max_abs_error_by_check": maxima,
        "fresh_inverse_calls": inverse_calls, "new_physics_steps": 0, "new_optimizer_updates": 0,
        "source": identity(__file__), "storage_status": "LOCAL_ONLY",
        "limits": "Independent geometry and MuJoCo inverse recomputation, not a repeated IK, policy evaluation, or dynamics rollout."}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": checks, "fresh_inverse_calls": inverse_calls,
                      "output": identity(output)}, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    audit(parser.parse_args())
