#!/usr/bin/env python3
"""Describe paired full-body traces and select arm references without policy scores."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_for_q1"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(record):
    from supermimic.q1_assets import identity
    path = Path(record["path"])
    if identity(path) != record:
        raise ValueError("Saved input identity changed: " + str(path))
    return path


def local_points(positions, quaternions, anchor=7):
    import torch
    from supermimic.kinematics.rotations import quat_apply_inverse
    p = torch.as_tensor(positions)
    q = torch.as_tensor(quaternions)
    rel = p - p[:, anchor:anchor + 1]
    rotation = q[:, anchor:anchor + 1].expand(-1, p.shape[1], -1)
    return quat_apply_inverse(rotation, rel).numpy()


def arm_profile(positions, quaternions):
    local = local_points(positions, quaternions)
    output = {}
    for side, shoulder, wrist in (("left", 8, 10), ("right", 11, 13)):
        height = local[:, wrist, 2] - local[:, shoulder, 2]
        output[side] = {
            "wrist_above_shoulder_peak_m": float(height.max()),
            "wrist_height_span_m": float(np.ptp(height)),
            "wrist_above_shoulder_fraction": float(np.mean(height >= 0.)),
            "peak_frame": int(np.argmax(height)),
        }
    return output


def select_arm_clips(rows):
    selected = []
    for side, key in (("left", "wrist_height_span_m"), ("right", "wrist_height_span_m"),
                      ("left", "wrist_above_shoulder_peak_m"),
                      ("right", "wrist_above_shoulder_peak_m")):
        candidates = sorted((r for r in rows if r["clip"] not in {s["clip"] for s in selected}),
                            key=lambda r: (-r["arms"][side][key], r["clip"]))
        if candidates:
            selected.append({**candidates[0], "selection_reason": side + "_" + key})
    return selected


def scan(args):
    from supermimic.q1_assets import TRACKED, identity
    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for item in manifest["motions"]:
        if not item.get("admitted_for_training"):
            continue
        path = args.motions / (item["clip"] + ".npz")
        if identity(path) != item["output"]:
            raise ValueError("Library reference identity changed")
        with np.load(path, allow_pickle=False) as ref:
            if ref["body_names"].tolist() != list(TRACKED):
                raise ValueError("Unexpected tracked-body order")
            row = {"clip": item["clip"], "reference": identity(path),
                   "frames": len(ref["qpos"]), "fps": float(ref["fps"]),
                   "arms": arm_profile(ref["body_pos_w"], ref["body_quat_w"])}
        rows.append(row)
    if len(rows) != 1024 or not all(r["clip"].startswith("train_") for r in rows):
        raise ValueError("This stage scans exactly the frozen 1024 train references")
    selected = select_arm_clips(rows)
    report = {"status": "ARM_REFERENCES_FROZEN_BEFORE_POLICY_EVALUATION", "source": identity(__file__),
              "reference_manifest": identity(manifest_path), "scanned_train_clips": len(rows),
              "selection_rule": "Distinct clips: maximum left height span, right height span, left peak height, right peak height, ties by clip name; reference torso-frame wrist minus shoulder z only; no policy results",
              "selected": selected, "coverage": {}, "new_physics_steps": 0,
              "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    for side in ("left", "right"):
        report["coverage"][side] = {
            "clips_with_any_wrist_above_shoulder": sum(r["arms"][side]["wrist_above_shoulder_peak_m"] >= 0 for r in rows),
            "span_quantiles_m": np.quantile([r["arms"][side]["wrist_height_span_m"] for r in rows], [0, .5, .9, 1]).tolist(),
        }
    write_json(args.output / "arm_reference_scan.json", {"rows": rows, **report})
    write_json(args.output / "arm_selection.json", report)
    print(json.dumps(report, indent=2), flush=True)


def normalized_terms(terms, expected_steps, max_reward):
    return {key: float(np.asarray(value, dtype=float).sum()) / (expected_steps * max_reward)
            for key, value in terms.items()}


def retarget_coverage(args):
    from collections import Counter
    from supermimic.q1_assets import identity
    manifest_path = args.motions / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_order = ["pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee",
                    "right_ankle", "spine3", "left_shoulder", "left_elbow", "left_wrist",
                    "right_shoulder", "right_elbow", "right_wrist"]
    rows, buckets = [], Counter()
    for item in manifest["motions"]:
        if not item.get("admitted_for_training"):
            continue
        with np.load(checked(item["kinematic_trace"]), allow_pickle=False) as data:
            names = data["source_body_names"].tolist()
            ids = [names.index(name) for name in source_order]
            source = arm_profile(data["source_position_m_world"][:, ids],
                                 data["source_orientation_wxyz_world"][:, ids])
            residual = np.linalg.norm(data["position_error_m"], axis=-1) if data["position_error_m"].ndim == 3 else data["position_error_m"].copy()
            wrist_residual = {side: float(np.quantile(residual[:, names.index(side + "_wrist")], .95))
                              for side in ("left", "right")}
        with np.load(checked(item["output"]), allow_pickle=False) as data:
            target = arm_profile(data["body_pos_w"], data["body_quat_w"])
        bucket = item.get("bucket", item.get("bones_source", {}).get("candidate_kind", "unknown"))
        buckets[bucket] += 1
        rows.append({"clip": item["clip"], "source": source, "q1": target,
                     "wrist_target_position_residual_p95_m": wrist_residual,
                     "source_csv": item.get("source_csv"), "kinematic_trace": item["kinematic_trace"]})
    summary = {"clips": len(rows), "candidate_buckets": dict(buckets), "arms": {}}
    for side in ("left", "right"):
        summary["arms"][side] = {
            "source_clips_any_wrist_above_shoulder": sum(r["source"][side]["wrist_above_shoulder_peak_m"] >= 0 for r in rows),
            "q1_clips_any_wrist_above_shoulder": sum(r["q1"][side]["wrist_above_shoulder_peak_m"] >= 0 for r in rows),
            "source_span_quantiles_m": np.quantile([r["source"][side]["wrist_height_span_m"] for r in rows], [0, .5, .9, 1]).tolist(),
            "q1_span_quantiles_m": np.quantile([r["q1"][side]["wrist_height_span_m"] for r in rows], [0, .5, .9, 1]).tolist(),
            "median_wrist_target_position_residual_p95_m": float(np.median([r["wrist_target_position_residual_p95_m"][side] for r in rows]))}
    report = {"status": "SOURCE_AND_Q1_UPPER_BODY_COVERAGE_DESCRIBED", "source": identity(__file__),
              "reference_manifest": identity(manifest_path), "summary": summary, "rows": rows,
              "limits": "Raw G1 and Q1 arm lengths differ. Source/Q1 absolute height span is not an amplitude ratio. IK wrist residual compares the actual Q1 with its mapped target, not human ground truth. No filters or reference changes.",
              "new_physics_steps": 0, "new_optimizer_updates": 0}
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


def kinematic_metrics(state, ref_qpos, body, ref_body, joint_names):
    count = len(state)
    error = state[:, 7:29].astype(float) - ref_qpos[:count, 7:29]
    output = {name + "_rmse_rad": float(np.sqrt(np.mean(error[:, ids] ** 2)))
              for name, ids in (("all_joints", slice(None)), ("legs", slice(0, 12)),
                                ("waist", slice(12, 14)), ("arms", slice(14, 22)))}
    output["upper_joint_rmse_rad"] = {joint_names[i]: float(np.sqrt(np.mean(error[:, i] ** 2)))
                                        for i in range(14, 22)}
    actual_local = local_points(body["pos"], body["quat"])
    target_local = local_points(ref_body["pos"][:count], ref_body["quat"][:count])
    output["arms_by_side"] = {}
    for side, shoulder, wrist, joints in (("left", 8, 10, slice(14, 18)),
                                         ("right", 11, 13, slice(18, 22))):
        actual_height = actual_local[:, wrist, 2] - actual_local[:, shoulder, 2]
        target_height = target_local[:, wrist, 2] - target_local[:, shoulder, 2]
        event = target_height >= 0.
        target_peak = int(np.argmax(target_height))
        actual_peak = int(np.argmax(actual_height))
        span = float(np.ptp(target_height))
        output["arms_by_side"][side] = {
            "wrist_torso_relative_rmse_m": float(np.sqrt(np.mean(np.sum(
                (actual_local[:, wrist] - target_local[:, wrist]) ** 2, axis=-1)))),
            "wrist_world_rmse_m": float(np.sqrt(np.mean(np.sum(
                (body["pos"][:, wrist] - ref_body["pos"][:count, wrist]) ** 2, axis=-1)))),
            "reference_height_peak_m": float(target_height.max()),
            "actual_height_peak_m": float(actual_height.max()),
            "actual_height_at_reference_peak_m": float(actual_height[target_peak]),
            "reference_height_span_m": span, "actual_height_span_m": float(np.ptp(actual_height)),
            "height_span_ratio": float(np.ptp(actual_height)) / span if span > 1e-8 else None,
            "reference_above_shoulder_frames": int(event.sum()),
            "synchronous_above_shoulder_fraction": float(np.mean(actual_height[event] >= 0.)) if event.any() else None,
            "reference_peak_frame": target_peak, "actual_peak_frame": actual_peak,
            "peak_time_difference_s": (actual_peak - target_peak) * .02,
            "joint_rmse_rad": float(np.sqrt(np.mean(error[:, joints] ** 2))),
        }
    return output


def analyze(args):
    import torch
    from contact_protocol import ContactCorrectedAssets
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.q1_assets import TRACKED, identity
    from supermimic.world_model.wm_trainer import reference_at
    torch.set_num_threads(2)
    assets = ContactCorrectedAssets(args.root)
    heads = assets.heads()
    evaluations = {}
    records = {}
    for label, path in args.evaluation:
        receipt = json.loads(Path(path).read_text())
        if receipt["assets"] != assets.manifest() or receipt["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
            raise ValueError("Evaluation contract changed")
        if label in evaluations:
            raise ValueError("Duplicate evaluation label")
        checked(receipt["checkpoint"])
        records[label] = identity(path)
        evaluations[label] = {r["clip"]: r for r in receipt["rows"]}
    if len(evaluations) != 2 or set(evaluations) != {"initial", "final"}:
        raise ValueError("Pair initial and final evaluations")
    if set(evaluations["initial"]) != set(evaluations["final"]):
        raise ValueError("Paired coverage changed")
    rows = []
    with torch.no_grad():
        for clip in sorted(evaluations["initial"]):
            pair = {label: evaluation[clip] for label, evaluation in evaluations.items()}
            if pair["initial"]["reference"] != pair["final"]["reference"]:
                raise ValueError("Paired references differ")
            path = checked(pair["initial"]["reference"])
            lib = MotionLibrary([path], TRACKED, assets.q0, device="cpu", active_indices=assets.active_indices)
            with np.load(path, allow_pickle=False) as ref:
                target_qpos = ref["qpos"].copy()
                ref_body = {"pos": ref["body_pos_w"].copy(), "quat": ref["body_quat_w"].copy()}
            common = min(r["executed_steps"] for r in pair.values())
            row = {"clip": clip, "reference": identity(path), "expected_steps": lib.num_frames,
                   "common_prefix_steps": common, "evaluations": {}}
            for label, r in pair.items():
                with np.load(checked(r["trace"]), allow_pickle=False) as trace:
                    state = torch.from_numpy(trace["state"].copy())
                    actions = torch.from_numpy(trace["actions"].copy()).clamp(-20, 20)
                    rewards = trace["reward"].copy()
                if len(state) != r["executed_steps"] or len(target_qpos) != r["expected_steps"]:
                    raise ValueError("Saved step accounting changed")
                previous = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]])
                body_t = heads.robot_bodies(state)
                terms_t = heads.reward(state, actions, previous, reference_at(lib, torch.arange(len(state))), body_t)
                terms = {k: v.numpy() for k, v in terms_t.items()}
                max_error = float(np.max(np.abs(terms["total"] - rewards)))
                if max_error > 2e-5:
                    raise ValueError("Reward replay disagrees with saved evaluation")
                body = {k: v.numpy() for k, v in body_t.items()}
                full = normalized_terms(terms, lib.num_frames, heads.reward_cfg.max_step_reward)
                if abs(full["total"] - r["normalized_return"]) > 2e-5:
                    raise ValueError("Return denominator changed")
                row["evaluations"][label] = {
                    "trace": r["trace"], "executed_steps": len(state), "failed": r["failed"],
                    "full_clip_completion": r["full_clip_completion"],
                    "full_reference_normalized_reward_terms": full,
                    "common_prefix_normalized_reward_terms": normalized_terms(
                        {k: v[:common] for k, v in terms.items()}, common, heads.reward_cfg.max_step_reward),
                    "root_position_score_below_0p01_fraction": float(np.mean(
                        terms["motion_global_root_pos"] / (heads.reward_cfg.dt * heads.reward_cfg.w_root_pos) < .01)),
                    "recorded_reward_max_abs_error": max_error,
                    "executed_kinematics": kinematic_metrics(state.numpy(), target_qpos, body, ref_body, assets.names),
                    "common_prefix_kinematics": kinematic_metrics(state[:common].numpy(), target_qpos,
                        {k: v[:common] for k, v in body.items()}, ref_body, assets.names)}
            rows.append(row)
    summary = {}
    for label in evaluations:
        kinematics = [row["evaluations"][label]["common_prefix_kinematics"] for row in rows]
        summary[label] = {
            "clips": len(rows),
            "mean_common_prefix_joint_rmse_rad": {key: float(np.mean([m[key] for m in kinematics]))
                for key in ("all_joints_rmse_rad", "legs_rmse_rad", "waist_rmse_rad", "arms_rmse_rad")},
            "mean_common_prefix_wrist_torso_relative_rmse_m": {side: float(np.mean([
                m["arms_by_side"][side]["wrist_torso_relative_rmse_m"] for m in kinematics])) for side in ("left", "right")},
            "mean_full_reference_reward_terms": {key: float(np.mean([
                row["evaluations"][label]["full_reference_normalized_reward_terms"][key] for row in rows]))
                for key in rows[0]["evaluations"][label]["full_reference_normalized_reward_terms"]},
            "mean_root_position_score_below_0p01_fraction": float(np.mean([
                row["evaluations"][label]["root_position_score_below_0p01_fraction"] for row in rows]))}
    report = {"status": "PAIRED_FULLBODY_ANALYSIS_COMPLETED", "source": identity(__file__),
              "evaluations": records, "summary": summary, "rows": rows,
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "limits": "Training clips only. Errors reported both executed and common-prefix. Root/body post-control alignment matches original evaluator. Peak time is descriptive, not a completed gesture classifier. Above-shoulder event is anatomical description, not data admission.",
              "storage_status": "LOCAL_ONLY"}
    write_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    import torch
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "analyze", "retarget"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path)
    parser.add_argument("--evaluation", nargs=2, action="append", metavar=("LABEL", "RECEIPT"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "scan":
        scan(args)
    elif args.mode == "retarget":
        retarget_coverage(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
