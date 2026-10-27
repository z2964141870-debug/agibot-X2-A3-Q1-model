"""Inspect saved RF09 joint errors and candidate reward scales without Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def identity(path):
    path = Path(path)
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def summarize(rows, joint_names):
    mean = statistics.fmean
    mse = sum(sum(x*x for x in r["joint_rmse_rad"][14:]) for r in rows)
    bias2 = sum(sum(x*x for x in r["joint_bias_rad"][14:]) for r in rows)
    by_joint = []
    for joint in range(14, 22):
        by_joint.append({
            "name": joint_names[joint],
            "mean_rmse_rad": mean(r["joint_rmse_rad"][joint] for r in rows),
            "mean_absolute_clip_bias_rad": mean(abs(r["joint_bias_rad"][joint]) for r in rows),
            "reference_soft_limit_frame_fraction": mean(r["reference_soft_limit_frame_fraction"][joint] for r in rows),
            "actual_range_rad": mean(r["actual_upper_range_rad"][joint-14] for r in rows),
            "reference_range_rad": mean(r["reference_upper_range_rad"][joint-14] for r in rows),
        })
    return {
        "clips": len(rows),
        "upper_bias_squared_fraction_of_mse": bias2 / mse,
        "upper_joint_rmse_rad": mean(r["upper_limb_rmse_rad"] for r in rows),
        "upper_target_reference_rmse_rad": mean(r["upper_joint_target_rmse_rad"] for r in rows),
        "upper_command_response_rmse_rad": mean(r["upper_joint_command_response_rmse_rad"] for r in rows),
        "upper_reference_soft_limit_fraction": mean(j["reference_soft_limit_frame_fraction"] for j in by_joint),
        "max_per_joint_reference_soft_limit_fraction": max(j["reference_soft_limit_frame_fraction"] for j in by_joint),
        "upper_actual_range_rad": mean(j["actual_range_rad"] for j in by_joint),
        "upper_reference_range_rad": mean(j["reference_range_rad"] for j in by_joint),
        "joints": by_joint,
        "weighting": "Each clip and upper joint has equal weight; these are saved executed prefixes. Bias fraction sums per-clip squared bias divided by per-clip squared RMSE, not a claim about a shared bias direction across clips.",
    }


def gaussian_sensitivity(angle, width):
    value = math.exp(-(angle / width)**2)
    return {"reward_fraction": value, "slope_magnitude_per_radian": 2*angle/(width*width)*value}


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.diagnosis.read_text())
    groups = {}
    for split in ("train", "validation"):
        for group in ("all", "walking", "gesture"):
            rows = [r for r in source["per_clip"] if r["split"] == split and (group == "all" or r["group"] == group)]
            if rows:
                groups[split+"/"+group] = summarize(rows, source["joint_names"])
    sensitivity = [{"illustrative_equivalent_angle_rad": angle,
        "std_0_4": gaussian_sensitivity(angle, .4),
        "std_0_8": gaussian_sensitivity(angle, .8),
        "std_1_2": gaussian_sensitivity(angle, 1.2)}
        for angle in (.1, .3, .5, .8, 1., 1.2, 1.5, 2., 3.)]
    scales = sorted(set(source["control"]["action_scale"]))
    smoothness = [{"action_scale_rad": alpha,
        "physical_sum_coefficient_equivalent_to_original": .1/(alpha*alpha),
        "physical_sum_coefficient_0_1_penalty_ratio": alpha*alpha,
        "physical_mean_coefficient_0_1_penalty_ratio": alpha*alpha/22} for alpha in scales]
    result = {
        "status": "SAVED_PREFIX_OFFLINE_SIGNAL_ANALYSIS_COMPLETED",
        "sources": [identity(args.diagnosis), identity(__file__)],
        "physics_integration_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
        "summary": groups,
        "gaussian_shape_examples": sensitivity,
        "smoothness_unit_conversion": smoothness,
        "limits": [
            "No new trajectories or policy improvements were measured.",
            "Joint-angle RMSE is not body rotation RMSE. Gaussian examples illustrate shapes only; no nonlinear reward is reconstructed from aggregate errors.",
            "Soft-limit metrics use original saved prefixes; they do not establish every retargeted clip is dynamically feasible.",
            "Actual command-target tracking error does not establish PD response bandwidth or torque reserves.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "output": identity(args.output),
        "summary": {k: {a: b for a, b in v.items() if a != "joints"} for k, v in groups.items()},
        "smoothness_unit_conversion": smoothness}, indent=2))


if __name__ == "__main__":
    main()
