#!/usr/bin/env python3
"""Compare learned dynamics with constant velocity on saved physical trajectories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.kinematics.lab_math import quat_box_minus
from supermimic.q1_assets import Q1Assets, identity
from supermimic.robot_spec import XI_DIM
from supermimic.world_model.state_spec import DELTA_DIM
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def metrics(spec, predicted, actual):
    return {"root_position_coordinate_rmse_m": float((predicted[:, :3] - actual[:, :3]).square().mean().sqrt()),
            "root_orientation_rmse_rad": float(quat_box_minus(predicted[:, 3:7], actual[:, 3:7]).square().sum(-1).mean().sqrt()),
            "joint_position_rmse_rad": float((predicted[:, 7:29] - actual[:, 7:29]).square().mean().sqrt()),
            "root_linear_velocity_coordinate_rmse_m_s": float((predicted[:, 29:32] - actual[:, 29:32]).square().mean().sqrt()),
            "root_angular_velocity_coordinate_rmse_rad_s": float((predicted[:, 32:35] - actual[:, 32:35]).square().mean().sqrt()),
            "joint_velocity_rmse_rad_s": float((predicted[:, 35:] - actual[:, 35:]).square().mean().sqrt()),
            "normalized_state_mse": float(spec.normalized_sq_error(predicted, actual).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.stride < 1:
        raise ValueError("Use a new output path and a positive stride")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    assets = Q1Assets(args.root)
    if assets.manifest() != checkpoint["manifest"]["assets"]:
        raise ValueError("Q1 model or control contract changed")
    heads = assets.heads(args.device)
    config = dict(checkpoint["manifest"]["world_model"])
    from supermimic.world_model.state_spec import StateSpecCfg

    config["state_spec"] = StateSpecCfg(**config["state_spec"])
    model = WorldModel(WorldModelCfg(**config), heads, args.device).eval()
    model.load_state_dict(checkpoint["wm"], strict=True)
    evaluation = json.loads(args.evaluation.read_text())
    if identity(args.checkpoint)["sha256"] != evaluation["checkpoint"]["sha256"]:
        raise ValueError("World-model checkpoint differs from the evaluated policy checkpoint")
    del checkpoint
    results = []
    with torch.no_grad():
        for row in evaluation["rows"]:
            for key in ("reference", "trace"):
                if identity(row[key]["path"])["sha256"] != row[key]["sha256"]:
                    raise ValueError("Preserved reference or trace changed")
            with np.load(row["reference"]["path"], allow_pickle=False) as reference:
                initial = np.concatenate([reference["qpos"][0], reference["qvel"][0]])
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                states = torch.as_tensor(np.concatenate([initial[None], trace["state"]], axis=0), device=args.device)
                actions = torch.as_tensor(trace["actions"], device=args.device).clamp(-20., 20.)
            for horizon in (1, 8, 32):
                starts = torch.arange(0, len(actions) - horizon + 1, args.stride, device=args.device)
                if not len(starts):
                    results.append({"clip": row["clip"], "split": row["split"], "horizon": horizon,
                                    "windows": 0, "status": "INSUFFICIENT_SAVED_STEPS"})
                    continue
                predicted = states[starts].clone()
                constant = predicted.clone()
                context = torch.zeros(len(starts), XI_DIM, device=args.device)
                context[:, 0] = 1.
                zero = torch.zeros(len(starts), DELTA_DIM, device=args.device)
                for offset in range(horizon):
                    predicted = model.step(predicted, actions[starts + offset], context)
                    constant = model.spec.integrate(constant, zero)
                actual = states[starts + horizon]
                learned = metrics(model.spec, predicted, actual)
                baseline = metrics(model.spec, constant, actual)
                result = {"clip": row["clip"], "split": row["split"], "horizon": horizon,
                          "seconds": horizon * .02, "windows": len(starts),
                          "learned": learned, "constant_velocity": baseline,
                          "learned_to_cv_ratio": {key: learned[key] / baseline[key] if baseline[key] else None
                                                   for key in learned}}
                results.append(result)
                print(json.dumps(result, allow_nan=False), flush=True)
    receipt = {"status": "SAVED_TRAJECTORY_MODEL_COMPARISON_COMPLETED", "rows": results,
               "checkpoint": identity(args.checkpoint), "evaluation": identity(args.evaluation),
               "source": identity(__file__), "stride": args.stride, "new_physics_steps": 0,
               "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
               "protocol": "Actual saved policy actions and initial states; friction1/encoder bias0/COM offset0; overlapping windows; no post-terminal targets",
               "limits": "Post-training diagnostic on the same policy evaluation traces; development evidence, not an independent model test set"}
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": identity(args.output)}), flush=True)


if __name__ == "__main__":
    main()
