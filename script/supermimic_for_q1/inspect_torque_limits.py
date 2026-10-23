#!/usr/bin/env python3
"""Measure PD saturation at saved R01 pre-control states, without integration."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import Q1Assets, TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluations", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    assets = Q1Assets(args.root)
    rows = []
    for evaluation_path in args.evaluations:
        evaluation = json.loads(evaluation_path.read_text())
        if identity(evaluation["checkpoint"]["path"]) != evaluation["checkpoint"]:
            raise ValueError("Checkpoint changed")
        ck = torch.load(evaluation["checkpoint"]["path"], mmap=True,
                        map_location="cpu", weights_only=False)
        if ck["manifest"]["assets"] != assets.manifest():
            raise ValueError("R01 control protocol changed")
        del ck
        for row in evaluation["rows"]:
            for key in ("trace", "reference"):
                if identity(row[key]["path"]) != row[key]:
                    raise ValueError("Preserved input changed")
            with np.load(row["reference"]["path"], allow_pickle=False) as reference:
                initial = np.concatenate([reference["qpos"][0], reference["qvel"][0]])
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                actions = trace["actions"].copy()
                previous_states = np.concatenate([initial[None], trace["state"][:-1]], axis=0)
            lib = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                                device="cpu", active_indices=assets.active_indices)
            sim = Q1Sim(assets, lib, SimCfg(num_envs=1, threads=1, domain_randomization=False,
                        observation_noise=False, action_noise=0.), assets.heads(), 20261023)
            model = sim.models[0]
            data = mujoco.MjData(model)
            demands, applied, max_error = [], [], 0.
            try:
                for state, action in zip(previous_states, actions, strict=True):
                    target = assets.q0 + np.clip(action, -20., 20.) * assets.scale
                    data.qpos[:], data.qvel[:] = state[:29], state[29:]
                    data.ctrl[assets.aids] = target
                    data.time = 0.
                    mujoco.mj_forward(model, data)
                    if data.time != 0.:
                        raise AssertionError("Unexpected physics integration")
                    raw = assets.kp * (target - state[assets.qadr]) - assets.kd * state[29 + assets.vadr]
                    actual = data.actuator_force[assets.aids].copy()
                    error = float(np.abs(actual - np.clip(raw, -assets.effort, assets.effort)).max())
                    max_error = max(max_error, error)
                    if error > 1e-8:
                        raise AssertionError("Analytic PD demand differs from native limited actuator force")
                    demands.append(raw)
                    applied.append(actual)
            finally:
                sim.close()
            demands, applied = np.asarray(demands), np.asarray(applied)
            ratio = np.abs(demands) / assets.effort
            saturated = ratio > 1. + 1e-10
            joint_rows = [{"joint": name, "cap_nm": float(assets.effort[i]),
                           "boundary_saturation_fraction": float(saturated[:, i].mean()),
                           "demand_to_cap_p50": float(np.quantile(ratio[:, i], .5)),
                           "demand_to_cap_p95": float(np.quantile(ratio[:, i], .95)),
                           "max_abs_demand_nm": float(np.abs(demands[:, i]).max())}
                          for i, name in enumerate(assets.names)]
            legs = np.array([any(term in name for term in ("hip", "knee", "ankle")) for name in assets.names])
            result = {"clip": row["clip"], "training_step": evaluation["training_step"],
                      "evaluation": identity(evaluation_path), "trace": row["trace"], "reference": row["reference"],
                      "control_boundaries": len(actions), "native_pd_max_abs_error_nm": max_error,
                      "all_joints_saturation_fraction": float(saturated.mean()),
                      "leg_saturation_fraction": float(saturated[:, legs].mean()),
                      "arm_saturation_fraction": float(saturated[:, ~legs].mean()),
                      "boundaries_with_any_saturation_fraction": float(saturated.any(axis=1).mean()),
                      "policy_action_clip_fraction": float((np.abs(actions) > 20.).mean()),
                      "joints": joint_rows}
            rows.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "joints"}), flush=True)
    receipt = {"status": "SAVED_BOUNDARY_TORQUE_AUDIT_COMPLETED", "rows": rows,
               "source": identity(__file__), "assets": assets.manifest(),
               "source_files": [identity(Path(__file__).parent / "supermimic" / name)
                                for name in ("q1_assets.py", "q1_sim.py")],
               "new_physics_steps": 0, "new_policy_calls": 0, "optimizer_updates": 0,
               "limits": "Pre-control boundaries only. Does not measure within-interval saturation, required feasible motion torques, or actual hardware capability. R01 effective contact friction remains0.6.",
               "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
