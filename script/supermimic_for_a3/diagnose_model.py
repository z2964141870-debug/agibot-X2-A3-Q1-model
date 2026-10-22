#!/usr/bin/env python3
"""Measure frozen model errors on existing evaluated trajectories, without physics or training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    evaluation = json.loads(args.evaluation.read_text())
    checkpoint_path = Path(evaluation["checkpoint"]["path"])
    if identity(checkpoint_path) != evaluation["checkpoint"]:
        raise ValueError("Evaluation checkpoint changed")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = ck["manifest"]
    assets = A3Assets(args.root)
    if assets.manifest() != manifest["assets"]:
        raise ValueError("Asset contract changed")
    receipt_path = manifest["backend_receipt"]["path"]
    if identity(receipt_path) != manifest["backend_receipt"]:
        raise ValueError("Backend receipt changed")
    for source in json.loads(Path(receipt_path).read_text())["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError("Verified backend source changed")
    wcfg = dict(manifest["world_model"])
    wcfg["state_spec"] = StateSpecCfg(**wcfg["state_spec"])
    heads = assets.heads(args.device)
    wm = WorldModel(WorldModelCfg(**wcfg), heads, args.device).eval()
    wm.load_state_dict(ck["wm"], strict=True)
    wm.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for row in evaluation["rows"]:
            for key in ("reference", "trace"):
                if identity(row[key]["path"]) != row[key]:
                    raise ValueError(f"Evaluation {key} changed")
            lib = MotionLibrary([Path(row["reference"]["path"])], TRACKED, assets.q0,
                                device=args.device, active_indices=assets.active_indices)
            qpos, qvel = lib.ref_qpos_qvel(torch.tensor([0], device=args.device))
            with np.load(row["trace"]["path"], allow_pickle=False) as trace:
                states = torch.tensor(trace["state"], device=args.device, dtype=torch.float32)
                actions = torch.tensor(trace["actions"], device=args.device, dtype=torch.float32).clamp(-20., 20.)
            truth = torch.cat([torch.cat([qpos, qvel], -1), states], 0)
            n = len(states)
            metrics = {}
            for horizon in (1, 8, 32):
                starts = torch.arange(n - horizon + 1, device=args.device)
                learned, prior = truth[starts].clone(), truth[starts].clone()
                xi = torch.zeros(len(starts), 33, device=args.device)
                xi[:, 0] = 1.
                for t in range(horizon):
                    learned = wm.step(learned, actions[starts + t], xi)
                    prior = wm.spec.integrate(prior, torch.zeros(len(starts), 82, device=args.device))
                target = truth[starts + horizon]
                values = {}
                for name, prediction in (("learned", learned), ("constant_velocity_prior", prior)):
                    if not torch.isfinite(prediction).all():
                        raise ValueError(f"Nonfinite {name} {horizon}-step prediction")
                    active_error = (prediction[:, 7:42] - target[:, 7:42])[:, heads.active_indices]
                    values[name] = {"normalized_state_mse": float(wm.spec.normalized_sq_error(prediction, target).mean()),
                                    "root_position_rmse_m": float((prediction[:, :3] - target[:, :3]).square().sum(-1).mean().sqrt()),
                                    "active_joint_rmse_rad": float(active_error.square().mean().sqrt())}
                metrics[str(horizon)] = {"windows": len(starts), **values}
            rows.append({"clip": row["clip"], "split": row["split"], "executed_steps": n,
                         "trace": row["trace"], "reference": row["reference"], "horizons": metrics})
    groups = {}
    for split in ("training", "heldout"):
        selected = [row for row in rows if row["split"] == split]
        groups[split] = {}
        for horizon in ("1", "8", "32"):
            groups[split][horizon] = {name: {key: float(np.mean([row["horizons"][horizon][name][key] for row in selected]))
                                                for key in ("normalized_state_mse", "root_position_rmse_m", "active_joint_rmse_rad")}
                                       for name in ("learned", "constant_velocity_prior")}
    result = {"status": "FROZEN_MODEL_TRACE_DIAGNOSTIC_COMPLETED", "checkpoint": evaluation["checkpoint"],
              "evaluation": identity(args.evaluation), "entry": identity(__file__), "groups": groups, "rows": rows,
              "protocol": "All valid overlapping 1/8/32-step windows of saved deterministic final-policy traces; logged actions clamped as executed; friction 1 and zero bias/COM offset.",
              "aggregation": "Window-mean metrics within each clip, then equal clip means within each group.",
              "new_simulation_steps": 0, "optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
              "interpretation": "Offline prediction evidence, not independent causal identification or tracking success; reserved clips are now development diagnostics."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "groups": groups}, indent=2), flush=True)


if __name__ == "__main__":
    main()
