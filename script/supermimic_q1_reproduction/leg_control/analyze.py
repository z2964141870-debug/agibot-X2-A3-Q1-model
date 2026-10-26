#!/usr/bin/env python3
"""Reconstruct contact support and PD saturation from a saved Q1 evaluation."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mujoco
    import numpy as np
    import torch
    from control import ContactCorrectedAssets, Leg50Assets, VARIANT, leg_mask
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.q1_assets import TRACKED, identity
    from supermimic.q1_sim import Q1Sim, SimCfg
    torch.set_num_threads(2)
    receipt = json.loads(args.evaluation.read_text())
    if receipt["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("Analyze a completed saved physical evaluation")
    variant = receipt["assets"].get("control_variant", {}).get("version")
    if variant not in (None, VARIANT):
        raise ValueError("Unknown saved Q1 control variant")
    assets = (Leg50Assets if variant else ContactCorrectedAssets)(args.root)
    if assets.manifest() != receipt["assets"]:
        raise ValueError("Saved evaluation and analysis control contracts differ")
    args.output.mkdir(parents=True, exist_ok=False)
    legs = leg_mask(assets.names)
    rows, frozen = [], [identity(args.evaluation)]
    for row in receipt["rows"]:
        for key in ("reference", "trace"):
            if identity(row[key]["path"]) != row[key]:
                raise ValueError("Saved physical evidence changed")
            frozen.append(row[key])
        with np.load(row["trace"]["path"], allow_pickle=False) as trace:
            states, actions = trace["state"].copy(), trace["actions"].copy()
            initial = trace["initial_state"].copy()
        lib = MotionLibrary([row["reference"]["path"]], TRACKED, assets.q0,
                            device="cpu", active_indices=assets.active_indices)
        sim = Q1Sim(assets, lib, SimCfg(num_envs=1, threads=1, domain_randomization=False,
            observation_noise=False, action_noise=0.), assets.heads(), seed=0)
        model = sim.models[0]
        sim.close()
        probe = mujoco.MjData(model)
        floor = model.geom("floor").id
        side = {int(g): int(model.body(int(model.geom_bodyid[g])).name.startswith("right"))
                for g in assets.foot_geoms}
        targets = assets.q0 + np.clip(actions, -20., 20.) * assets.scale
        previous = np.concatenate([initial[None], states[:-1]])
        ratio = np.abs(assets.kp * (targets - previous[:, assets.qadr]) -
                       assets.kd * previous[:, 29 + assets.vadr]) / assets.effort
        forces = []
        for state, target in zip(states, targets):
            probe.qpos[:], probe.qvel[:] = state[:29], state[29:]
            probe.ctrl[assets.aids] = target
            mujoco.mj_forward(model, probe)
            support = np.zeros(2)
            for index, contact in enumerate(probe.contact):
                other = int(contact.geom2) if contact.geom1 == floor else (
                    int(contact.geom1) if contact.geom2 == floor else -1)
                if other in side:
                    wrench = np.empty(6)
                    mujoco.mj_contactForce(model, probe, index, wrench)
                    support[side[other]] += max(0., float(wrench[0]))
            forces.append(support)
        forces = np.asarray(forces)
        if len(states) != row["executed_steps"] or not np.isfinite(forces).all():
            raise ValueError("Invalid saved support reconstruction")
        supported = forces > 1.
        trace_path = args.output / (row["clip"] + "_support.npz")
        np.savez_compressed(trace_path, post_boundary_normal_force_n=forces,
                            pre_boundary_demand_to_cap=ratio)
        rows.append({"clip": row["clip"], "steps": len(states), "reference": row["reference"],
            "leg_boundary_saturation_fraction": float((ratio[:, legs] > 1. + 1e-10).mean()),
            "joint_saturation_fraction": {name: float((ratio[:, j] > 1. + 1e-10).mean())
                                          for j, name in enumerate(assets.names)},
            "double_support_boundary_fraction": float(supported.all(axis=1).mean()),
            "no_support_boundary_fraction": float((~supported.any(axis=1)).mean()),
            "left_only_support_boundary_fraction": float((supported[:, 0] & ~supported[:, 1]).mean()),
            "right_only_support_boundary_fraction": float((supported[:, 1] & ~supported[:, 0]).mean()),
            "support_state_changes": int(np.any(supported[1:] != supported[:-1], axis=1).sum()),
            "trace": identity(trace_path)})
    if frozen != [identity(record["path"]) for record in frozen]:
        raise ValueError("Saved evaluation changed during analysis")
    result = {"status": "Q1_SAVED_LEG_SUPPORT_ANALYSIS_COMPLETED", "source": identity(__file__),
        "evaluation": frozen[0], "assets": assets.manifest(), "rows": rows,
        "support_threshold_n": 1., "new_physics_integration_steps": 0,
        "new_policy_inference": 0, "new_optimizer_updates": 0,
        "limits": "Contact force is reconstructed at each post-control boundary; PD demand is sampled at the pre-control boundary. These are not substep peaks or a walking success classifier.",
        "storage_status": "LOCAL_ONLY"}
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
