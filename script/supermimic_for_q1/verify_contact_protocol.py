#!/usr/bin/env python3
"""Verify the corrected contact adapter through actual Q1Sim resets and physics."""

import argparse
import copy
import json
from pathlib import Path
import traceback

import mujoco
import numpy as np
import torch

from audit_contacts import inspect_contacts
from contact_protocol import ContactCorrectedAssets
from verify_backend import check_fk, check_graph, diagnostic_reference
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, GOAL_DIM, OBS_DIM, STATE_DIM, XI_DIM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261023)
    result = {"status": "FAILED", "physics_control_steps": 0,
              "optimizer_updates": 0, "hardware_control": False, "storage_status": "LOCAL_ONLY"}
    sim = None
    try:
        assets = ContactCorrectedAssets(args.root)
        result["assets"] = assets.manifest()
        result["fk"] = check_fk(assets)
        diagnostic_reference(assets, args.output / "diagnostic_only.npz")
        heads = assets.heads()
        lib = MotionLibrary([args.output / "diagnostic_only.npz"], TRACKED, assets.q0,
                            device="cpu", active_indices=assets.active_indices)
        sim = Q1Sim(assets, lib, SimCfg(num_envs=2, threads=2, domain_randomization=True,
                                      observation_noise=False, action_noise=0.), heads, 20261023)
        floor = assets.model.geom("floor").id
        feet = set(int(x) for x in assets.foot_geoms)
        pose = assets.model.qpos0.copy()
        pose[assets.qadr] = assets.q0
        pose[2] -= .01
        checks = []
        for _ in range(3):
            sim.reset(sim.ids, torch.zeros(2, dtype=torch.long))
            for i, model in enumerate(sim.models):
                requested = float(sim.context[i, 0])
                contacts = inspect_contacts(model, pose, feet, floor)
                effective = sorted({float(c["effective_friction"][0]) for c in contacts})
                if effective != [requested]:
                    raise AssertionError("Actual Q1Sim reset context/contact mismatch")
                checks.append({"requested": requested, "effective": effective,
                               "contacts": len(contacts)})
        result["randomized_reset_contacts"] = checks
        sim.cfg.domain_randomization = False
        sim.reset(sim.ids, torch.zeros(2, dtype=torch.long))
        contacts = inspect_contacts(sim.models[0], pose, feet, floor)
        result["deterministic_contact_friction"] = sorted({float(c["effective_friction"][0]) for c in contacts})
        if result["deterministic_contact_friction"] != [1.]:
            raise AssertionError("Deterministic evaluation must have actual friction1")
        initial = sim.state[0].numpy().copy()
        action = torch.full((2, ACT_DIM), .05)
        model = copy.copy(sim.models[0])
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = initial[:29], initial[29:]
        data.ctrl[assets.aids] = assets.q0 + action[0].numpy() * assets.scale
        for _ in range(20):
            mujoco.mj_step(model, data)
        actual = sim.step(action)["state"][0].numpy()
        result["physics_control_steps"] += 2
        error = float(np.abs(actual - np.concatenate([data.qpos, data.qvel])).max())
        if error > 1e-5 or not np.isclose(data.time, .02):
            raise AssertionError("Corrected native/explicit PD disagreement")
        result["native_explicit_state_max_abs"] = error
        replay = ReplayBuffer(2, 200, STATE_DIM, ACT_DIM, XI_DIM, "cpu")
        for _ in range(40):
            sim.step(torch.randn(2, ACT_DIM) * .05, replay)
            result["physics_control_steps"] += 2
        if (sim.policy_inputs()[0].shape != (2, OBS_DIM * 10)
                or sim.policy_inputs()[1].shape != (2, GOAL_DIM)):
            raise AssertionError("Corrected Q1 policy input dimensions changed")
        graph_lib = MotionLibrary([args.output / "diagnostic_only.npz"], TRACKED, assets.q0,
                                  device="cpu", dtype=torch.float64, active_indices=assets.active_indices)
        result["graph"] = check_graph(assets, graph_lib, replay)
        result["status"] = "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if sim is not None:
            sim.close()
        package = Path(__file__).parent
        paths = sorted((package / "supermimic").rglob("*.py"))
        paths += [package / name for name in ("contact_protocol.py", "verify_backend.py", "audit_contacts.py")]
        paths.append(Path(__file__))
        result["source_files"] = [identity(path) for path in paths]
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("source_files", "assets", "traceback")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
