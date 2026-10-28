"""Zero-optimizer restore, gait metric and evaluation-preservation checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from base import prepare, check_contract, make_learner, preserve_hp, enrich, rollout, gait_summary
from base import preserve_random_state, fixed_blocks, save, stamp, identity, process_identity


def equal_tree(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("Restored dictionary keys changed")
        for key in actual:
            equal_tree(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError("Restored sequence length changed")
        for a, b in zip(actual, expected):
            equal_tree(a, b)
    elif actual != expected:
        raise AssertionError("Restored scalar changed")


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(),
              "num_envs": a.envs, "optimizer_steps": 0}
    save(a.output / "status.json", status)
    env = independent = None
    try:
        torch.set_num_threads(8)
        _, env, cfg = prepare(a, num_envs=a.envs)
        status["physical_contract"] = check_contract(env)
        learner = make_learner(env, cfg, a.output, a.checkpoint)
        original = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        for name, module in (("pi", learner.pi), ("wm", learner.wm), ("critic", learner.critic),
                             ("opt_pi", learner.opt_pi), ("opt_wm", learner.opt_wm), ("opt_v", learner.opt_v)):
            equal_tree(module.state_dict(), original[name])
        status["restored_models_and_optimizers_exact"] = True
        with torch.no_grad():
            learner.collect(12)
            info = learner._state_info()
            goals = learner.mdp.goal(info["f"], 10)
            action = learner.pi(learner.obs_hist, goals)
            padded = fixed_blocks(learner.pi, learner.obs_hist, goals)
            torch.testing.assert_close(action, padded, atol=0 if a.envs == 4096 else .02, rtol=.01)
            if not bool(torch.isfinite(action).all()):
                raise AssertionError("Nonfinite restored policy action")
        before = {name: {k:v.clone() for k,v in module.state_dict().items()}
                  for name, module in (("wm", learner.wm), ("pi", learner.pi), ("critic", learner.critic))}
        from evaluation import FIXED_DEMOS
        source = json.loads(a.library.read_text())
        rows = [dict(r, split="train") for r in source["splits"]["train"] if r["name"] in FIXED_DEMOS]
        if len(rows) != 3:
            raise ValueError("Fixed ordinary walking and raising identities missing")
        library = a.output / "fixed_library.json"
        save(library, {"splits": {"train": rows}})
        with preserve_hp(learner):
            with preserve_random_state():
                _, independent, _ = prepare(SimpleNamespace(runtime=a.runtime, library=library, output=a.output),
                                            num_envs=3, eval_mode=True)
                def tick(frame):
                    save(a.output / "status.json", {**status, "phase": "fixed_physical_evaluation", "frame": frame})
                trace = enrich(rollout(independent,
                    lambda history, goal: fixed_blocks(learner.pi, history, goal).float(), tick), independent.supermimic_mdp)
                measured = gait_summary(trace, rows)
                save(a.output / "fixed_gait_metrics.json", measured)
                np.savez_compressed(a.output / "fixed_trajectory.npz", **trace)
        for name, module in (("wm", learner.wm), ("pi", learner.pi), ("critic", learner.critic)):
            equal_tree(module.state_dict(), before[name])
        if learner.wm_updates != 32710 or learner.actor_updates != 2771:
            raise AssertionError("Preflight performed an optimizer update")
        status.update(status="ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED", completed_utc=stamp(),
                      evaluation_preserved_live_hp_state_and_rng=True, optimizer_steps=0,
                      fixed_forward_block=4096, checkpoint=identity(a.checkpoint),
                      actual_contact_and_foot_metrics=True, fresh_replay_steps=12,
                      replay_and_simulator_not_resumed=True)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for physical in (independent, env):
            if physical is not None:
                physical.close()
                physical.backend.close()
        save(a.output / "status.json", status)
        save(a.output / "receipt.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    run(p.parse_args())
