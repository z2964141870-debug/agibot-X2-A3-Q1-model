"""Exercise native contact observation and saved-trajectory metrics with no learning."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from worker import prepare
from resources import save, identity, stamp
from evaluation import FIXED_DEMOS, rollout, summarize, preserve_random_state, preserve_learner
from adapter import HPActor, PhysicalAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    library = json.loads(args.library.read_text())
    rows = [r for r in library["splits"]["train"] if r["name"] in FIXED_DEMOS]
    rows += library["splits"]["validation"][:1]
    if len(rows) != 4:
        raise ValueError("Expected fixed three demos and one validation")
    combined = args.output / "evaluation_library.json"
    save(combined, {"splits": {"train": rows}})
    env = live = None
    try:
        torch.set_num_threads(8)
        _, training_env, _ = prepare(args, num_envs=16)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        payload = torch.load(args.checkpoint, map_location=training_env.device, weights_only=True)
        actor = HPActor(load_policy(training_env, payload)).to(training_env.device)
        live = PhysicalAdapter(training_env, actor)
        with torch.no_grad():
            for _ in range(24):
                live.step(actor(live.observations))
        with preserve_learner(live):
            _, env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=combined, output=args.output), num_envs=4, eval_mode=True)
        policy = load_policy(env, payload)
        rng = torch.get_rng_state().clone()
        with preserve_learner(live):
            trace = rollout(env, policy)
        if not torch.equal(rng, torch.get_rng_state()):
            raise AssertionError("Evaluation RNG restoration failed")
        for index in range(len(rows)):
            count = int(trace["steps"][index])
            start = int(env.supermimic_mdp.motion.clip_start[index])
            np.testing.assert_array_equal(trace["frames"][:count, index], np.arange(start + 1, start + count + 1))
        if not np.isfinite(trace["force"]).all() or not np.isfinite(trace["slip"]).all() or not (trace["force"] > 5).any():
            raise AssertionError("Native solved contact contract failed")
        result = summarize(trace, rows)
        np.savez_compressed(args.output / "native_protocol_trace.npz", **trace)
        result.update(status="NATIVE_CONTACT_METRICS_AND_RNG_PROTOCOL_PASSED", completed_utc=stamp(),
                      source_checkpoint=identity(args.checkpoint), new_optimizer_updates=0,
                      native_observer_state_invariance=["qpos", "qvel", "qacc_warmstart"],
                      contiguous_f_plus_one_frames=True, RNG_restored=True,
                      live_learner_envs=16, live_learner_transitions=384,
                      evaluation_preserves_live_backend_history_accumulators_and_cuda_rng=True)
        save(args.output / "receipt.json", result)
        print(json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        if live is not None:
            live.close()


if __name__ == "__main__":
    main()
