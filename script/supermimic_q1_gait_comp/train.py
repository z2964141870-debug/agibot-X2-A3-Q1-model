"""One new lower-body gait PPO with first-hour physical extension criteria."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "project_tools")]
from worker import prepare, process_identity
from gait_adapter import Adapter, Actor, make_ppo, bootstrap_rewards
from physical import rollout, preserve_balance, preserve_random_state, summarize, comparison, gait_summary, FootReference
from decisions import gate
from timing import ConditionalClock
from overnight import (OPTIMIZATION_DEADLINE, DELIVERY_DEADLINE, CHECKS, continuing_evidence)
from resources import save, identity, stamp
from diagnose_q1_upper_reference import policy_digest
from metrics import checkpoint_rank


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "process": process_identity(), "started_utc": stamp(),
              "code_commit": a.code_commit, "iteration": 0, "new_world_model_updates": 0,
              "training_seconds": 0., "algorithm": "Installed RSL-RL PPO, lower latent14, immutable selected HP base"}
    save(a.output / "status.json", status)
    clock = ConditionalClock(a.output / "training_clock.json", torch.cuda.synchronize)
    adapter = env = eval_env = None
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stages, final_paths = [], []
    iteration = 0
    outcome = "overnight_budget_completed"
    try:
        torch.set_num_threads(8)
        cfg, env, _ = prepare(a, num_envs=4096)
        from omegaconf import OmegaConf
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        OmegaConf.save(cfg, a.output / "config.yaml", resolve=True)
        teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
        teacher_digest = policy_digest(teacher)
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        if time.time() >= OPTIMIZATION_DEADLINE-300:
            raise RuntimeError("Too late for a new overnight training run")
        adapter = Adapter(env, teacher, thresholds, a.upper_alpha)
        actor = Actor().to(env.device)
        algorithm = make_ppo(actor, adapter.observations, 4096)
        rows = [dict(row, split=split) for split in ("train", "validation")
                for row in json.loads(a.library.read_text())["splits"][split]]
        if len(rows) != 330 or sum(r["split"] == "train" for r in rows) != 300:
            raise ValueError("Keep original AP02 300/30")
        evaluation = a.output.parent / "evaluation"
        evaluation.mkdir(exist_ok=False)
        combined = evaluation / "evaluation_library.json"
        save(combined, {"splits": {"train": rows}, "source": identity(a.library)})
        with preserve_random_state():
            _, eval_env, _ = prepare(SimpleNamespace(runtime=a.runtime, library=combined, output=evaluation),
                                      num_envs=330, eval_mode=True)
            eval_foot = FootReference(eval_env.supermimic_mdp)

        def progress(phase, **more):
            status.update(status="RUNNING", phase=phase, updated_utc=stamp(), iteration=iteration,
                training_seconds=clock.total, sim_steps=iteration*98304, optimizer_steps=iteration*20,
                actor_optimizer_steps=max(iteration-10, 0)*20, current_cap_seconds=clock.cap, **more)
            save(a.output / "status.json", status)

        def checkpoint(name):
            numpy_rng = np.random.get_state()
            payload = {"format": "Q1_RF15_GAIT_HEAD_V1", "actor": actor.state_dict(),
                "critic": algorithm.critic.state_dict(), "optimizer": algorithm.optimizer.state_dict(),
                "source_base": identity(a.checkpoint), "source_teacher_digest": teacher_digest,
                "calibration": identity(a.calibration), "iteration": iteration, "sim_steps": iteration*98304,
                "training_seconds": clock.total, "current_cap_seconds": clock.cap,
                "optimizer_steps": iteration*20, "actor_optimizer_steps": max(iteration-10, 0)*20,
                "new_world_model_updates": 0, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                "code_commit": a.code_commit, "latent_std": .1, "residual_radius_rad": .15,
                "input_features": 1420, "maximum_training_seconds": 14400,
                "upper_reference_alpha": a.upper_alpha, "learning_rate": algorithm.learning_rate,
                "desired_kl": .01, "adaptive_learning_rate_bounds": [1e-5, 5e-5],
                "optimization_wall_deadline_unix": OPTIMIZATION_DEADLINE,
                "delivery_wall_deadline_unix": DELIVERY_DEADLINE,
                "user_accepted_checkpoint": None,
                "note": "Requires identified immutable selected HP base and gait adapter. Not a full simulator-resume checkpoint; no automatic resume."}
            path = a.output / name
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            return path

        def evaluate(label, mode, initial=None, baseline=None):
            paths, repeats = [], []
            with preserve_balance(adapter):
                for repeat in range(3):
                    def tick(frame):
                        progress("evaluation", evaluation_label=label, evaluation_repeat=repeat, evaluation_frame=frame)
                    tick(0)
                    trace = rollout(eval_env, teacher, actor.head, thresholds, eval_foot, mode, tick,
                                    upper_alpha=a.upper_alpha)
                    path = evaluation / f"{label}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, training_seconds=clock.total, mode=mode)
                    for name, previous_paths in (("initial_static075", initial), ("selected_base", baseline)):
                        if previous_paths:
                            with np.load(previous_paths[repeat], allow_pickle=False) as saved:
                                summary[name] = comparison(trace, {k:saved[k] for k in saved.files}, rows)
                    if initial:
                        repeats.append(summary["initial_static075"])
                    save(path.with_suffix(".json"), summary)
                    paths.append(path)
            if policy_digest(teacher) != teacher_digest:
                raise AssertionError("Evaluation changed frozen base or normalization")
            return paths, repeats

        base_paths, _ = evaluate("selected_base", "teacher")
        initial_paths, _ = evaluate("initial_static075", "static075", baseline=base_paths)
        initial_checkpoint = checkpoint("initial.pt")
        candidates = [{"label": "initial_static", "checkpoint": identity(initial_checkpoint),
                       "traces": [str(p) for p in initial_paths],
                       "rank": checkpoint_rank([json.loads(p.with_suffix(".json").read_text()) for p in initial_paths])}]
        initial_actor = copy.deepcopy(actor.state_dict())
        actor.head.requires_grad_(False)
        checks = iter(CHECKS)
        next_check = next(checks)
        last_saved = 0.
        while not requested:
            if time.time() >= OPTIMIZATION_DEADLINE-120:
                outcome = "overnight_optimization_wall_deadline"
                break
            if clock.total >= clock.cap-100.:
                outcome = "overnight_soft_training_budget"
                break
            progress("collecting", next_evaluation_seconds=next_check)
            clock.begin()
            try:
                if iteration == 10:
                    for key,value in actor.state_dict().items():
                        torch.testing.assert_close(value, initial_actor[key], atol=0, rtol=0)
                    actor.head.requires_grad_(True)
                    del initial_actor
                report = {}
                with torch.no_grad():
                    for _ in range(24):
                        latent = algorithm.act(adapter.observations)
                        obs, reward, done, extras = adapter.step(latent)
                        algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
                        for key,value in extras["report"].items():
                            report[key] = report.get(key, 0) + value
                    algorithm.compute_returns(adapter.observations)
                progress("updating")
                sample = algorithm.storage.observations[0].clone()
                with torch.no_grad():
                    previous_mean = actor(sample).clone()
                losses = algorithm.update()
                if not all(np.isfinite(v) for v in losses.values()):
                    raise FloatingPointError("Nonfinite PPO loss")
                with torch.no_grad():
                    current_mean = actor(sample)
                    exact_kl = .5*((current_mean-previous_mean)/.1).square().sum(-1)
                    report.update(policy_update_kl_mean=float(exact_kl.mean()),
                                  policy_update_kl_p99=float(torch.quantile(exact_kl, .99)),
                                  latent_mean_rms=float(current_mean.square().mean().sqrt()),
                                  learning_rate=algorithm.learning_rate)
                iteration += 1
            finally:
                clock.end()
            row = {"step": iteration, "training_seconds": clock.total, "sim_steps": iteration*98304,
                   "optimizer_steps": iteration*20, "actor_optimizer_steps": max(iteration-10, 0)*20,
                   "wm_err8": None, "imag_survival_H": None, **report,
                   **{"ppo_"+key:value for key,value in losses.items()}}
            with (a.output / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            progress("complete_update")
            if clock.total-last_saved >= 300:
                checkpoint("last.pt")
                last_saved = clock.total
            if clock.total < next_check:
                continue
            number = len(stages)+1
            path = checkpoint(f"stage_{number:02d}.pt")
            final_paths, comparisons = evaluate(f"stage_{number:02d}", "trained075", initial_paths, base_paths)
            decision = gate(comparisons)
            continuation = continuing_evidence(comparisons)
            stages.append({"iteration": iteration, "training_seconds": clock.total, "checkpoint": identity(path),
                           "comparisons": comparisons, "gate": decision, "overnight_continuation": continuation,
                           "traces": [identity(p) for p in final_paths]})
            save(a.output / "stage_decisions.json", stages)
            candidates.append({"label": f"stage_{number:02d}", "checkpoint": identity(path),
                "traces": [str(p) for p in final_paths],
                "rank": checkpoint_rank([json.loads(p.with_suffix(".json").read_text()) for p in final_paths])})
            print(json.dumps({"stage": number, "training_seconds": clock.total, "gate": decision}), flush=True)
            if decision["physical_regression"]:
                outcome = "physical_regression"
                break
            if clock.total >= 3500 and not continuation["continue_authorized"]:
                outcome = "no_confirmed_physical_improvement_at_hour_check"
                break
            if clock.total >= 14300:
                outcome = "overnight_four_hour_budget_completed"
                break
            next_check = next(checks)
        if requested:
            raise InterruptedError("External stop requested; no automatic restart")
        if not stages or stages[-1]["iteration"] != iteration:
            number = len(stages)+1
            path = checkpoint(f"stage_{number:02d}.pt")
            final_paths, comparisons = evaluate(f"stage_{number:02d}", "trained075", initial_paths, base_paths)
            stages.append({"iteration": iteration, "training_seconds": clock.total, "checkpoint": identity(path),
                           "comparisons": comparisons, "gate": gate(comparisons),
                           "overnight_continuation": continuing_evidence(comparisons),
                           "traces": [identity(p) for p in final_paths]})
            save(a.output / "stage_decisions.json", stages)
            candidates.append({"label": f"stage_{number:02d}", "checkpoint": identity(path),
                "traces": [str(p) for p in final_paths],
                "rank": checkpoint_rank([json.loads(p.with_suffix(".json").read_text()) for p in final_paths])})
        if clock.total > clock.cap or clock.total > 14400:
            raise RuntimeError("Conditional training budget exceeded")
        final = checkpoint("final.pt")
        recommendation = min(candidates, key=lambda row: row["rank"])
        save(a.output / "checkpoint_recommendation.json", {"candidates": candidates,
             "recommended": recommendation, "user_accepted_checkpoint": None,
             "note": "Gait-based numerical recommendation only; compare upper amplitude/timing and simulation videos before adoption"})
        if policy_digest(teacher) != teacher_digest:
            raise AssertionError("Frozen base changed during PPO")
        save(evaluation / "receipt.json", {"rows": rows, "initial_traces": [str(p) for p in initial_paths],
             "base_traces": [str(p) for p in base_paths], "final_traces": [str(p) for p in final_paths],
             "training_seconds": clock.total, "fixed_forward_block": 4096,
             "upper_reference_alpha": a.upper_alpha, "user_accepted_checkpoint": None,
             "recommended_traces": recommendation["traces"], "recommended_label": recommendation["label"]})
        status.update(status="TRAINING_COMPLETED", outcome=outcome, training_seconds=clock.total,
                      iteration=iteration, sim_steps=iteration*98304, optimizer_steps=iteration*20,
                      actor_optimizer_steps=max(iteration-10, 0)*20, final=identity(final))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        clock.end()
        if adapter is not None:
            adapter.close()
        elif env is not None:
            env.close()
            env.backend.close()
        if eval_env is not None:
            eval_env.close()
            eval_env.backend.close()
        status["ended_utc"] = stamp()
        save(a.output / "status.json", status)
        save(a.output / "receipt.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--upper-alpha", type=float, default=.75)
    run(parser.parse_args())
