"""One RF12 PPO learner with in-process evaluation and cumulative budgets."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from worker import prepare, process_identity
from adapter import HPActor, PhysicalAdapter, bootstrap_rewards, make_ppo
from evaluation import rollout, summarize, compare, preserve_random_state, preserve_learner
from gates import CHECKPOINTS, MAX_TOTAL_SECONDS, classify_repeats, next_phase
from resources import identity, save, stamp


class TrainingClock:
    def __init__(self, path):
        self.path = path
        self.total = 0.
        self.active = None
        self.phase = "A"
        self.cap = 3600.
        self.flush()

    def flush(self):
        save(self.path, {"status": "ACTIVE" if self.active is not None else "PAUSED_FOR_NONTRAINING",
                         "updated_utc": stamp(), "process": process_identity(),
                         "accumulated_training_seconds": self.total, "active_started_monotonic": self.active,
                         "phase": self.phase, "current_cap_seconds": self.cap,
                         "absolute_maximum_seconds": MAX_TOTAL_SECONDS,
                         "definition": "Synchronized physical collection, GAE and optimizer/diagnostic time; excludes initialization, evaluation, checkpoint I/O and media."})

    def begin(self):
        if self.active is not None:
            raise RuntimeError("Training clock already active")
        torch.cuda.synchronize()
        self.active = time.monotonic()
        self.flush()

    def end(self):
        if self.active is not None:
            torch.cuda.synchronize()
            self.total += time.monotonic() - self.active
            self.active = None
            self.flush()


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(),
              "code_commit": args.code_commit, "algorithm": "installed RSL-RL PPO", "iteration": 0,
              "training_seconds": 0., "optimizer_steps": 0, "actor_optimizer_steps": 0}
    save(args.output / "status.json", status)
    clock = TrainingClock(args.output / "training_clock.json")
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    adapter = eval_env = None
    iteration = 0
    stage_b_started = stage_c_started = extension_deadline = None
    severe_streak = 0
    checkpoints = []
    baseline_paths = []
    final_paths = []
    outcome = None
    try:
        torch.set_num_threads(8)
        cfg, physical, _ = prepare(args, num_envs=4096)
        from omegaconf import OmegaConf
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        from unisupermimic.rl.algos.supermimic import networks
        OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
        inherited = torch.load(args.checkpoint, map_location=physical.device, weights_only=True)
        actor = HPActor(load_policy(physical, inherited)).to(physical.device)
        adapter = PhysicalAdapter(physical, actor)
        algorithm = make_ppo(actor, adapter.observations, 4096)
        original_normalizers = {key: value.clone() for key, value in actor.policy.state_dict().items() if "_norm." in key}
        eval_library = json.loads(args.evaluation_library.read_text())
        rows = eval_library["splits"]["train"] + eval_library["splits"]["validation"]
        evaluation = args.output.parent / "evaluation"
        evaluation.mkdir(exist_ok=False)
        combined = evaluation / "evaluation_library.json"
        save(combined, {"splits": {"train": rows}, "original_library": identity(args.evaluation_library)})
        with preserve_random_state():
            _, eval_env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=combined, output=evaluation),
                                     num_envs=len(rows), eval_mode=True)
        baseline_policy = load_policy(eval_env, inherited)

        def progress(phase, **values):
            status.update(phase=phase, updated_utc=stamp(), iteration=iteration,
                          training_seconds=clock.total, recipe_phase=clock.phase, **values)
            save(args.output / "status.json", status)

        def evaluate(policy, label, baseline=False):
            paths, comparisons = [], []
            with preserve_learner(adapter):
                for repeat in range(3):
                    progress("evaluation", evaluation_label=label, evaluation_repeat=repeat)
                    trace = rollout(eval_env, policy,
                                    progress=lambda frame: progress("evaluation", evaluation_label=label,
                                                                    evaluation_repeat=repeat, evaluation_frame=frame))
                    path = evaluation / f"{label}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    if not baseline:
                        with np.load(baseline_paths[repeat], allow_pickle=False) as stored:
                            previous = {key: stored[key] for key in stored.files}
                        comparison = compare(trace, previous, rows)
                        comparisons.append(comparison)
                        summary["RF11_common_prefix_comparison"] = comparison
                        del previous
                    summary.update(training_seconds=clock.total, recipe_phase=clock.phase,
                                   repeat=repeat, source_checkpoint=identity(args.checkpoint) if baseline else None)
                    save(path.with_suffix(".json"), summary)
                    paths.append(path)
                    del trace
            return paths, comparisons

        def checkpoint(name):
            numpy_rng = np.random.get_state()
            payload = {"pi": actor.policy.state_dict(), "args": inherited["args"],
                       "policy_input_spec": inherited["policy_input_spec"],
                       "critic": algorithm.critic.state_dict(), "optimizer": algorithm.optimizer.state_dict(),
                       "distribution": actor.distribution.state_dict(), "iteration": iteration,
                       "sim_steps": iteration * 24 * 4096, "wall_time": clock.total,
                       "training_seconds": clock.total, "recipe_phase": clock.phase,
                       "reward_alpha": adapter.alpha, "new_motion_fraction": adapter.new_fraction,
                       "stage_b_started": stage_b_started, "stage_c_started": stage_c_started,
                       "extension_deadline": extension_deadline, "source_RF11": identity(args.checkpoint),
                       "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                       "python_rng": random.getstate(),
                       "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                       "normalization": "copied RF11, frozen", "fixed_policy_forward_block": 4096,
                       "optimizer_steps": iteration * 20, "actor_optimizer_steps": max(iteration - 10, 0) * 20,
                       "code_commit": args.code_commit, "saved_at": stamp(), "algorithm": "RSL-RL PPO"}
            path = args.output / name
            atomic_checkpoint(path, payload)
            return path

        baseline_paths, _ = evaluate(baseline_policy, "RF11_baseline", baseline=True)
        del baseline_policy
        initial = checkpoint("initial.pt")
        save(evaluation / "initial_sources.json", {"traces": [identity(p) for p in baseline_paths],
                                                    "checkpoint": identity(initial), "repeats": 3})
        # First ten complete rollout/update iterations warm only the new critic.
        for parameter in actor.policy.parameters():
            parameter.requires_grad_(False)
        initial_actor = copy.deepcopy(actor.policy.state_dict())
        next_evaluation = CHECKPOINTS[0]
        last_checkpoint_seconds = 0.
        torch.cuda.reset_peak_memory_stats()
        while not requested:
            if clock.total >= min(clock.cap - 55, MAX_TOTAL_SECONDS - 55):
                raise RuntimeError("Independent cumulative budget reserve reached before a planned checkpoint")
            if stage_b_started is not None:
                adapter.alpha = min(1., max(0., (clock.total - stage_b_started) / 600.))
            if stage_c_started is not None:
                adapter.new_fraction = .5 * min(1., max(0., (clock.total - stage_c_started) / 3600.))
            progress("collecting", status="RUNNING", next_evaluation_training_seconds=next_evaluation)
            clock.begin()
            report = {}
            try:
                if iteration == 10:
                    for key, value in actor.policy.state_dict().items():
                        torch.testing.assert_close(value, initial_actor[key], rtol=0, atol=0)
                    for parameter in actor.policy.parameters():
                        parameter.requires_grad_(True)
                    del initial_actor
                with torch.no_grad():
                    for _ in range(24):
                        action = algorithm.act(adapter.observations)
                        obs, reward, done, extras = adapter.step(action)
                        algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
                        for key, value in extras["report"].items():
                            if key not in ("reward_alpha", "new_motion_fraction"):
                                report[key] = report.get(key, 0) + value
                    algorithm.compute_returns(adapter.observations)
                progress("updating")
                losses = algorithm.update()
                if not all(np.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite stock PPO losses")
                iteration += 1
            finally:
                clock.end()
            for key, value in original_normalizers.items():
                torch.testing.assert_close(actor.policy.state_dict()[key], value, rtol=0, atol=0)
            row = {"step": iteration, "training_seconds": clock.total, "recipe_phase": clock.phase,
                   "sim_steps": iteration * 24 * 4096, "optimizer_steps": iteration * 20,
                   "actor_optimizer_steps": max(iteration - 10, 0) * 20,
                   "reward_alpha": adapter.alpha, "new_motion_fraction": adapter.new_fraction,
                   "wm_err8": None, "imag_survival_H": None, **report, **{"ppo_" + k: v for k, v in losses.items()}}
            with (args.output / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            progress("complete_update", sim_steps=row["sim_steps"], optimizer_steps=row["optimizer_steps"],
                     actor_optimizer_steps=row["actor_optimizer_steps"], reward_alpha=adapter.alpha,
                     new_motion_fraction=adapter.new_fraction)
            if clock.total - last_checkpoint_seconds >= 120:
                checkpoint("last.pt")
                last_checkpoint_seconds = clock.total
            if clock.total < next_evaluation:
                continue
            path = checkpoint(f"stage_{len(checkpoints) + 1:02d}.pt")
            paths, comparisons = evaluate(actor.policy, f"stage_{len(checkpoints) + 1:02d}")
            final_paths = paths
            gate = classify_repeats(comparisons)
            decision = next_phase(clock.phase, clock.total, gate, extension_deadline, severe_streak)
            entry = {"training_seconds": clock.total, "iteration": iteration, "phase": clock.phase,
                     "checkpoint": identity(path), "comparisons": comparisons, "gate": gate, "decision": decision,
                     "traces": [identity(p) for p in paths]}
            checkpoints.append(entry)
            save(args.output / "stage_decisions.json", checkpoints)
            print(json.dumps({"training_seconds": clock.total, "iteration": iteration,
                              "phase": clock.phase, "gate": gate, "decision": decision}), flush=True)
            progress("stage_delivery", stage=len(checkpoints))
            command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"), "-u",
                       str(HERE / "deliver.py"), "--runtime", str(args.runtime),
                       "--trial", str(args.output.parent), "--video-suite", str(args.video_suite),
                       "--stage", str(len(checkpoints))]
            try:
                with (args.output / f"stage_{len(checkpoints):02d}_delivery.log").open("x") as stream:
                    delivered = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=840)
                save(args.output / f"stage_{len(checkpoints):02d}_delivery.json",
                     {"returncode": delivered.returncode, "updated_utc": stamp(), "new_optimizer_steps": 0})
            except subprocess.TimeoutExpired:
                save(args.output / f"stage_{len(checkpoints):02d}_delivery.json",
                     {"error": "Nontraining stage delivery timed out; saved physical evidence retained",
                      "updated_utc": stamp(), "new_optimizer_steps": 0})
            if decision["action"] == "stop":
                outcome = decision["reason"]
                break
            if decision["phase"] == "B":
                if stage_b_started is None:
                    stage_b_started = clock.total
                next_evaluation = next(point for point in CHECKPOINTS if point > clock.total)
                clock.phase, clock.cap = "B", next_evaluation + 100
            else:
                if stage_c_started is None:
                    stage_c_started = clock.total
                    extension_deadline = decision["deadline"]
                severe_streak = decision["severe_streak"]
                clock.phase, clock.cap = "C", extension_deadline
                next_evaluation = min(clock.total + 3600., extension_deadline - 60.)
            clock.flush()
        if requested:
            raise InterruptedError("External stop requested; no automatic restart")
        final = checkpoint("final.pt")
        status.update(status="TRAINING_COMPLETED", stop_reason=outcome, training_seconds=clock.total,
                      iteration=iteration, sim_steps=iteration * 24 * 4096,
                      optimizer_steps=iteration * 20, actor_optimizer_steps=max(iteration - 10, 0) * 20,
                      completed_utc=stamp(), checkpoint=identity(final),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      final_traces=[identity(p) for p in final_paths], baseline_traces=[identity(p) for p in baseline_paths],
                      physical_gate_met=any(r["gate"]["extension_qualified"] for r in checkpoints),
                      source_RF11=identity(args.checkpoint), no_automatic_retry=True)
        save(args.output / "receipt.json", status)
        save(evaluation / "receipt.json", {"status": "PHYSICAL_BASELINE_AND_STAGE_REPEATS_COMPLETED",
                                            "baseline_traces": [str(p) for p in baseline_paths],
                                            "final_traces": [str(p) for p in final_paths], "rows": rows,
                                            "protocol": "Same 330 references, finite clip-end task, deterministic mean policy and no observation noise; three repeats with native contact observation."})
    except BaseException as error:
        clock.end()
        status.update(status="STOPPED_SIGNAL" if isinstance(error, InterruptedError) else "FAILED",
                      training_seconds=clock.total, iteration=iteration, error=f"{type(error).__name__}: {error}",
                      ended_utc=stamp(), no_automatic_retry=True)
        raise
    finally:
        save(args.output / "status.json", status)
        if eval_env is not None:
            eval_env.close()
            eval_env.backend.close()
        if adapter is not None:
            adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "evaluation-library", "video-suite", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
