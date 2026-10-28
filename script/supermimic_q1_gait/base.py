"""Continue the RF11 full HP checkpoint within a cumulative four-hour cap."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"),
                str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "supermimic_q1_balance")]
from worker import prepare, check_contract, learner_class, process_identity
from resources import save, stamp, identity
from evaluation import rollout, summarize, compare, preserve_random_state
from control import fixed_blocks
from clock import TrainingClock
from metrics import enrich, gait_summary, checkpoint_rank


def fingerprint(learner):
    import warp as wp
    digest = hashlib.sha256()
    values = list(learner._state_info().values()) + [learner.obs_hist]
    for owner in (learner.env._supermimic_command, learner.buffer):
        for key, value in sorted(vars(owner).items()):
            if isinstance(value, torch.Tensor) and owner is not learner.buffer:
                values.append(value)
            elif isinstance(value, (int, float, bool)):
                digest.update(str((key, value)).encode())
    for key in ("qpos", "qvel", "qacc_warmstart", "ctrl", "time"):
        if hasattr(learner.env.backend._device_data, key):
            values.append(wp.to_torch(getattr(learner.env.backend._device_data, key)))
    for module in (learner.pi, learner.wm, learner.critic):
        values.extend(module.state_dict().values())
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def preserve_hp(learner):
    before = fingerprint(learner)
    with preserve_random_state():
        yield
    if before != fingerprint(learner):
        raise AssertionError("Evaluation changed active HP models, environment, history or replay counters")


def make_learner(env, cfg, output, checkpoint):
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
    Parent, _ = learner_class()

    class Continued(Parent):
        def collect(self, steps):
            original = self.env.step
            def observed(action):
                state = original(action)
                tr = state.info[SUPERMIMIC_TRANSITION_KEY]
                r = self.reporting
                r["transitions"] += self.env.num_envs
                r["reward_sum"] += float(tr["r"].sum())
                count = int(tr["done"].sum())
                r["completed_episodes"] += count
                r["hard_terminations"] += int(tr["term"].sum())
                if count:
                    if tr["ep_len"].numel() != count:
                        raise ValueError("Invalid completed episode reporting")
                    r["normalized_return_sum"] += float(tr["ep_ret_norm"].sum())
                    r["length_steps_sum"] += float(tr["ep_len"].sum())
                return state
            self.env.step = observed
            try:
                return SuperMimicLearner.collect(self, steps)
            finally:
                self.env.step = original

    learner = Continued(env, cfg, output, device="cuda:0", backend="mjwarp")
    learner.load(checkpoint)
    learner.wm_updates = learner.iteration * cfg.wm_batches
    learner.actor_updates = max(learner.iteration - cfg.wm_only_iters, 0)
    return learner


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "process": process_identity(), "started_utc": stamp(),
              "code_commit": a.code_commit, "optimizer_continuation": True,
              "simulator_replay_rng_continuation": False}
    save(a.output / "status.json", status)
    env = eval_env = None
    clock = None
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stages = []
    outcome = "cumulative_four_hour_budget_completed"
    try:
        torch.set_num_threads(8)
        cfg, env, learner_cfg = prepare(a, num_envs=4096)
        save(a.output / "contract.json", check_contract(env))
        from omegaconf import OmegaConf
        from unisupermimic.rl.algos.supermimic.learner import _empty_stats
        OmegaConf.save(cfg, a.output / "config.yaml", resolve=True)
        learner = make_learner(env, learner_cfg, a.output, a.checkpoint)
        inherited = float(learner.wall_time)
        if abs(inherited - 3500.2659895760007) > .001 or learner.iteration != 3271:
            raise ValueError("RF14 must continue the identified RF11 final checkpoint")
        clock = TrainingClock(a.output / "training_clock.json", inherited, 14400., torch.cuda.synchronize)
        rows = [dict(row, split=split) for split in ("train", "validation")
                for row in json.loads(a.library.read_text())["splits"][split]]
        if len(rows) != 330 or sum(r["split"] == "train" for r in rows) != 300:
            raise ValueError("Original AP02 300/30 only")
        evaluation = a.output.parent / "evaluation"
        evaluation.mkdir(exist_ok=False)
        combined = evaluation / "evaluation_library.json"
        save(combined, {"splits": {"train": rows}, "source": identity(a.library)})
        with preserve_random_state():
            _, eval_env, _ = prepare(SimpleNamespace(runtime=a.runtime, library=combined, output=evaluation),
                                      num_envs=330, eval_mode=True)

        def progress(phase, **more):
            status.update(status="RUNNING", phase=phase, updated_utc=stamp(), iteration=learner.iteration,
                          training_seconds=clock.total, new_training_seconds=clock.total-inherited,
                          sim_steps=learner.sim_steps, wm_updates=learner.wm_updates,
                          actor_updates=learner.actor_updates, **more)
            save(a.output / "status.json", status)

        def checkpoint(name):
            learner.wall_time = clock.total
            path = learner.save(a.output / name)
            return path

        def evaluate(label, baseline=None):
            paths, summaries = [], []
            with preserve_hp(learner):
                for repeat in range(3):
                    def tick(frame):
                        progress("evaluation", evaluation_label=label, evaluation_repeat=repeat, evaluation_frame=frame)
                    tick(0)
                    policy = lambda history, goal: fixed_blocks(learner.pi, history, goal).float()
                    trace = enrich(rollout(eval_env, policy, tick), eval_env.supermimic_mdp)
                    path = evaluation / f"{label}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, training_seconds=clock.total)
                    if baseline:
                        with np.load(baseline[repeat], allow_pickle=False) as saved:
                            summary["RF11_common_prefix"] = compare(trace, {k:saved[k] for k in saved.files}, rows)
                    save(path.with_suffix(".json"), summary)
                    paths.append(path)
                    summaries.append(summary)
            return paths, summaries

        initial = checkpoint("initial.pt")
        baseline, summaries = evaluate("RF11_initial")
        candidates = [{"label": "RF11_initial", "checkpoint": identity(initial),
                       "training_seconds": clock.total, "rank": checkpoint_rank(summaries),
                       "traces": [str(p) for p in baseline]}]
        save(a.output / "checkpoint_selection.json", {"candidates": candidates, "selected": candidates[0]})
        progress("fresh_replay_collection")
        clock.begin()
        try:
            learner.collect(max(learner_cfg.warmup_steps, learner_cfg.hist, learner_cfg.wm_window))
        finally:
            clock.end()
        learner.reporting = {k: 0. if isinstance(v, float) else 0 for k,v in learner.reporting.items()}
        learner.stats = _empty_stats()
        checks = iter((3600., 7200., 14300.))
        next_check = next(checks)
        accumulator = 0.
        last_save = clock.total
        final_paths = baseline
        while not requested and clock.total < 14300:
            if clock.total >= 14340 - 40:
                raise RuntimeError("Training exit reserve exhausted")
            progress("collecting", next_evaluation_seconds=next_check)
            clock.begin()
            try:
                learning_rate = learner.lr_at(learner.iteration)
                for optimizer in (learner.opt_wm, learner.opt_pi, learner.opt_v):
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                accumulator += 1. / learner_cfg.utd
                steps = int(accumulator)
                accumulator -= steps
                learner.collect(steps)
                progress("updating")
                for _ in range(learner_cfg.wm_batches):
                    learner.wm_update()
                if learner.iteration >= learner_cfg.wm_only_iters:
                    learner.policy_update()
                learner.iteration += 1
                r = dict(learner.reporting)
                row = {"step": learner.iteration, "sim_steps": learner.sim_steps,
                       "wm_updates": learner.wm_updates, "actor_updates": learner.actor_updates,
                       "wm_err8": learner.latest_wm[1],
                       "imag_survival_H": learner.latest_policy[4] if learner.latest_policy else None,
                       "reward_mean": r["reward_sum"] / max(r["transitions"], 1), "lr": learning_rate, **r}
                learner.reporting = {k: 0. if isinstance(v, float) else 0 for k,v in r.items()}
                learner.stats = _empty_stats()
            finally:
                clock.end()
            row.update(training_seconds=clock.total, new_training_seconds=clock.total-inherited)
            with (a.output / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            progress("complete_update")
            if clock.total - last_save >= 300:
                checkpoint("last.pt")
                last_save = clock.total
            if clock.total < next_check:
                continue
            number = len(stages) + 1
            path = checkpoint(f"stage_{number:02d}.pt")
            final_paths, summaries = evaluate(f"stage_{number:02d}", baseline)
            rank = checkpoint_rank(summaries)
            candidate = {"label": f"stage_{number:02d}", "checkpoint": identity(path),
                         "training_seconds": clock.total, "rank": rank, "traces": [str(p) for p in final_paths]}
            candidates.append(candidate)
            selected = min(candidates, key=lambda item: item["rank"])
            save(a.output / "checkpoint_selection.json", {"candidates": candidates, "selected": selected,
                 "root_exact_tracking_is_diagnostic": True, "visual_acceptance_pending": True})
            failures = [s["summary"]["validation"]["failures"] for s in summaries]
            severe = sum(f >= 6 for f in failures) >= 2
            stages.append({**candidate, "validation_failures": failures, "severe_regression": severe,
                           "automatic_new_training": False})
            save(a.output / "stage_decisions.json", stages)
            print(json.dumps(stages[-1]), flush=True)
            if severe:
                outcome = "physical_regression"
                break
            if clock.total >= 14300:
                break
            next_check = next(checks)
        if requested:
            raise InterruptedError("Stop requested; no automatic restart")
        final = checkpoint("final.pt")
        if clock.total > 14400:
            raise RuntimeError("Cumulative four-hour cap exceeded")
        save(evaluation / "receipt.json", {"rows": rows, "initial_traces": [str(p) for p in baseline],
             "final_traces": [str(p) for p in final_paths], "training_seconds": clock.total,
             "new_training_seconds": clock.total-inherited, "fixed_forward_block": 4096})
        status.update(status="TRAINING_COMPLETED", outcome=outcome, training_seconds=clock.total,
                      new_training_seconds=clock.total-inherited, iteration=learner.iteration,
                      sim_steps=learner.sim_steps, wm_updates=learner.wm_updates, actor_updates=learner.actor_updates,
                      final=identity(final), selected=identity(a.output / "checkpoint_selection.json"))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if clock is not None:
            clock.end()
        for physical in (eval_env, env):
            if physical is not None:
                physical.close()
                physical.backend.close()
        status["ended_utc"] = stamp()
        save(a.output / "status.json", status)
        save(a.output / "receipt.json", status)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--code-commit", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(arguments())
