"""Single-use short base or residual PPO experiment for GAIT_GOAL01."""

import argparse
import json
from pathlib import Path
import signal
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from trial_support import (Actor, CandidateAdapter, base_learner, bootstrap_rewards,
    candidate_prepare, evaluate_trial, identity, make_ppo, physical_decision,
    preserve_balance, preserve_random_state, reward_types, save, stamp)
from base import preserve_hp
from clock import TrainingClock

CHECKS = (300., 600., 1100.)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = dict(status="INITIALIZING", line=a.line, started_utc=stamp(), code_commit=a.code_commit,
                  training_seconds=0., maximum_training_seconds=1200., user_accepted_checkpoint=None)
    save(a.output / "status.json", status)
    clock = TrainingClock(a.output / "training_clock.json", 0., 1200., torch.cuda.synchronize)
    env = eval_env = adapter = None
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        torch.set_num_threads(8)
        from contact_reference import configure
        from diagnose_q1_upper_reference import policy_digest
        configure(a.library)
        _, env, cfg = candidate_prepare(a, count=4096, base_candidate=a.line == "base")
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        if a.line == "base":
            learner = base_learner(env, cfg, a.output, a.checkpoint)
            actor = algorithm = None
            teacher = learner.pi
        else:
            from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
            teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
            teacher_hash = policy_digest(teacher)
            adapter = CandidateAdapter(env, teacher, thresholds, .5, runtime=a.runtime)
            actor = Actor().to(env.device)
            algorithm = make_ppo(actor, adapter.observations, 4096)
            actor.head.requires_grad_(False)
            initial_actor = {key: value.clone() for key, value in actor.state_dict().items()}
        rows = [dict(row, split=split) for split in ("train", "validation")
                for row in json.loads(a.library.read_text())["splits"][split]]
        if len(rows) != 330 or sum(row["split"] == "train" for row in rows) != 300:
            raise ValueError("Original300/30 reference contract required")
        evaluation = a.output.parent / "evaluation"
        evaluation.mkdir(exist_ok=False)
        combined = evaluation / "evaluation_library.json"
        save(combined, dict(splits=dict(train=rows), source=identity(a.library)))
        with preserve_random_state():
            _, eval_env, _ = candidate_prepare(a, count=330, evaluation=True, library=combined)
            _, Reference = reward_types(a.runtime)
            reference = Reference(eval_env.supermimic_mdp, a.runtime / "q1_assets/scene.xml")
        iterations = actor_steps = wm_steps = critic_steps = 0
        accumulator = 0.
        def progress(**values):
            status.update(status="RUNNING", updated_utc=stamp(), iteration=iterations,
                          training_seconds=clock.total, actor_optimizer_steps=actor_steps,
                          wm_optimizer_steps=wm_steps, critic_optimizer_steps=critic_steps, **values)
            save(a.output / "status.json", status)
        def checkpoint(label):
            path = a.output / (label + ".pt")
            if a.line == "base":
                learner.wall_time = clock.total
                learner.save(path)
            else:
                torch.save(dict(format="Q1_GAIT_GOAL01_PPO_V1", actor=actor.state_dict(),
                    critic=algorithm.critic.state_dict(), optimizer=algorithm.optimizer.state_dict(),
                    source_base=identity(a.checkpoint), training_seconds=clock.total,
                    iteration=iterations, upper_reference_alpha=.5, user_accepted_checkpoint=None,
                    maximum_training_seconds=1200., code_commit=a.code_commit), path)
            return identity(path)
        def evaluation_gate(label):
            with preserve_hp(learner) if a.line == "base" else preserve_balance(adapter):
                return evaluate_trial(eval_env, teacher, actor.head if actor else None, thresholds,
                    reference, rows, evaluation, label, clock.total, progress,
                    trained=a.line == "ppo" and iterations > 0)
        initial_checkpoint = checkpoint("initial")
        baseline, initial_summary, initial_paths = evaluation_gate("initial")
        candidates = [dict(label="initial", checkpoint=initial_checkpoint, traces=initial_paths,
                           training_seconds=0., fixed_numeric_passed=False,
                           decision=physical_decision(baseline, baseline))]
        stages = []
        outcome = "short_budget_completed"
        next_gate = 0
        if a.line == "base":
            progress(phase="fresh_replay_collection")
            clock.begin()
            try:
                learner.collect(64)
            finally:
                clock.end()
            learner.reporting = {key: 0. if isinstance(value, float) else 0
                                 for key, value in learner.reporting.items()}
            from unisupermimic.rl.algos.supermimic.learner import _empty_stats
            learner.stats = _empty_stats()
        while not requested and clock.total < 1100.:
            progress(phase="training", next_gate_seconds=CHECKS[next_gate])
            clock.begin()
            try:
                if a.line == "base":
                    accumulator += 1. / learner.cfg.utd
                    steps = int(accumulator)
                    accumulator -= steps
                    learner.collect(steps)
                    for _ in range(learner.cfg.wm_batches):
                        learner.wm_update()
                        wm_steps += 1
                    original_step = learner.opt_pi.step
                    if iterations < 10:
                        learner.opt_pi.step = lambda *values, **options: None
                    try:
                        learner.policy_update()
                    finally:
                        learner.opt_pi.step = original_step
                    actor_steps += int(iterations >= 10)
                    critic_steps += 20
                    learner.iteration += 1
                    report = dict(learner.reporting,
                                  wm_err8=learner.latest_wm[1], imag_survival_H=learner.latest_policy[4],
                                  reward_mean=learner.reporting["reward_sum"] / max(learner.reporting["transitions"], 1),
                                  actor_loss=learner.latest_policy[0], critic_loss=learner.latest_policy[1],
                                  reward_maximum=.11, learning_rate=learner.cfg.lr)
                    learner.reporting = {key: 0. if isinstance(value, float) else 0
                                         for key, value in learner.reporting.items()}
                    from unisupermimic.rl.algos.supermimic.learner import _empty_stats
                    learner.stats = _empty_stats()
                else:
                    if iterations == 10:
                        if any(not torch.equal(value, initial_actor[key])
                               for key, value in actor.state_dict().items()):
                            raise AssertionError("Critic-only warm-up changed actor")
                        actor.head.requires_grad_(True)
                    reports = {}
                    with torch.no_grad():
                        for _ in range(24):
                            latent = algorithm.act(adapter.observations)
                            obs, rewards, done, extras = adapter.step(latent)
                            algorithm.process_env_step(obs, bootstrap_rewards(rewards, extras, algorithm.critic), done, {})
                            for key, value in extras["report"].items():
                                reports[key] = reports.get(key, 0) + value
                        algorithm.compute_returns(adapter.observations)
                    sample = algorithm.storage.observations[0].clone()
                    with torch.no_grad():
                        previous_mean = actor(sample).clone()
                    losses = algorithm.update()
                    if not all(np.isfinite(value) for value in losses.values()):
                        raise FloatingPointError("Nonfinite PPO update")
                    actor_steps += 20 * int(iterations >= 10)
                    critic_steps += 20
                    with torch.no_grad():
                        current_mean = actor(sample)
                        kl = .5 * ((current_mean - previous_mean) / .1).square().sum(-1)
                        reports.update(policy_update_kl_mean=float(kl.mean()),
                            policy_update_kl_p99=float(torch.quantile(kl, .99)),
                            latent_mean_rms=float(current_mean.square().mean().sqrt()))
                    report = dict(reports, wm_err8=None, imag_survival_H=None,
                        reward_mean=reports["reward_sum"] / reports["transitions"],
                        learning_rate=algorithm.learning_rate, reward_maximum=.15,
                        **{"ppo_" + key: value for key, value in losses.items()})
                iterations += 1
            finally:
                clock.end()
            report.update(step=iterations, training_seconds=clock.total, actor_optimizer_steps=actor_steps,
                          critic_optimizer_steps=critic_steps, wm_optimizer_steps=wm_steps)
            with (a.output / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            progress(phase="update_completed")
            if clock.total >= CHECKS[next_gate]:
                label = f"stage_{next_gate+1:02d}"
                saved = checkpoint(label)
                metrics, summaries, paths = evaluation_gate(label)
                decision = physical_decision(baseline, metrics)
                validation = [summary["summary"]["validation"]["failures"] for summary in summaries]
                decision["validation_failures"] = validation
                decision["physical_regression_stop"] |= sum(value >= 6 for value in validation) >= 2
                row = dict(label=label, checkpoint=saved, traces=paths, training_seconds=clock.total,
                           fixed_numeric_passed=decision["fixed_numeric_passed"], decision=decision)
                candidates.append(row)
                stages.append(row)
                save(a.output / "stage_decisions.json", stages)
                if decision["physical_regression_stop"]:
                    outcome = "physical_regression"
                    break
                if decision["fixed_numeric_passed"]:
                    outcome = "fixed_numeric_candidate_pending_review"
                    break
                next_gate += 1
                if next_gate == len(CHECKS):
                    break
        if requested:
            raise InterruptedError("Owned run stopped; no automatic retry")
        if clock.total >= 1200.:
            raise RuntimeError("Training cap exceeded")
        final_checkpoint = checkpoint("final")
        final_candidate = candidates[-1]
        if final_candidate["training_seconds"] != clock.total:
            metrics, summaries, paths = evaluation_gate("final")
            final_candidate = dict(label="final", checkpoint=final_checkpoint, traces=paths,
                                  training_seconds=clock.total, decision=physical_decision(baseline, metrics))
            candidates.append(final_candidate)
        def rank(candidate):
            decision = candidate["decision"]
            failures = sum(not row["passed"] for row in decision["per_repeat"])
            ratios = [value["ratio"] for row in decision["per_walk"] for value in row["metrics"].values()]
            return (int(not decision["fixed_numeric_passed"]), failures,
                    max(value if value is not None else 1e6 for value in ratios), candidate["training_seconds"])
        recommended = min(candidates, key=rank)
        if a.line == "ppo" and policy_digest(teacher) != teacher_hash:
            raise AssertionError("PPO changed the frozen teacher")
        save(a.output / "checkpoint_selection.json", dict(candidates=candidates,
            numeric_recommendation=recommended, final=final_candidate, user_accepted_checkpoint=None,
            automatic_extension=False, goal_complete=False))
        save(evaluation / "receipt.json", dict(rows=rows, initial_traces=[row["path"] for row in initial_paths],
            final_traces=[row["path"] for row in final_candidate["traces"]], training_seconds=clock.total,
            recommended_traces=[row["path"] for row in recommended["traces"]],
            line=a.line, upper_reference_alpha=.5, user_accepted_checkpoint=None))
        status.update(status="TRAINING_COMPLETED", outcome=outcome, training_seconds=clock.total,
                      final_checkpoint=final_checkpoint, optimizer_steps=actor_steps+critic_steps+wm_steps,
                      actor_optimizer_steps=actor_steps, critic_optimizer_steps=critic_steps, wm_optimizer_steps=wm_steps)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        clock.end()
        if adapter is not None:
            adapter.close()
            env = None
        for physical in (eval_env, env):
            if physical is not None:
                physical.close()
                physical.backend.close()
        status.update(ended_utc=stamp(), training_seconds=clock.total)
        save(a.output / "status.json", status)
        save(a.output / "receipt.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--code-commit", required=True)
    run(parser.parse_args())
