"""One independent, one-hour residual learner with in-process physical checks."""

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
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "project_tools")]
from worker import prepare, process_identity
from balance_adapter import BalanceActor, BalanceAdapter, make_ppo, bootstrap_rewards
from balance_evaluation import rollout, summarize_balance, compare, preserve_balance, preserve_random_state
from control import MAX_ALPHA
from resources import save, identity, stamp
from diagnose_q1_upper_reference import policy_digest


class TrainingClock:
    def __init__(self, path):
        self.path, self.total, self.active = path, 0., None
        self.flush()

    def flush(self):
        save(self.path, {"status": "ACTIVE" if self.active is not None else "PAUSED_FOR_NONTRAINING",
            "updated_utc": stamp(), "process": process_identity(), "accumulated_training_seconds": self.total,
            "active_started_monotonic": self.active, "current_cap_seconds": 3600,
            "absolute_maximum_seconds": 3600, "soft_stop_seconds": 3500,
            "definition": "Synchronized physical collection, GAE, PPO update and per-update diagnostics; initialization/evaluation/checkpoint/media excluded."})

    def begin(self):
        if self.active is not None:
            raise RuntimeError("Clock already running")
        torch.cuda.synchronize()
        self.active = time.monotonic()
        self.flush()

    def end(self):
        if self.active is not None:
            torch.cuda.synchronize()
            self.total += time.monotonic() - self.active
            self.active = None
            self.flush()


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def physical_gate(comparisons, initial_comparisons):
    def finite(value):
        return value is not None and bool(np.isfinite(value))
    results = []
    for current, initial in zip(comparisons, initial_comparisons):
        r, b = current["ratios"], initial["ratios"]
        complete = all(finite(v) for v in r.values())
        passed = complete and r["upper_rmse"] <= .85 and r["wrist_relative_rmse"] <= .85
        passed = passed and r["raised_peak_error"] <= .75 and r["root_xy_rmse"] <= 1.10 and r["leg_rmse"] <= 1.10
        passed = passed and r["target_oscillation"] <= 1.20 and r["loaded_slip"] <= 1.20
        passed = passed and current["failures"] <= 1 and current["completion"] >= .98
        severe = not all(finite(b[k]) for k in ("root_xy_rmse", "upper_rmse"))
        severe = severe or initial["failures"] >= 6 or b["root_xy_rmse"] >= 1.25 or b["upper_rmse"] >= 1.25
        results.append({"RF11_physical_targets_passed": bool(passed), "severe_vs_initial_candidate": bool(severe),
                        "fixed_demos_complete": current["fixed_demos_complete"]})
    qualified = sum(r["RF11_physical_targets_passed"] for r in results) >= 2 and all(r["fixed_demos_complete"] for r in results)
    return {"repeats": results, "physical_targets_met": bool(qualified),
            "severe_regression": sum(r["severe_vs_initial_candidate"] for r in results) >= 2,
            "automatic_extension_allowed": False, "maximum_training_seconds": 3600}


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(),
              "code_commit": a.code_commit, "iteration": 0, "training_seconds": 0., "optimizer_steps": 0,
              "actor_optimizer_steps": 0, "new_world_model_updates": 0, "algorithm": "installed RSL-RL PPO, latent14"}
    save(a.output / "status.json", status)
    clock = TrainingClock(a.output / "training_clock.json")
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    adapter = eval_env = None
    iteration = 0
    stages = []
    baseline_paths, initial_paths, static_paths = [], [], []
    final_paths = []
    outcome = "one_hour_budget_completed"
    try:
        torch.set_num_threads(8)
        cfg, physical, _ = prepare(a, num_envs=4096)
        from omegaconf import OmegaConf
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        OmegaConf.save(cfg, a.output / "config.yaml", resolve=True)
        inherited = torch.load(a.checkpoint, map_location=physical.device, weights_only=True)
        teacher = load_policy(physical, inherited).requires_grad_(False).eval()
        teacher_digest = policy_digest(teacher)
        frozen_normalizers = {key: value.clone() for key, value in teacher.state_dict().items() if "_norm." in key}
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        adapter = BalanceAdapter(physical, teacher, thresholds)
        actor = BalanceActor().to(physical.device)
        algorithm = make_ppo(actor, adapter.observations, 4096)
        library = json.loads(a.library.read_text())
        rows = [dict(row, split=split) for split in ("train", "validation") for row in library["splits"][split]]
        if len(library["splits"]["train"]) != 300 or len(library["splits"]["validation"]) != 30:
            raise ValueError("RF13 retains original AP02 300/30 only")
        evaluation = a.output.parent / "evaluation"
        evaluation.mkdir(exist_ok=False)
        combined = evaluation / "evaluation_library.json"
        save(combined, {"splits": {"train": rows}, "source": identity(a.library)})
        with preserve_random_state():
            _, eval_env, _ = prepare(SimpleNamespace(runtime=a.runtime, library=combined, output=evaluation), num_envs=330, eval_mode=True)

        def progress(phase, **values):
            status.update(phase=phase, updated_utc=stamp(), iteration=iteration, training_seconds=clock.total, **values)
            save(a.output / "status.json", status)

        def evaluate(label, mode, compare_paths=None):
            paths, comparisons = [], []
            with preserve_balance(adapter):
                for repeat in range(3):
                    def tick(frame):
                        progress("evaluation", evaluation_label=label, evaluation_repeat=repeat, evaluation_frame=frame)
                    tick(0)
                    trace = rollout(eval_env, teacher, actor.head, thresholds, mode, tick)
                    path = evaluation / f"{label}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize_balance(trace, rows)
                    summary.update(training_seconds=clock.total, repeat=repeat, mode=mode)
                    for key, previous_paths in (compare_paths or {}).items():
                        with np.load(previous_paths[repeat], allow_pickle=False) as stored:
                            previous = {k: stored[k] for k in stored.files}
                        summary[key] = compare(trace, previous, rows)
                        del previous
                    if compare_paths:
                        comparisons.append({key: summary[key] for key in compare_paths})
                    save(path.with_suffix(".json"), summary)
                    paths.append(path)
                    del trace
            if policy_digest(teacher) != teacher_digest:
                raise AssertionError("Evaluation changed frozen teacher/normalizers")
            return paths, comparisons

        def checkpoint(name):
            numpy_rng = np.random.get_state()
            payload = {"format": "Q1_RF13_BALANCE_HEAD_V1", "actor": actor.state_dict(), "critic": algorithm.critic.state_dict(),
                "optimizer": algorithm.optimizer.state_dict(), "source_RF11": identity(a.checkpoint),
                "source_teacher_digest": teacher_digest, "calibration": identity(a.calibration),
                "iteration": iteration, "sim_steps": iteration * 24 * 4096, "training_seconds": clock.total,
                "optimizer_steps": iteration * 20, "actor_optimizer_steps": max(iteration - 10, 0) * 20,
                "new_world_model_updates": 0, "curriculum_cap": adapter.cap,
                "gate": {"alpha": adapter.gate.alpha, "filtered": adapter.gate.filtered, "cooldown": adapter.gate.cooldown},
                "keep_baseline": adapter.keep_baseline, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
                "code_commit": a.code_commit, "saved_utc": stamp(), "latent_std": .1,
                "residual_radius_rad": .15, "maximum_training_seconds": 3600,
                "note": "Head requires the identified frozen RF11 teacher and this adapter. RNG/gate recorded; not a full simulator-resume checkpoint. No automatic resume."}
            path = a.output / name
            atomic_checkpoint(path, payload)
            return path

        baseline_paths, _ = evaluate("RF11_fixedblock", "teacher")
        static_paths, _ = evaluate("static075", "static075", {"RF11": baseline_paths})
        initial_paths = static_paths
        initial_checkpoint = checkpoint("initial.pt")
        save(evaluation / "initial_sources.json", {"RF11_traces": [identity(p) for p in baseline_paths],
            "static075_traces": [identity(p) for p in static_paths], "initial_candidate_traces": [identity(p) for p in initial_paths],
            "checkpoint": identity(initial_checkpoint), "forward_block": 4096, "repeats": 3})
        initial_actor = copy.deepcopy(actor.state_dict())
        actor.head.requires_grad_(False)
        checks = iter((1200., 2400., 3500.))
        next_check = next(checks)
        last_checkpoint = 0.
        while not requested:
            if clock.total >= 3500:
                break
            if clock.total >= 3540 - 40:
                raise RuntimeError("Training reserve exhausted before final evaluation")
            adapter.cap = MAX_ALPHA * min(clock.total / 1200., 1.)
            progress("collecting", status="RUNNING", curriculum_cap=adapter.cap, next_evaluation_seconds=next_check)
            clock.begin()
            report = {}
            try:
                if iteration == 10:
                    for key, value in actor.state_dict().items():
                        torch.testing.assert_close(value, initial_actor[key], rtol=0, atol=0)
                    actor.head.requires_grad_(True)
                    del initial_actor
                with torch.no_grad():
                    for _ in range(24):
                        latent = algorithm.act(adapter.observations)
                        obs, reward, done, extras = adapter.step(latent)
                        algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
                        for key, value in extras["report"].items():
                            report[key] = report.get(key, 0) + value
                    algorithm.compute_returns(adapter.observations)
                progress("updating")
                losses = algorithm.update()
                if not all(np.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("Nonfinite RSL-RL PPO loss")
                for key, value in frozen_normalizers.items():
                    torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
                iteration += 1
            finally:
                clock.end()
            row = {"step": iteration, "training_seconds": clock.total, "sim_steps": iteration * 98304,
                "optimizer_steps": iteration * 20, "actor_optimizer_steps": max(iteration - 10, 0) * 20,
                "curriculum_cap": adapter.cap, "wm_err8": None, "imag_survival_H": None,
                **report, **{"ppo_" + key: value for key, value in losses.items()}}
            with (a.output / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            progress("complete_update", sim_steps=row["sim_steps"], optimizer_steps=row["optimizer_steps"],
                     actor_optimizer_steps=row["actor_optimizer_steps"])
            if clock.total - last_checkpoint >= 120:
                checkpoint("last.pt")
                last_checkpoint = clock.total
            if clock.total < next_check:
                continue
            number = len(stages) + 1
            path = checkpoint(f"stage_{number:02d}.pt")
            final_paths, comparisons = evaluate(f"stage_{number:02d}", "trained_static075",
                {"RF11": baseline_paths, "initial_candidate": initial_paths})
            gate = physical_gate([v["RF11"] for v in comparisons], [v["initial_candidate"] for v in comparisons])
            entry = {"iteration": iteration, "training_seconds": clock.total, "checkpoint": identity(path),
                     "comparisons": comparisons, "gate": gate, "traces": [identity(p) for p in final_paths]}
            stages.append(entry)
            save(a.output / "stage_decisions.json", stages)
            print(json.dumps({"stage": number, "training_seconds": clock.total, "iteration": iteration, "gate": gate}), flush=True)
            if gate["severe_regression"]:
                outcome = "physical_regression_vs_initial_candidate"
                break
            if clock.total >= 3500:
                break
            next_check = next(checks)
        if requested:
            outcome = "STOPPED_SIGNAL"
        clock.end()
        final = checkpoint("final.pt")
        if policy_digest(teacher) != teacher_digest:
            raise AssertionError("Frozen teacher changed during training")
        if not requested and not final_paths:
            raise AssertionError("No final physical evaluation was saved")
        if clock.total > 3600:
            raise RuntimeError("Absolute one-hour training contract violated")
        if not requested:
            _, gated_comparisons = evaluate("final_gated_diagnostic", "gated",
                                           {"RF11": baseline_paths, "initial_candidate": initial_paths, "trained_static075": final_paths})
            save(evaluation / "receipt.json", {"rows": rows, "baseline_traces": [str(p) for p in baseline_paths],
                "initial_traces": [str(p) for p in initial_paths], "static075_traces": [str(p) for p in static_paths],
                "final_traces": [str(p) for p in final_paths], "stages": stages,
                "training_seconds": clock.total, "outcome": outcome, "new_world_model_updates": 0,
                "teacher_unchanged": True, "normalizers_unchanged": True, "fallback_enabled_in_training": False,
                "final_gated_diagnostic": gated_comparisons,
                "initial_policy_equivalence": "Zero residual head executes the static075 physical controller exactly; reuse its three saved traces."})
        status.update(status="STOPPED_SIGNAL" if requested else "TRAINING_COMPLETED", outcome=outcome,
            iteration=iteration, training_seconds=clock.total, sim_steps=iteration * 98304,
            optimizer_steps=iteration * 20, actor_optimizer_steps=max(iteration - 10, 0) * 20,
            teacher_unchanged=True, completed_utc=stamp(), final_checkpoint=identity(final))
        save(a.output / "receipt.json", status)
    except BaseException as error:
        clock.end()
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=stamp(), training_seconds=clock.total)
        save(a.output / "receipt.json", status)
        raise
    finally:
        clock.end()
        if eval_env is not None:
            eval_env.close()
            eval_env.backend.close()
        if adapter is not None:
            adapter.close()
        save(a.output / "status.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--code-commit", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
