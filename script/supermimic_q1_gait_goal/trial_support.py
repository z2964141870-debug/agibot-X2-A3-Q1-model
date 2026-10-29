"""Shared new-trial setup and saved physical evidence for GAIT_GOAL01."""

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_comp"),
               str(HERE.parent / "supermimic_q1_gait"), str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "supermimic_q1_balance"),
               str(HERE.parent / "supermimic_q1_contact_comp")]
from worker import prepare, check_contract, learner_class
from resources import identity, save, stamp
from physical import rollout, preserve_balance, preserve_random_state
from evaluation import summarize, clip_metrics
from metrics import gait_clip
from physical_gate import FIXED, evaluate
from contact_adapter import Adapter as OriginalAdapter, Actor, make_ppo, bootstrap_rewards


def reward_types(runtime):
    sys.path[:0] = [str(runtime / "src"), str(runtime)]
    from reward_candidates import PhaseSoleMDP, PhaseSoleReference
    return PhaseSoleMDP, PhaseSoleReference


def candidate_prepare(args, *, count, base_candidate=False, evaluation=False, library=None):
    from types import SimpleNamespace
    a = SimpleNamespace(runtime=args.runtime, library=library or args.library, output=args.output)
    cfg, env, learner_cfg = prepare(a, num_envs=count, eval_mode=evaluation)
    Base, _ = reward_types(args.runtime)
    if base_candidate:
        old = env.supermimic_mdp
        env._supermimic_command.mdp = Base(old.kin, old.motion, old.default_q, old.cfg,
            scene=args.runtime / "q1_assets/scene.xml", library=args.library, observation_mode=old.observation_mode)
        import unisupermimic.tasks.motion_tracking.q1.supermimic.manager_terms as terms
        terms.MAX_STEP_REWARD = env.supermimic_mdp.max_step_reward
    learner_cfg.compile = False
    return cfg, env, learner_cfg


def base_learner(env, config, output, checkpoint):
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY, validate_policy_input_spec
    saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
    cfg = SuperMimicConfig(**{field.name: saved["args"][field.name]
                             for field in fields(SuperMimicConfig) if field.name in saved["args"]})
    cfg.compile = False
    cfg.lr = 5e-5
    Parent, _ = learner_class()
    class Observed(Parent):
        def collect(self, steps):
            original_step = self.env.step
            def step(action):
                state = original_step(action)
                transition = state.info[SUPERMIMIC_TRANSITION_KEY]
                report = self.reporting
                report["transitions"] += self.env.num_envs
                report["reward_sum"] += float(transition["r"].sum())
                count = int(transition["done"].sum())
                report["completed_episodes"] += count
                report["hard_terminations"] += int(transition["term"].sum())
                if count:
                    if transition["ep_len"].numel() != count:
                        raise AssertionError("Completed-episode contract changed")
                    report["normalized_return_sum"] += float(transition["ep_ret_norm"].sum())
                    report["length_steps_sum"] += float(transition["ep_len"].sum())
                return state
            self.env.step = step
            try:
                return SuperMimicLearner.collect(self, steps)
            finally:
                self.env.step = original_step
    learner = Observed(env, cfg, output, device="cuda:0", backend="mjwarp")
    validate_policy_input_spec(saved, learner.mdp)
    for name in ("pi", "wm", "critic"):
        getattr(learner, name).load_state_dict(saved[name])
    learner.iteration = 0
    learner.wm_updates = learner.actor_updates = 0
    return learner


class CandidateAdapter(OriginalAdapter):
    def __init__(self, env, teacher, thresholds, upper_alpha=.5, *, runtime):
        super().__init__(env, teacher, thresholds, upper_alpha)
        _, Reference = reward_types(runtime)
        self.foot_reference = Reference(env.supermimic_mdp, runtime / "q1_assets/scene.xml")
        self.observations, self.nominal, self.reference_action = self.build(
            self.history, self.current, self.gate.alpha, self.gate.filtered,
            self.gate.cooldown, self.caps())


def fixed_metrics(trace, rows, reference, repeat, label):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    output = []
    for index, row in enumerate(rows):
        if row["name"] not in FIXED:
            continue
        count = int(trace["steps"][index])
        metrics = dict(gait_clip(trace, index), **{
            key: value for key, value in clip_metrics(trace, index).items()
            if key not in ("steps", "failed", "completion")})
        frames = torch.from_numpy(trace["frames"][:count, index]).long().to(reference.motion.device)
        start = reference.motion.clip_start[index]
        full_frames = start + torch.arange(1, int(trace["reference_steps"][index]) + 1, device=frames.device)
        expected = full_frames[:count]
        if not torch.equal(frames, expected) or not trace["alive"][:count, index].all():
            raise AssertionError("Fixed metrics require an exact pre-reset prefix")
        with torch.no_grad():
            state = torch.from_numpy(trace["states"][:count, index]).to(frames.device)
            fk = reference.mdp.kin.forward(*unflatten_state(state), all_bodies=True)
            error = reference.sole(fk[0], fk[1]) - reference.reference_sole[frames]
            swing = reference.swing[frames]
            full_swing = reference.swing[full_frames].cpu().numpy()
            rmse = float(error[swing].square().mean().sqrt()) if bool(swing.any()) else None
        metrics.update(label=label, repeat=repeat, clip=row["name"],
            reference_steps=int(trace["reference_steps"][index]),
            swing_sole_z_rmse_m=rmse, swing_sole_foot_frames=int(full_swing.sum()),
            swing_sole_phase_sha256=hashlib.sha256(full_swing.tobytes()).hexdigest())
        if rmse is None and row["name"] != FIXED[2]:
            metrics["swing_sole_z_rmse_m"] = 1e6
        output.append(metrics)
    return output


def physical_decision(baseline, candidate):
    result = evaluate(baseline, candidate)
    severe = any(not row["complete"] for row in result["per_repeat"])
    for row in result["per_repeat"]:
        severe |= any(value["ratio"] is not None and value["ratio"] > 1.25 for value in row["guards"].values())
        severe |= not row.get("progress_preserved", True)
    for row in result["per_walk"]:
        severe |= any(value["ratio"] is None or value["ratio"] > 1.5 for value in row["metrics"].values())
    result["physical_regression_stop"] = bool(severe)
    return result


def evaluate_trial(env, policy, head, thresholds, reference, rows, output, label, seconds,
                   progress=lambda **values: None, trained=False):
    metrics, summaries, paths = [], [], []
    for repeat in range(3):
        progress(phase="evaluation", label=label, repeat=repeat, frame=0)
        trace = rollout(env, policy, head, thresholds, reference,
            mode="trained075" if trained else "static075", upper_alpha=.5,
            progress=lambda frame: progress(phase="evaluation", label=label, repeat=repeat, frame=frame))
        summary = summarize(trace, rows)
        summary.update(repeat=repeat, training_seconds=seconds, label=label)
        fixed = fixed_metrics(trace, rows, reference, repeat, label)
        summary["goal_fixed"] = fixed
        path = output / f"{label}_repeat{repeat}.npz"
        np.savez_compressed(path, **trace)
        save(path.with_suffix(".json"), summary)
        paths.append(identity(path))
        metrics.extend(fixed)
        summaries.append(summary)
    save(output / f"{label}_goal_fixed.json", metrics)
    return metrics, summaries, paths
