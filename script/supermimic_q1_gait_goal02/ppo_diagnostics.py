"""Observe the existing PPO update without changing its actions or loss."""

import json
from pathlib import Path

import torch


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def summary(values, mask):
    selected = values[mask]
    count = selected.numel()
    return dict(count=count, mean=float(selected.mean()) if count else None,
                positive_fraction=float((selected > 0).float().mean()) if count else None)


class PPODiagnostics:
    def __init__(self, algorithm, adapter, output):
        self.algorithm, self.adapter, self.output = algorithm, adapter, Path(output)
        self.iteration = 0
        self.batch = None
        self.minibatches, self.gradient_clips = [], []

    def returns_ready(self):
        storage = self.algorithm.storage
        records = self.adapter.diagnostic_steps
        steps = storage.num_transitions_per_env
        if len(records) != steps:
            raise AssertionError("PPO diagnostics must align one record per stored control transition")
        stacked = {key: torch.stack([row[key] for row in records]) for key in records[0]}
        advantage = storage.advantages.squeeze(-1)
        raw = (storage.returns - storage.values).squeeze(-1)
        if advantage.shape != stacked["frames"].shape:
            raise AssertionError("PPO advantage and transition dimensions disagree")
        conditions = []
        for foot, side in enumerate(("left", "right")):
            swing = stacked["swing"][..., foot] & (stacked["reference_sole"][..., foot] >= .01)
            low_force = stacked["last_solve_force"][..., foot] <= 5.
            high = stacked["sole"][..., foot] > .01
            masks = dict(reference_swing=swing,
                         delayed_low_force_grounded=swing & low_force & (stacked["sole"][..., foot] <= .005),
                         delayed_low_force_raised=swing & low_force & high,
                         delayed_loaded_grounded=swing & ~low_force & (stacked["sole"][..., foot] <= .005))
            for condition, mask in masks.items():
                conditions.append(dict(side=side, condition=condition,
                    normalized_advantage=summary(advantage, mask), raw_advantage=summary(raw, mask)))
        # Deterministic bounded samples do not consume the policy RNG.
        flat = advantage.flatten()
        indices = torch.linspace(0, flat.numel() - 1, min(32, flat.numel()),
                                 device=flat.device).long()
        values = dict(normalized_advantage=advantage, raw_advantage=raw,
                      value=storage.values.squeeze(-1), returns=storage.returns.squeeze(-1),
                      reward=storage.rewards.squeeze(-1),
                      old_log_prob=storage.actions_log_prob.squeeze(-1))
        selected = {key: value.flatten()[indices].detach().cpu().tolist()
                    for key, value in values.items()}
        for key, value in stacked.items():
            selected[key] = value.flatten(0, 1)[indices].detach().cpu().tolist()
        selected["latent_action"] = storage.actions.flatten(0, 1)[indices].detach().cpu().tolist()
        selected["done"] = storage.dones.flatten(0, 1)[indices].detach().cpu().tolist()
        selected["flat_transition_index"] = indices.cpu().tolist()
        samples = [{key: value[i] for key, value in selected.items()} for i in range(len(indices))]
        append(self.output / "ppo_phase_advantages.jsonl", dict(iteration=self.iteration,
            transition_count=flat.numel(), conditions=conditions, samples=samples,
            force_pose_time_offset_seconds=-.005,
            note="Delayed force with post-step pose, not simultaneous contact evidence"))
        self.adapter.diagnostic_steps.clear()

    def attach(self):
        algorithm, actor, storage = self.algorithm, self.algorithm.actor, self.algorithm.storage
        if actor.is_recurrent or algorithm.symmetry is not None or algorithm.rnd is not None:
            raise ValueError("This observer requires the original nonrecurrent PPO without extensions")
        returns = algorithm.compute_returns
        generator = storage.mini_batch_generator
        log_prob = actor.get_output_log_prob
        update = algorithm.update
        process = algorithm.process_env_step

        def process_env_step(*args, **kwargs):
            record = self.adapter.foot_reference.latest_diagnostics
            if record is None or len(self.adapter.diagnostic_steps) >= storage.num_transitions_per_env:
                raise AssertionError("Missing or excess PPO transition diagnostics")
            result = process(*args, **kwargs)
            self.adapter.diagnostic_steps.append(record)
            self.adapter.foot_reference.latest_diagnostics = None
            return result

        def compute_returns(*args, **kwargs):
            result = returns(*args, **kwargs)
            self.returns_ready()
            return result

        def batches(*args, **kwargs):
            try:
                for batch in generator(*args, **kwargs):
                    self.batch = batch
                    yield batch
            finally:
                self.batch = None

        def output_log_prob(*args, **kwargs):
            result = log_prob(*args, **kwargs)
            if self.batch is not None:
                with torch.no_grad():
                    ratio = torch.exp(result - self.batch.old_actions_log_prob.squeeze(-1))
                    advantage = self.batch.advantages.squeeze(-1)
                    bound = algorithm.clip_param
                    active = ((advantage > 0) & (ratio > 1 + bound)
                              | (advantage < 0) & (ratio < 1 - bound))
                    kl = actor.get_kl_divergence(self.batch.old_distribution_params,
                                                actor.output_distribution_params)
                    self.minibatches.append(dict(
                        count=ratio.numel(), ratio_mean=float(ratio.mean()),
                        outside_clip_fraction=float(((ratio < 1 - bound) | (ratio > 1 + bound)).float().mean()),
                        active_clipped_surrogate_fraction=float(active.float().mean()),
                        actual_minibatch_kl_mean=float(kl.mean()),
                        actor_requires_grad=any(p.requires_grad for p in actor.parameters())))
            return result

        def observed_update(*args, **kwargs):
            self.minibatches, self.gradient_clips = [], []
            original_clip = torch.nn.utils.clip_grad_norm_
            actor_ids = {id(p) for p in actor.parameters()}
            critic_ids = {id(p) for p in algorithm.critic.parameters()}

            def clip(parameters, *values, **options):
                parameters = tuple(parameters)
                ids = {id(p) for p in parameters}
                name = "actor" if ids == actor_ids else "critic" if ids == critic_ids else None
                if name is None:
                    raise AssertionError("Unexpected gradient group during observed PPO update")
                before = original_clip(parameters, *values, **options)
                after = sum(float(p.grad.detach().float().square().sum())
                            for p in parameters if p.grad is not None)**.5
                self.gradient_clips.append(dict(module=name, before_norm=float(before), after_norm=after))
                return before

            torch.nn.utils.clip_grad_norm_ = clip
            try:
                result = update(*args, **kwargs)
            finally:
                torch.nn.utils.clip_grad_norm_ = original_clip
                self.batch = None
            expected = algorithm.num_learning_epochs * algorithm.num_mini_batches
            if len(self.minibatches) != expected or len(self.gradient_clips) != 2 * expected:
                raise AssertionError("PPO observer did not see every minibatch and gradient clipping call")
            append(self.output / "ppo_update_diagnostics.jsonl", dict(iteration=self.iteration,
                minibatches=self.minibatches, gradient_clips=self.gradient_clips,
                final_learning_rate=algorithm.learning_rate))
            self.iteration += 1
            return result

        algorithm.compute_returns = compute_returns
        algorithm.process_env_step = process_env_step
        storage.mini_batch_generator = batches
        actor.get_output_log_prob = output_log_prob
        algorithm.update = observed_update
        return self
