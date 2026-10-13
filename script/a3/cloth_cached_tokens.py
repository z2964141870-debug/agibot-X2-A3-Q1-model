"""Frozen-codec PPO evaluates sampled actions with their actual sampled tokens."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_constrained_trainer import ClothConstrainedTrainer, DECODER
from script.a3.zhy_likelihood_trainer import snapshot


def cached_distribution(policy, obs, tokens, actions):
    if any(p.requires_grad for name,p in policy.named_parameters() if not name.startswith(DECODER)):
        raise ValueError('Sampled token reuse requires every other Actor parameter frozen')
    if tokens.requires_grad or tokens.shape != (*actions.shape[:2], 2, 32):
        raise ValueError('Detached sampled token ABI changed')
    if not torch.isfinite(tokens).all():
        raise ValueError('Nonfinite sampled tokens')
    mean = policy.actor_module.forward_with_external_tokens(obs, tokens)
    if mean.shape != actions.shape:
        raise ValueError('External-token decoder action ABI changed')
    std = policy.get_std
    std = torch.where(torch.isfinite(std), std.clamp_min(1e-6), torch.full_like(std, 1e-6))
    policy.distribution = torch.distributions.Normal(mean, std)
    return dict(action_mean=policy.action_mean, action_std=policy.action_std,
        logprobs=policy.get_actions_log_prob(actions), entropy=policy.entropy)


class ClothCachedTokenTrainer(ClothConstrainedTrainer):
    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('Cached-token pilot currently requires one process')
        self._sample_token_records = []
        self._collect_sample_tokens = False
        self._cached_token_batches = []
        policy = self.accelerator.unwrap_model(self.model).policy
        self._sample_token_hook = policy.actor_module.register_forward_hook(self._capture_tokens)
        return result

    def _capture_tokens(self, module, arguments, output):
        if self._collect_sample_tokens:
            value = module._last_full_latent_flat
            if value.shape != (16, 1, 64):
                raise ValueError('Actual sampling token layout changed')
            self._sample_token_records.append(value.detach().clone().reshape(16, 1, 2, 32))

    def _rollout_step(self, *args, **kwargs):
        self._sample_token_records = []
        self._collect_sample_tokens = True
        try:
            return super()._rollout_step(*args, **kwargs)
        finally:
            self._collect_sample_tokens = False

    def _get_rollout_data(self, *args, **kwargs):
        rollout = super()._get_rollout_data(*args, **kwargs)
        if len(self._sample_token_records) != 24:
            raise ValueError('Sampled token coverage differs from rollout')
        self._sampled_tokens = torch.cat(self._sample_token_records, dim=1)
        if self._sampled_tokens.shape[:2] != rollout['old_mu_batch'].shape[:2]:
            raise ValueError('Sampled tokens differ from rollout environment/time identity')
        if not self.state.global_step:
            torch.save(dict(tokens=snapshot(self._sampled_tokens), rollout=snapshot(rollout)),
                       Path(os.environ['YUANQI_RESUME_LOG_DIR']) / 'initial_sampled_tokens.pt')
        return rollout

    def _get_mb_rollout_data(self, rollout, indices):
        batch = super()._get_mb_rollout_data(rollout, indices)
        batch['mb_sampled_tokens'] = self._sampled_tokens[indices]
        return batch

    def _forward_model(self, model, batch):
        self._assert_frozen()
        result = super()._forward_model(model, batch)
        raw = snapshot(result['policy_results']['action_mean']) if not self._cached_token_batches else None
        policy = self.accelerator.unwrap_model(self.model).policy
        result['policy_results'].update(cached_distribution(policy, batch['mb_obs_dict'],
            batch['mb_sampled_tokens'], batch['mb_actions']))
        self._cached_token_batches.append(dict(global_step=int(self.state.global_step),
            optimizer_steps_before=self._optimizer_step_ordinal,
            environment_indices=batch['micro_batch_inds'].detach().cpu().tolist(),
            tokens_shape=list(batch['mb_sampled_tokens'].shape),
            decoder_gradient_enabled=result['policy_results']['action_mean'].requires_grad))
        directory = Path(os.environ['YUANQI_RESUME_LOG_DIR'])
        if raw is not None:
            torch.save(dict(raw_reencoded_mean=raw, corrected=snapshot(result['policy_results']),
                batch=snapshot(batch)), directory / 'initial_token_repair.pt')
            self._save_precision_contract()
        atomic_json(directory / 'sampled_token_contract.json', dict(
            batches=self._cached_token_batches, frozen_codec_required=True,
            collection_source='ACTUAL_ASSEMBLED_TOKEN_AT_SAMPLING_NOT_REENCODED_OR_TEACHER_MEAN',
            auxiliary_loss_preserved=True, decoder_gradients_preserved=True,
            batch_identity_uses_same_environment_indices_as_rollout=True))
        return result
