"""Fresh035 zzx expert: native PPO, full decoder, frozen codec, sampled tokens."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_constrained_trainer import initial_probability, restrict_policy
from script.a3.consistent_precision import ConsistentPrecisionTrainer, require_fp32
from script.a3.gradient_probe import parameter_digest
from script.a3.humanplus_specialist_protocol import (
    CHECKPOINTS, DECODER, ENVS, MINIBATCHES, ROUTE, STEPS, UPDATES,
    ownership, serialized_owners,
)
from script.a3.resumable_sonic import DurableModelSaveCallback
from script.a3.zhy_likelihood_trainer import snapshot


class SpecialistTrainer(ConsistentPrecisionTrainer):
    """No constrained/head/finite trainer appears in this class's MRO."""

    def create_optimizer(self):
        self._expert_scope = restrict_policy(self.model.policy)
        result = super().create_optimizer()
        owned = [p for group in self.optimizer.param_groups for p in group['params']]
        expected = [p for p in self.model.parameters() if p.requires_grad]
        if len(set(map(id, owned))) != len(owned) or set(map(id, owned)) != set(map(id, expected)):
            raise ValueError('Exact decoder plus original Critic optimizer ownership required')
        self._expert_ownership = ownership(self.model, self.optimizer)
        return result

    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError('Specialist trial has no recovery')
        result = super().load_checkpoint(checkpoint_path, resume=False)
        if self.accelerator.num_processes != 1:
            raise ValueError('Single GPU only')
        policy = self.accelerator.unwrap_model(self.model).policy
        if policy.running_mean_std is not None:
            raise ValueError('Unexpected Actor normalization')
        policy.get_std
        self._expert_frozen = [(n, p) for n, p in policy.named_parameters() if not p.requires_grad]
        self._expert_frozen_digest = parameter_digest(self._expert_frozen)
        self._expert_buffers = {n: p.detach().cpu().clone() for n, p in policy.named_buffers()}
        self._expert_directory = Path(os.environ['YUANQI_RESUME_LOG_DIR'])
        self._expert_collect = False
        self._expert_tokens = []
        self._expert_rollouts = 0
        self._expert_native_calls = 0
        self._expert_decoder_calls = 0
        self._expert_rollout_calls = 0
        self._expert_probability = None
        self._expert_hook = policy.actor_module.register_forward_hook(self._capture_tokens)
        self._save_expert()
        return result

    def _assert_expert_frozen(self, *, hash_parameters=True):
        policy = self.accelerator.unwrap_model(self.model).policy
        if ((hash_parameters and parameter_digest(self._expert_frozen) != self._expert_frozen_digest)
                or any(p.grad is not None for _, p in self._expert_frozen)):
            raise ValueError('Frozen codec or std changed')
        live = dict(policy.named_buffers())
        if live.keys() != self._expert_buffers.keys() or any(
                not torch.equal(p.detach().cpu(), self._expert_buffers[n]) for n, p in live.items()):
            raise ValueError('Actor buffer changed')

    def _capture_tokens(self, module, arguments, output):
        if not self._expert_collect:
            return
        value = module._last_full_latent_flat
        route = module.parse_tokenizer_obs(arguments[0])['encoder_index']
        if (value.shape != (ENVS, 1, 64) or route.shape != (ENVS, 1, 4)
                or not torch.equal(route, route.new_tensor(ROUTE).expand_as(route))):
            raise ValueError('Actual native route or sampled-token ABI changed')
        self._expert_tokens.append(value.detach().clone().reshape(ENVS, 1, 2, 32))
        self._expert_rollout_calls += 1

    def _rollout_step(self, *args, **kwargs):
        if self._expert_rollouts >= UPDATES:
            raise ValueError('Specialist rollout budget exhausted')
        self._expert_tokens = []
        self._expert_collect = True
        try:
            value = super()._rollout_step(*args, **kwargs)
        finally:
            self._expert_collect = False
        if len(self._expert_tokens) != STEPS:
            raise ValueError('Missing sampled tokens')
        self._expert_rollouts += 1
        return value

    def _get_rollout_data(self, *args, **kwargs):
        value = super()._get_rollout_data(*args, **kwargs)
        self._expert_sampled = torch.cat(self._expert_tokens, dim=1)
        if self._expert_sampled.shape[:2] != value['old_mu_batch'].shape[:2]:
            raise ValueError('Sampled token environment/time identity changed')
        self._assert_expert_frozen()
        if self._expert_rollouts == 1:
            torch.save(dict(tokens=snapshot(self._expert_sampled), rollout=snapshot(value)),
                       self._expert_directory / 'initial_sampled_tokens.pt')
        return value

    def _get_mb_rollout_data(self, rollout, indices):
        value = super()._get_mb_rollout_data(rollout, indices)
        value['mb_sampled_tokens'] = self._expert_sampled[indices]
        return value

    def _forward_model(self, model, batch):
        if self._expert_native_calls >= UPDATES * MINIBATCHES:
            raise ValueError('Training forward budget exhausted')
        result = super()._forward_model(model, batch)
        self._expert_native_calls += 1
        policy = self.accelerator.unwrap_model(self.model).policy
        result['policy_results'].update(cached_distribution(
            policy, batch['mb_obs_dict'], batch['mb_sampled_tokens'], batch['mb_actions']))
        self._expert_decoder_calls += 1
        if self._expert_probability is None:
            if self.state.global_step or self._optimizer_step_ordinal:
                raise ValueError('Initial probability check occurred after an update')
            self._expert_probability = initial_probability(result['policy_results'], batch)
            torch.save(dict(corrected=snapshot(result['policy_results']), batch=snapshot(batch)),
                       self._expert_directory / 'initial_probability.pt')
        return result

    def _compute_loss(self, forward_results, batch):
        require_fp32(self.accelerator, self.args)
        result = super()._compute_loss(forward_results, batch)
        if not torch.isfinite(result['loss']):
            raise ValueError('Nonfinite specialist objective')
        return result

    def _before_optimizer_step(self, optimizer, args, kwargs):
        if self._optimizer_step_ordinal >= UPDATES * MINIBATCHES:
            raise ValueError('Specialist optimizer budget exhausted')
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise ValueError('Nonfinite specialist gradient')
        return super()._before_optimizer_step(optimizer, args, kwargs)

    def _after_optimizer_step(self, optimizer, args, kwargs):
        super()._after_optimizer_step(optimizer, args, kwargs)
        self._assert_expert_frozen(hash_parameters=False)
        if any(not torch.isfinite(p).all() for group in optimizer.param_groups for p in group['params']):
            raise ValueError('Nonfinite specialist parameter')
        self._save_expert()

    def _save_expert(self):
        atomic_json(self._expert_directory / 'specialist_contract.json', dict(
            updates=self._expert_rollouts, interactions=self._expert_rollouts * ENVS * STEPS,
            optimizer_steps=getattr(self, '_optimizer_step_ordinal', 0),
            rollout_actor_calls=self._expert_rollout_calls,
            native_training_actor_calls=self._expert_native_calls,
            cached_decoder_calls=self._expert_decoder_calls, teacher_retention_calls=0,
            initial_probability=self._expert_probability, scope=self._expert_scope,
            frozen_parameter_digest=self._expert_frozen_digest,
            all_actor_buffers_unchanged=True, decoder_prefix=DECODER,
            actual_sampled_tokens=True, original_auxiliary_objective_preserved=True,
            ownership=self._expert_ownership,
            role='INTERMEDIATE_ZZX_EXPERT_NOT_UNIFIED_POLICY'))

    def append_to_log_dict(self, log_dict):
        super().append_to_log_dict(log_dict)
        from script.a3.humanplus_supervisor import append
        metrics = {k: float(v) for k, v in log_dict.items() if isinstance(v, (int, float))}
        append(self._expert_directory / 'learning_metrics.jsonl', dict(
            update=int(self.state.global_step), optimizer_steps=self._optimizer_step_ordinal, metrics=metrics))


class SpecialistCheckpointCallback(DurableModelSaveCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target_global_step != UPDATES:
            raise ValueError('Fixed specialist budget required')

    def _save(self, args, state, **kwargs):
        if state.global_step not in CHECKPOINTS:
            raise ValueError('Only initial and final specialist checkpoints permitted')
        original = self.save_dir
        self.save_dir = original / ('milestone_' + str(state.global_step))
        try:
            result = super()._save(args, state, **kwargs)
            optimizer = kwargs['optimizer']
            contract = ownership(kwargs['model'], optimizer)
            atomic_json(self.save_dir / 'specialist_owners.json', dict(
                ownership=contract,
                serialized_id_names=serialized_owners(optimizer.state_dict()['param_groups'], contract),
                optimizer_options=[{k: v for k, v in g.items() if k not in ('params', 'lr', 'initial_lr')}
                                   for g in optimizer.param_groups]))
            return result
        finally:
            self.save_dir = original

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == UPDATES:
            self._save(args, state, **kwargs)
            control.should_training_stop = True
        return control
