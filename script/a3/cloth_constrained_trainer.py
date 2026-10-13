"""Small clothing PPO trial with a frozen codec and fixed official action anchors."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.consistent_precision import ConsistentPrecisionTrainer, require_fp32
from script.a3.gradient_probe import parameter_digest
from script.a3.likelihood_probe import compare
from script.a3.zhy_likelihood_trainer import snapshot


DECODER = 'actor_module.decoders.g1_dyn.'


def restrict_policy(policy):
    selected, frozen = [], []
    for name, value in policy.named_parameters():
        allowed = name.startswith(DECODER)
        value.requires_grad_(allowed)
        (selected if allowed else frozen).append(name)
    if (not selected or not any(n.startswith('actor_module.encoders.') for n in frozen)
            or 'std' not in frozen):
        raise ValueError('Expected official encoder, dynamics decoder and std are absent')
    return dict(trainable=selected, frozen=frozen)


def initial_probability(result, batch):
    valid = ~batch['mb_padding_mask']
    if not valid.any():
        raise ValueError('Initial PPO batch has no valid transitions')
    record = compare(result['action_mean'][valid], batch['mb_old_mu'][valid],
        result['action_std'][valid], batch['mb_old_sigma'][valid],
        batch['mb_actions'][valid], batch['mb_logprobs'][valid])
    if record['action_rmse'] > 1e-5 or record['max_abs_ratio_minus_one'] > .001:
        raise ValueError('Initial clothing likelihood contract failed: ' + repr(record))
    return record


class ClothConstrainedTrainer(ConsistentPrecisionTrainer):
    def create_optimizer(self):
        self._cloth_scope = restrict_policy(self.model.policy)
        optimizer = super().create_optimizer()
        ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if any(id(p) in ids for p in self.model.parameters() if not p.requires_grad):
            raise ValueError('Optimizer owns a frozen parameter')
        return optimizer

    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError('This clothing pilot has no recovery')
        result = super().load_checkpoint(checkpoint_path, resume=False)
        require_fp32(self.accelerator, self.args)
        policy = self.accelerator.unwrap_model(self.model).policy
        if policy.running_mean_std is not None:
            raise ValueError('Clothing pilot requires the verified non-normalized Actor')
        policy.get_std
        self._cloth_frozen = [(n, p) for n, p in policy.named_parameters() if not p.requires_grad]
        self._cloth_frozen_digest = parameter_digest(self._cloth_frozen)
        self._cloth_loaded_buffers = snapshot(dict(policy.named_buffers()))
        self._cloth_directory = Path(os.environ['YUANQI_RESUME_LOG_DIR'])
        self._cloth_records = []
        self._cloth_initial_probability = None
        rollout = torch.load(os.environ['YUANQI_CLOTH_ANCHOR_ROLLOUT'],
                             map_location='cpu', weights_only=False)
        golden = torch.load(os.environ['YUANQI_CLOTH_ANCHOR_OUTPUTS'],
                            map_location='cpu', weights_only=False)['initial']['mean']
        self._cloth_obs = {name: value.to(self.accelerator.device)
                           for name, value in rollout['all_obs_dict'].items()}
        self._cloth_targets = golden.to(self.accelerator.device)
        if self._cloth_targets.shape != (16, 24, 29):
            raise ValueError('Official anchor coverage changed')
        with torch.no_grad():
            means = torch.cat([policy.forward({n: v[i:i+4] for n, v in self._cloth_obs.items()},
                is_training=False) for i in range(0, 16, 4)], dim=0)
        difference = float((means - self._cloth_targets).abs().max())
        if difference > 1e-5:
            raise ValueError('Official saved teacher anchors do not match warm start')
        self._cloth_initial_anchor_difference = difference
        self._save_cloth_receipt()
        return result

    def _get_rollout_data(self, *args, **kwargs):
        result = super()._get_rollout_data(*args, **kwargs)
        self._assert_frozen()
        return result

    def _compute_loss(self, forward_results, mb_rollout_data):
        require_fp32(self.accelerator, self.args)
        if self._cloth_initial_probability is None:
            if self.state.global_step or self._optimizer_step_ordinal:
                raise ValueError('Initial contract was not checked before the first update')
            self._cloth_initial_probability = initial_probability(
                forward_results['policy_results'], mb_rollout_data)
            torch.save(dict(policy=snapshot(forward_results['policy_results']),
                batch=snapshot(mb_rollout_data)), self._cloth_directory / 'initial_probability.pt')
            self._save_cloth_receipt()
        result = super()._compute_loss(forward_results, mb_rollout_data)
        policy = self.accelerator.unwrap_model(self.model).policy
        start = self._optimizer_step_ordinal % 4 * 4
        means = policy.forward({n: v[start:start+4] for n, v in self._cloth_obs.items()},
                               is_training=False)
        retention = (means - self._cloth_targets[start:start+4]).square().mean()
        if not torch.isfinite(retention):
            raise ValueError('Nonfinite official teacher retention loss')
        result['loss'] = result['loss'] + retention
        self._cloth_last_retention_loss = float(retention.detach())
        return result

    def _assert_frozen(self):
        policy = self.accelerator.unwrap_model(self.model).policy
        if (parameter_digest(self._cloth_frozen) != self._cloth_frozen_digest
                or any(p.grad is not None for _, p in self._cloth_frozen)):
            raise ValueError('Frozen Actor parameter or gradient changed')
        live = dict(policy.named_buffers())
        if live.keys() != self._cloth_loaded_buffers.keys() or any(
                not torch.equal(v.detach().cpu(), self._cloth_loaded_buffers[n]) for n, v in live.items()):
            raise ValueError('Frozen Actor buffer changed')

    def _after_optimizer_step(self, optimizer, args, kwargs):
        super()._after_optimizer_step(optimizer, args, kwargs)
        self._assert_frozen()
        policy = self.accelerator.unwrap_model(self.model).policy
        with torch.no_grad():
            means = torch.cat([policy.forward({n: v[i:i+4] for n, v in self._cloth_obs.items()},
                is_training=False) for i in range(0, 16, 4)], dim=0)
        difference = means - self._cloth_targets
        row = dict(optimizer_step=self._optimizer_step_ordinal,
            teacher_action_rmse=float(difference.square().mean().sqrt()),
            teacher_action_max_abs=float(difference.abs().max()),
            retention_loss_before_step=self._cloth_last_retention_loss,
            frozen_parameter_digest=parameter_digest(self._cloth_frozen))
        self._cloth_records.append(row)
        self._save_cloth_receipt()
        if row['teacher_action_rmse'] > .05 or row['teacher_action_max_abs'] > .5:
            raise ValueError('Official action anchor drift budget exceeded; stop without retry')

    def _save_cloth_receipt(self):
        atomic_json(self._cloth_directory / 'cloth_contract.json', dict(
            scope=self._cloth_scope, initial_anchor_max_abs=self._cloth_initial_anchor_difference,
            frozen_parameter_digest=self._cloth_frozen_digest,
            initial_probability=self._cloth_initial_probability, anchor_frames=384,
            retention_coefficient=1., anchor_action_rmse_budget=.05,
            anchor_action_max_abs_budget=.5, optimizer_steps=self._cloth_records,
            anchors_are_official_development_states_not_unseen_test=True))
