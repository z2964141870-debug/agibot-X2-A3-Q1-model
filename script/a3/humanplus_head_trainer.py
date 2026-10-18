"""D01: preserve finite PPO transactions, train exactly the original action head."""

import json
import os
from pathlib import Path

import torch
from transformers import Trainer

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.finite_adam import FiniteAdamW
from script.a3.finite_retention_trainer import FiniteRetentionTrainer, FiniteCheckpointCallback
from script.a3.humanplus_head_protocol import HEAD_KEYS, digest, ownership, restrict_head, serialized_owners
from script.a3.retention_direction_probe import tree_digest


def group_options(optimizer):
    return [{k: v for k, v in group.items() if k not in ('params', 'lr', 'initial_lr')}
            for group in optimizer.param_groups]


class HeadOnlyTrainer(FiniteRetentionTrainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            if self.optimizer is not getattr(self, '_head_optimizer', None):
                raise ValueError('Only the originally constructed finite optimizer may be reused')
            self._check_head(self.model)
            return self.optimizer
        if self.optimizer_cls_and_kwargs is not None:
            raise ValueError('Arbitrary optimizer overrides are outside D01')
        self._cloth_scope = restrict_head(self.model.policy)
        # Bypass ClothConstrainedTrainer, which would re-enable the entire decoder.
        native = Trainer.create_optimizer(self)
        if type(native) is not torch.optim.AdamW or native.state:
            raise ValueError('Pinned native empty AdamW required')
        contract = ownership(self.model, native)
        groups = [dict(group, params=list(group['params'])) for group in native.param_groups]
        options = group_options(native)
        self.optimizer = FiniteAdamW(groups, evaluate=self._evaluate_candidate, record=self._record_candidate)
        if group_options(self.optimizer) != options or ownership(self.model, self.optimizer) != contract:
            raise ValueError('Finite wrapping changed native ownership or hyperparameters')
        self._head_optimizer = self.optimizer
        self._head_contract = contract
        self._head_options = options
        self._head_owner_sha256 = digest(contract)
        return self.optimizer

    def _check_head(self, wrapper=None):
        if wrapper is None:
            wrapper = self.accelerator.unwrap_model(self.model)
        if (ownership(wrapper, self._head_optimizer) != self._head_contract
                or group_options(self._head_optimizer) != self._head_options):
            raise ValueError('Head scope, group ordering or Adam options changed')
        if any(p.grad is not None for n, p in wrapper.policy.named_parameters() if n not in HEAD_KEYS):
            raise ValueError('Frozen Actor acquired a gradient')
        if hasattr(self, '_cloth_frozen_digest'):
            self._assert_frozen()
        return wrapper

    def load_checkpoint(self, checkpoint_path, resume=False):
        if resume:
            raise ValueError('D01 has no resume path')
        result = super().load_checkpoint(checkpoint_path, resume=False)
        wrapper = self._check_head()
        self._head_initial_state = tree_digest(dict(wrapper.policy.named_parameters()))
        self._head_steps = []
        self._save_head()
        return result

    def _save_head(self):
        atomic_json(self._cloth_directory / 'head_contract.json', dict(
            ownership=self._head_contract, owner_sha256=self._head_owner_sha256,
            native_group_options=self._head_options, initial_actor_sha256=self._head_initial_state,
            scope='EXACT_ORIGINAL_OUTPUT_HEAD_AND_ORIGINAL_CRITIC', steps=self._head_steps))

    def _forward_model(self, model, batch):
        self._check_head()
        return super()._forward_model(model, batch)

    def _before_optimizer_step(self, optimizer, args, kwargs):
        wrapper = self._check_head()
        gradients = {}
        for name, parameter in wrapper.policy.named_parameters():
            if name in HEAD_KEYS:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError('Missing or nonfinite output-head gradient')
                gradients[name] = dict(shape=list(parameter.grad.shape),
                    sha256=tree_digest(parameter.grad), max_abs=float(parameter.grad.abs().max()))
        self._head_gradients = gradients
        return super()._before_optimizer_step(optimizer, args, kwargs)

    def _record_candidate(self, row):
        self._check_head()
        return super()._record_candidate(dict(row, owner_sha256=self._head_owner_sha256,
                                              head_gradients=self._head_gradients))

    def _after_optimizer_step(self, optimizer, args, kwargs):
        result = super()._after_optimizer_step(optimizer, args, kwargs)
        self._check_head()
        self._head_steps.append(dict(commit=self._optimizer_step_ordinal,
            proposal=self._finite_proposals, owner_sha256=self._head_owner_sha256,
            gradients=self._head_gradients, frozen_actor_unchanged=True))
        self._save_head()
        return result

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except BaseException:
            wrapper = self._check_head()
            pending = getattr(self, '_finite_pending_batch', False)
            committed = (self._finite_last_committed is not None and
                         self._finite_last_committed['proposal'] == self._finite_proposals)
            before = tree_digest(self._finite_before_buffers) if pending else None
            after = tree_digest(dict(wrapper.named_buffers()))
            if pending and not committed and before != after:
                raise ValueError('Uncommitted batch buffer rollback failed')
            atomic_json(self._cloth_directory / 'head_stop.json', dict(
                owner_sha256=self._head_owner_sha256, pending=pending, committed=committed,
                before_buffer_sha256=before, after_buffer_sha256=after,
                frozen_actor_unchanged=True, tensors_reloaded=0))
            raise


class HeadCheckpointCallback(FiniteCheckpointCallback):
    def _save(self, args, state, **kwargs):
        result = super()._save(args, state, **kwargs)
        if self.saved_step != state.global_step:
            return result
        directory = self.save_dir / ('milestone_' + str(state.global_step))
        target = directory / 'head_owners.json'
        if target.exists():
            return result
        optimizer = kwargs['optimizer']
        contract = ownership(kwargs['model'], optimizer)
        saved_groups = optimizer.state_dict()['param_groups']
        mapping = serialized_owners(saved_groups, contract)
        checkpoint = directory / f'model_step_{state.global_step:06d}.pt'
        atomic_json(target, dict(ownership=contract, owner_sha256=digest(contract),
            serialized_id_names=mapping, optimizer_options=group_options(optimizer),
            checkpoint_sha256=sha256(checkpoint), step=int(state.global_step)))
        return result
