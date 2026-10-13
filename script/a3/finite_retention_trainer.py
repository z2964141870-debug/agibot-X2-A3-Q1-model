"""Fresh E51 training: finite original-anchor acceptance before any live commit."""

import copy
import json
import math
import os
import time

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_cached_tokens import cached_distribution
from script.a3.cloth_curve_trainer import ClothCurveTrainer
from script.a3.finite_adam import FiniteAdamW
from script.a3.finite_retention_protocol import CHECKPOINTS, UPDATES, STEPS
from script.a3.retention_direction_probe import ActorLossHarness, tree_digest
from script.a3.resumable_sonic import DurableModelSaveCallback


class FunctionalCandidate(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, kind, data):
        if kind == 'teacher':
            return self.actor.forward(data, is_training=False)
        self.actor.forward(data['mb_obs_dict'], is_training=True,
                           episode_attnmask=data['episode_attnmask'])
        aux, coefs = self.actor.aux_losses, self.actor.aux_loss_coef
        result = cached_distribution(self.actor, data['mb_obs_dict'],
            data['mb_sampled_tokens'], data['mb_actions'])
        result.update(aux_losses=aux, aux_loss_coef=coefs)
        return result


class FiniteRetentionTrainer(ClothCurveTrainer):
    def create_optimizer(self):
        super().create_optimizer()
        old = self.optimizer
        if isinstance(old, FiniteAdamW):
            return old
        if type(old) is not torch.optim.AdamW or old.state:
            raise ValueError('Requires a fresh official warm start with native empty AdamW')
        groups = [dict(g, params=list(g['params'])) for g in old.param_groups]
        self.optimizer = FiniteAdamW(groups, evaluate=self._evaluate_candidate,
                                    record=self._record_candidate)
        return self.optimizer

    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        wrapper = self.accelerator.unwrap_model(self.model)
        if self.args.max_grad_norm != .1 or self.use_symmetry or self.compute_imgaug_bc_loss:
            raise ValueError('Unexpected clipping or loss mode')
        if (self.args.num_micro_batches != 1 or self.args.num_mini_batches != 4 or
                self.args.num_ppo_epochs != 1):
            raise ValueError('Registered one-epoch four-batch transaction cadence changed')
        if any(isinstance(m, torch.nn.modules.dropout._DropoutNd) and m.p for m in wrapper.modules()):
            raise ValueError('Stochastic dropout is outside the registered candidate contract')
        self._finite_names = {id(p): n for n, p in wrapper.named_parameters() if p.requires_grad}
        self._finite_proposals = 0
        self._finite_evaluations = 0
        self._finite_current_evaluations = 0
        self._finite_rows = []
        self._finite_pending_batch = False
        self._finite_last_committed = None
        return result

    def _forward_model(self, model, batch):
        if self._finite_pending_batch:
            raise ValueError('Previous batch did not commit')
        wrapper = self.accelerator.unwrap_model(self.model)
        self._finite_before_buffers = {n: v.detach().clone() for n, v in wrapper.named_buffers()}
        self._finite_pending_batch = True
        return super()._forward_model(model, batch)

    def _compute_loss(self, forward_results, batch):
        result = super()._compute_loss(forward_results, batch)
        self._finite_batch = batch
        self._finite_value = forward_results['value_results'].detach().clone()
        self._finite_original_loss = float(result['loss'].detach())
        self._finite_harness = ActorLossHarness(copy.deepcopy(self.args), self.config)
        return result

    def _gradient_clipping(self):
        norm = super()._gradient_clipping()
        if norm is None or not math.isfinite(float(norm)):
            raise ValueError('Nonfinite gradient: stop, do not silently skip a batch')
        self._finite_gradient_norm = float(norm)
        return norm

    def _before_optimizer_step(self, optimizer, args, kwargs):
        super()._before_optimizer_step(optimizer, args, kwargs)
        if not self._finite_pending_batch or self._finite_proposals >= STEPS:
            raise ValueError('Unexpected or excess finite optimizer proposal')
        self._finite_proposals += 1
        self._finite_current_evaluations = 0
        self._finite_before_state = tree_digest((
            [p for g in optimizer.param_groups for p in g['params']], optimizer.state_dict()))
        self._finite_optimizer = optimizer

    @torch.no_grad()
    def _evaluate_candidate(self, live, values):
        # 1 baseline plus at most 8 candidates. No new batch, gradient or LR update.
        self._finite_current_evaluations += 1
        self._finite_evaluations += 1
        if self._finite_current_evaluations > 9 or self._finite_evaluations > STEPS * 9:
            raise ValueError('Finite evaluation budget exhausted')
        wrapper = self.accelerator.unwrap_model(self.model)
        policy = wrapper.policy
        if set(map(id, live)) != set(self._finite_names):
            raise ValueError('Optimizer parameters do not cover the exact full trainable model')
        mapping = {'actor.' + self._finite_names[id(p)].removeprefix('policy.'): v
                   for p, v in zip(live, values) if self._finite_names[id(p)].startswith('policy.')}
        module = FunctionalCandidate(policy)
        buffer_before = tree_digest(dict(wrapper.named_buffers()))
        with torch.random.fork_rng(devices=[self.accelerator.device.index or 0]):
            output = torch.func.functional_call(module, mapping, ('cloth', self._finite_batch))
            losses = self._finite_harness._compute_loss(
                dict(policy_results=output, value_results=self._finite_value), self._finite_batch)
            teachers, maxima = [], []
            for start in range(0, 16, 4):
                obs = {n: v[start:start+4] for n, v in self._cloth_obs.items()}
                mean = torch.func.functional_call(module, mapping, ('teacher', obs))
                error = mean - self._cloth_targets[start:start+4]
                teachers.append(error.square().mean())
                maxima.append(float(error.abs().max()))
            objective = float(losses['loss'] + teachers[self._optimizer_step_ordinal % 4])
            mse = float(torch.stack(teachers).mean())
        if tree_digest(dict(wrapper.named_buffers())) != buffer_before:
            raise ValueError('Candidate validation mutated model buffers')
        if self._finite_current_evaluations == 1 and not math.isclose(
                objective, self._finite_original_loss, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('No-update objective differs from original full loss')
        return dict(objective=objective, teacher_mse=mse, teacher_rmse=math.sqrt(mse),
                    teacher_max_abs=max(maxima))

    def _record_candidate(self, row):
        row = dict(row, proposal=self._finite_proposals, global_step=int(self.state.global_step),
            nominal_group_lrs=list(self._pending_step_lrs), full_gradient_norm=self._finite_gradient_norm,
            evaluations=self._finite_current_evaluations, at=time.time())
        optimizer = self._finite_optimizer
        now = tree_digest(([p for g in optimizer.param_groups for p in g['params']], optimizer.state_dict()))
        row['parameters_and_optimizer_before_sha256'] = self._finite_before_state
        row['parameters_and_optimizer_now_sha256'] = now
        if row['status'] != 'COMMITTED' and now != self._finite_before_state:
            raise ValueError('Speculative proposal leaked into live state')
        path = self._cloth_directory / 'finite_candidates.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        if row['status'] == 'COMMITTED':
            self._finite_last_committed = row

    def _after_optimizer_step(self, optimizer, args, kwargs):
        if not self._finite_last_committed or self._finite_last_committed['proposal'] != self._finite_proposals:
            raise ValueError('Post-step hook without an accepted transaction')
        # Retain the original .05/.5 check after commit as a second guard.
        super()._after_optimizer_step(optimizer, args, kwargs)
        actual = self._cloth_records[-1]
        expected = self._finite_last_committed['candidates'][-1]
        if not math.isclose(actual['teacher_action_rmse'], expected['teacher_rmse'], abs_tol=1e-7, rel_tol=1e-6):
            raise ValueError('Committed teacher differs from accepted functional candidate')
        self._finite_rows.append(self._finite_last_committed)
        atomic_json(self._cloth_directory / 'finite_contract.json', dict(
            proposals=self._finite_proposals, committed=len(self._finite_rows),
            evaluations=self._finite_evaluations, rows=self._finite_rows,
            original_absolute_teacher_guard=True, e50_nonincrease_gate_not_used=True,
            same_batch_objective_tolerance=1e-7, checkpoint_effect_not_proven=True))
        self._finite_pending_batch = False

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except BaseException:
            # Norm update belongs to the uncommitted batch; remove it on rejection.
            if getattr(self, '_finite_pending_batch', False):
                wrapper = self.accelerator.unwrap_model(self.model)
                committed = (self._finite_last_committed is not None and
                             self._finite_last_committed['proposal'] == self._finite_proposals)
                if not committed:
                    with torch.no_grad():
                        for name, value in wrapper.named_buffers():
                            value.copy_(self._finite_before_buffers[name])
                atomic_json(self._cloth_directory / 'finite_stop.json', dict(
                    at=time.time(), proposals=self._finite_proposals,
                    committed=self._optimizer_step_ordinal,
                    committed_before_failure=committed,
                    uncommitted_batch_buffers_restored=not committed,
                    no_restart_or_evaluation=True))
            raise


class FiniteCheckpointCallback(DurableModelSaveCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target_global_step != UPDATES:
            raise ValueError('Registered E51 update budget changed')

    def _save(self, args, state, **kwargs):
        directory = self.save_dir
        self.save_dir = directory / ('milestone_' + str(state.global_step))
        try:
            return super()._save(args, state, **kwargs)
        finally:
            self.save_dir = directory

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in CHECKPOINTS:
            self._save(args, state, **kwargs)
        if state.global_step >= UPDATES:
            control.should_training_stop = True
        return control
