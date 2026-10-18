"""D02 changes only regularizer examples; original384 remains the hard guard."""

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.cloth_constrained_trainer import initial_probability
from script.a3.consistent_precision import ConsistentPrecisionTrainer, require_fp32
from script.a3.finite_retention_trainer import FunctionalCandidate
from script.a3.humanplus_head_trainer import HeadOnlyTrainer, HeadCheckpointCallback
from script.a3.humanplus_replay_protocol import COMMAND, INPUT_ORDER, ORIENTATION, slices, validate_selection
from script.a3.retention_direction_probe import ActorLossHarness, tree_digest
from script.a3.zhy_likelihood_trainer import snapshot


def native_observations(policy, flat):
    module = policy.actor_module
    layout = slices(module.tokenizer_obs_specs, module.tokenizer_obs_total_dim,
                    module.encoder_sample_probs, module.proprioception_features)
    if flat.shape != (16, 24, 1570) or flat.dtype != torch.float32 or not torch.isfinite(flat).all():
        raise ValueError('Expected finite FP32 original384 replay observations')
    tokenizer = flat.new_zeros(16, 24, module.tokenizer_obs_total_dim)
    for name, value in ((COMMAND, flat[..., :580]), (ORIENTATION, flat[..., 580:640])):
        start, end, _ = layout[name]
        tokenizer[..., start:end] = value
    start, end, _ = layout['encoder_index']
    tokenizer[..., start + 3] = 1
    obs = dict(actor_obs=flat[..., 640:], tokenizer=tokenizer)
    parsed = module.parse_tokenizer_obs(obs)
    if not torch.equal(parsed['encoder_index'], flat.new_tensor([0, 0, 0, 1]).expand(16, 24, 4)):
        raise ValueError('Replay examples must use only the native a3_fast route')
    if (not torch.equal(parsed[COMMAND].reshape(16, 24, 580), flat[..., :580])
            or not torch.equal(parsed[ORIENTATION].reshape(16, 24, 60), flat[..., 580:640])):
        raise ValueError('Replay native feature round trip changed values')
    return obs


class ReplayLossTrainer(HeadOnlyTrainer):
    def load_checkpoint(self, checkpoint_path, resume=False):
        result = super().load_checkpoint(checkpoint_path, resume=resume)
        path = Path(os.environ['YUANQI_REPLAY_DESIGN'])
        design = json.loads(path.read_text())
        selected = validate_selection(design)
        root = Path(os.environ['YUANQI_REPLAY_ROOT'])
        observations, targets, identities = [], [], []
        for row in selected:
            source = root / row['source']
            identity = design['read_only_sources'][row['source']]
            if source.stat().st_size != identity['bytes'] or sha256(source) != identity['sha256']:
                raise ValueError('Replay data identity changed')
            with np.load(source, allow_pickle=False) as data:
                x, y = data['observations'], data['actions']
                n = row['full_policy_steps']
                if (x.shape != (n, 1570) or y.shape != (n, 29)
                        or x.dtype != np.float32 or y.dtype != np.float32
                        or str(data['input_order']) != INPUT_ORDER
                        or not np.isfinite(x).all() or not np.isfinite(y).all()):
                    raise ValueError('Original preclip replay data ABI changed')
                observations.append(x[row['selected_policy_ticks']].copy())
                targets.append(y[row['selected_policy_ticks']].copy())
            identities.append(dict(row, **identity))
        device = self.accelerator.device
        flat = torch.from_numpy(np.concatenate(observations)).reshape(16, 24, 1570).to(device)
        self._replay_targets = torch.from_numpy(np.concatenate(targets)).reshape(16, 24, 29).to(device)
        policy = self.accelerator.unwrap_model(self.model).policy
        self._replay_obs = native_observations(policy, flat)
        before = tree_digest(dict(policy.named_buffers()))
        with torch.no_grad(), torch.random.fork_rng(devices=[device.index or 0]):
            means = torch.cat([policy.forward(self._replay_quarter(i), is_training=False) for i in range(4)])
        difference = float((means - self._replay_targets).abs().max())
        if not math.isfinite(difference) or difference > 1e-5:
            raise ValueError('Fresh035 does not reconstruct original preclip replay labels')
        if tree_digest(dict(policy.named_buffers())) != before:
            raise ValueError('Replay initialization mutated Actor buffers')
        self._check_head()
        self._replay_contract = dict(design_sha256=sha256(path), sources=identities,
            fresh035_label_max_abs=difference, label_tolerance=1e-5, frames=384,
            input_order=INPUT_ORDER, roles='DEVELOPMENT_REGULARIZATION_NOT_HELD_OUT',
            legacy_hard_guard_unchanged=True, coefficient=1., initial_calls=4,
            native_route=[0, 0, 0, 1], preclip_labels=True, batches=[])
        self._save_replay()
        return result

    def _replay_quarter(self, index):
        start = index * 4
        return {n: v[start:start + 4] for n, v in self._replay_obs.items()}

    def _save_replay(self):
        atomic_json(self._cloth_directory / 'replay_loss_contract.json', self._replay_contract)

    def _compute_loss(self, forward_results, batch):
        require_fp32(self.accelerator, self.args)
        if self._cloth_initial_probability is None:
            if self.state.global_step or self._optimizer_step_ordinal:
                raise ValueError('Initial likelihood check must precede all updates')
            self._cloth_initial_probability = initial_probability(forward_results['policy_results'], batch)
            torch.save(dict(policy=snapshot(forward_results['policy_results']), batch=snapshot(batch)),
                       self._cloth_directory / 'initial_probability.pt')
            self._save_cloth_receipt()
        # Use the original base PPO loss, replacing (not adding to) Cloth's regularizer.
        result = ConsistentPrecisionTrainer._compute_loss(self, forward_results, batch)
        policy = self.accelerator.unwrap_model(self.model).policy
        quarter = self._optimizer_step_ordinal % 4
        mean = policy.forward(self._replay_quarter(quarter), is_training=False)
        target = self._replay_targets[quarter * 4:quarter * 4 + 4]
        retention = (mean - target).square().mean()
        if not torch.isfinite(retention):
            raise ValueError('Nonfinite replay regularizer')
        result['loss'] = result['loss'] + retention
        self._cloth_last_retention_loss = float(retention.detach())
        self._replay_contract['batches'].append(dict(commit_before=self._optimizer_step_ordinal,
            quarter=quarter, frames=96, loss=self._cloth_last_retention_loss))
        self._save_replay()
        self._finite_batch = batch
        self._finite_value = forward_results['value_results'].detach().clone()
        self._finite_original_loss = float(result['loss'].detach())
        self._finite_harness = ActorLossHarness(copy.deepcopy(self.args), self.config)
        return result

    @torch.no_grad()
    def _evaluate_candidate(self, live, values):
        self._finite_current_evaluations += 1
        self._finite_evaluations += 1
        if self._finite_current_evaluations > 9 or self._finite_evaluations > 288:
            raise ValueError('Fixed finite evaluation budget exhausted')
        wrapper = self._check_head()
        if set(map(id, live)) != set(self._finite_names):
            raise ValueError('Finite optimizer ownership differs from registered trainable model')
        mapping = {'actor.' + self._finite_names[id(p)].removeprefix('policy.'): v
                   for p, v in zip(live, values) if self._finite_names[id(p)].startswith('policy.')}
        module = FunctionalCandidate(wrapper.policy)
        before = tree_digest(dict(wrapper.named_buffers()))
        with torch.random.fork_rng(devices=[self.accelerator.device.index or 0]):
            output = torch.func.functional_call(module, mapping, ('cloth', self._finite_batch))
            losses = self._finite_harness._compute_loss(
                dict(policy_results=output, value_results=self._finite_value), self._finite_batch)
            teachers, maxima = [], []
            for start in range(0, 16, 4):
                obs = {n: v[start:start + 4] for n, v in self._cloth_obs.items()}
                mean = torch.func.functional_call(module, mapping, ('teacher', obs))
                error = mean - self._cloth_targets[start:start + 4]
                teachers.append(error.square().mean())
                maxima.append(float(error.abs().max()))
            quarter = self._optimizer_step_ordinal % 4
            mean = torch.func.functional_call(module, mapping, ('teacher', self._replay_quarter(quarter)))
            retention = (mean - self._replay_targets[quarter * 4:quarter * 4 + 4]).square().mean()
            objective = float(losses['loss'] + retention)
            mse = float(torch.stack(teachers).mean())
        if tree_digest(dict(wrapper.named_buffers())) != before:
            raise ValueError('Functional candidate mutated model buffers')
        if self._finite_current_evaluations == 1 and not math.isclose(
                objective, self._finite_original_loss, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Live candidate objective differs from actual D02 training loss')
        return dict(objective=objective, teacher_mse=mse, teacher_rmse=math.sqrt(mse),
                    teacher_max_abs=max(maxima), replay_retention_loss=float(retention),
                    replay_quarter=quarter)


class ReplayCheckpointCallback(HeadCheckpointCallback):
    pass
