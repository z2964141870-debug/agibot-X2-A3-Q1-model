"""Capture inputs at sampling, before environment buffers can change."""

import os
from pathlib import Path
from collections.abc import Mapping

import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.gradient_probe import parameter_digest
from script.a3.runtime_likelihood_probe import RuntimeLikelihoodTrainer


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, Mapping):
        return {key:snapshot(v) for key,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    return value


def sampling_identity(records, rollout):
    mean = torch.cat([r['mean'][:, -1:] for r in records], dim=1)
    if mean.shape != rollout['old_mu_batch'].shape:
        raise ValueError('Sampling capture does not match rollout coverage')
    deltas = {}
    for key in rollout['all_obs_dict']:
        sampled = torch.cat([r['inputs'][key][:, -1:] for r in records], dim=1)
        saved = rollout['all_obs_dict'][key].detach().cpu()
        if sampled.shape != saved.shape:
            raise ValueError('Sampling observation dimensions changed')
        deltas[key] = dict(max_abs_difference=float((sampled-saved).abs().max()),
            exact=torch.equal(sampled, saved), shape=list(saved.shape))
    old = rollout['old_mu_batch'].detach().cpu()
    return dict(sampling_calls=len(records), all_inputs_exact=all(r['exact'] for r in deltas.values()),
        inputs=deltas, sampled_means_match_storage=torch.equal(mean, old),
        mean_max_abs_difference=float((mean-old).abs().max()),
        scope='FIRST_FRESH_ROLLOUT_SAMPLING_INPUTS_COMPARED_WITH_TRAINING_STORAGE')


class ZHYRuntimeLikelihoodTrainer(RuntimeLikelihoodTrainer):
    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        policy = self.accelerator.unwrap_model(self.model).policy
        self._sampling_records = []
        self._sampling_forward = policy.forward
        def capture(obs_dict, *arguments, **options):
            inputs = snapshot(obs_dict)
            mean = self._sampling_forward(obs_dict, *arguments, **options)
            self._sampling_records.append(dict(inputs=inputs, mean=snapshot(mean),
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, training=policy.training))
            return mean
        policy.forward = capture
        return result

    def _get_rollout_data(self, *args, **kwargs):
        policy = self.accelerator.unwrap_model(self.model).policy
        policy.forward = self._sampling_forward
        rollout = super()._get_rollout_data(*args, **kwargs)
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        path = output / 'sampled_inputs.pt'
        torch.save(self._sampling_records, path)
        identity = sampling_identity(self._sampling_records, rollout)
        identity.update(capture_sha256=sha256(path), capture_bytes=path.stat().st_size)
        atomic_json(output / 'sampling_identity.json', identity)
        return rollout

    def _compute_loss(self, forward_results, mb_rollout_data):
        model = self.accelerator.unwrap_model(self.model)
        named = list(model.named_parameters())
        changed = [name for name,p in named if parameter_digest([(name,p)]) != self._loaded_hashes[name]]
        clamped = self._loaded_std.clamp(model.policy.algo_config.std_clamp_min,
                                         model.policy.algo_config.std_clamp_max)
        if changed and (changed != ['policy.std'] or not torch.equal(clamped, model.policy.std)):
            raise ValueError('Unexpected parameter changes before diagnosis')
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        minibatch = output / 'actual_first_minibatch.pt'
        torch.save(dict(indices=snapshot(mb_rollout_data['micro_batch_inds']),
            mean=snapshot(forward_results['policy_results']['action_mean']),
            std=snapshot(forward_results['policy_results']['action_std'])), minibatch)
        atomic_json(Path(os.environ['YUANQI_PROBE_OUTPUT']) / 'runtime_precision_before.json',
            dict(cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32, policy_training=model.policy.training,
                policy_running_mean_std=model.policy.running_mean_std is not None,
                parameter_changes=changed, sampling_precision=[{k:r[k] for k in
                    ('cuda_matmul_tf32', 'cudnn_tf32', 'training')} for r in self._sampling_records],
                actual_first_minibatch_sha256=sha256(minibatch),
                diagnostic_cpu_capture_changes_timing=True))
        return super()._compute_loss(forward_results, mb_rollout_data)
