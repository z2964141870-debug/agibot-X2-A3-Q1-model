"""Zero-update control with matching full precision at sampling and learning."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.gradient_probe import cpu_tree
from script.a3.likelihood_probe import compare
from script.a3.zhy_likelihood_trainer import ZHYRuntimeLikelihoodTrainer, snapshot


def validate_initial_batches(records, source):
    coverage = [i for r in records for i in r['indices'].tolist()]
    if len(records) != 4 or any(len(r['indices']) != 4 for r in records):
        raise ValueError('Require four four-environment minibatches')
    if sorted(coverage) != list(range(16)):
        raise ValueError('Minibatches must cover each environment exactly once')
    rows = []
    for index, record in enumerate(records):
        inds = record['indices']
        expected = source['old_mu_batch'][inds]
        if record['mean'].shape != expected.shape or record['std'].shape != expected.shape:
            raise ValueError('Minibatch distribution shape mismatch')
        rows.append(dict(batch=index, indices=inds.tolist(),
            origin='ACTUAL_FIRST_BATCH' if index == 0 else 'EXTRA_DISJOINT_BATCH',
            **compare(record['mean'], expected, record['std'], source['old_sigma_batch'][inds],
                source['actions'][inds], source['logprobs'][inds])))
    return rows


class ZHYFullPrecisionTrainer(ZHYRuntimeLikelihoodTrainer):
    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        before = dict(matmul=torch.backends.cuda.matmul.allow_tf32,
                      cudnn=torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        atomic_json(Path(os.environ['YUANQI_PROBE_OUTPUT']) / 'precision_override.json',
            dict(before=before, sampling_and_training_tf32=False,
                 scope='SINGLE_NUMERICAL_VARIABLE_NO_PARAMETER_UPDATE'))
        return result

    def _compute_loss(self, forward_results, mb_rollout_data):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('Sampling and first training forward must both disable TF32')
        first = mb_rollout_data['micro_batch_inds']
        records = [dict(indices=snapshot(first),
            mean=snapshot(forward_results['policy_results']['action_mean']),
            std=snapshot(forward_results['policy_results']['action_std']))]
        remaining = [i for i in range(16) if i not in first.tolist()]
        with torch.no_grad():
            for start in range(0, len(remaining), 4):
                indices = torch.tensor(remaining[start:start+4], device=first.device)
                minibatch = self._get_mb_rollout_data(self._probe_rollout, indices)
                result = self._forward_model(self.model, minibatch)['policy_results']
                records.append(dict(indices=snapshot(indices), mean=snapshot(result['action_mean']),
                    std=snapshot(result['action_std'])))
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        capture = output / 'initial_batches.pt'
        torch.save(records, capture)
        atomic_json(output / 'initial_batches.json', dict(rows=validate_initial_batches(
            records, cpu_tree(self._probe_rollout)), capture_sha256=sha256(capture),
            coverage='FIRST_ACTUAL_PLUS_THREE_EXTRA_DISJOINT_BATCHES_NOT_ORIGINAL_RANDOM_ORDER'))
        return super()._compute_loss(forward_results, mb_rollout_data)
