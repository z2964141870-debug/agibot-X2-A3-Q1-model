"""Verify the precision fix on one real and three disjoint initial batches."""

import os
from pathlib import Path

import torch

from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.consistent_precision import FullPrecisionContractMixin
from script.a3.zhy_entry_trainer import ZHYEntryTrainer
from script.a3.zhy_likelihood_trainer import snapshot
from script.a3.zhy_precision_trainer import validate_initial_batches


class ZHYConsistencyProbeTrainer(FullPrecisionContractMixin, ZHYEntryTrainer):
    def _compute_loss(self, forward_results, mb_rollout_data):
        first = mb_rollout_data['micro_batch_inds']
        initial = forward_results['policy_results']
        records = [dict(indices=snapshot(first), mean=snapshot(initial['action_mean']),
                        std=snapshot(initial['action_std']))]
        remaining = [i for i in range(16) if i not in first.tolist()]
        with torch.no_grad():
            for start in range(0, len(remaining), 4):
                indices = torch.tensor(remaining[start:start+4], device=first.device)
                extra = self._get_mb_rollout_data(self._probe_rollout, indices)
                result = TRLPPOTrainer._forward_model(self, self.model, extra)['policy_results']
                records.append(dict(indices=snapshot(indices), mean=snapshot(result['action_mean']),
                                    std=snapshot(result['action_std'])))
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        path = output / 'consistency_batches.pt'
        torch.save(records, path)
        atomic_json(output / 'consistency_batches.json', dict(
            rows=validate_initial_batches(records, snapshot(self._probe_rollout)),
            capture_sha256=sha256(path), actor_precision_receipts=self._precision_receipts,
            actor_precision_calls=dict(self._precision_calls),
            coverage='ONE_REAL_PLUS_THREE_DISJOINT_BATCHES_NOT_ORIGINAL_RANDOM_ORDER'))
        self._save_precision_contract()
        return super()._compute_loss(forward_results, mb_rollout_data)

