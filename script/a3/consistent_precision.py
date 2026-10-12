"""Explicit FP32 contract for new A3 trials, checked inside the Actor."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.verified_finetune import VerifiedFineTuneTrainer


def precision_overrides():
    return ['++algo.trl.bf16=false', '++algo.trl.fp16=false', '++algo.trl.tf32=false']


def precision_state(accelerator, args):
    return dict(mixed_precision=accelerator.mixed_precision,
        native_amp=accelerator.native_amp, bf16=args.bf16, fp16=args.fp16,
        cuda_autocast=torch.is_autocast_enabled('cuda'),
        cpu_autocast=torch.is_autocast_enabled('cpu'),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32)


def require_fp32(accelerator, args):
    state = precision_state(accelerator, args)
    if state['mixed_precision'] != 'no' or any(
            value for key,value in state.items() if key != 'mixed_precision'):
        raise ValueError('A3 FP32 contract violated: ' + repr(state))
    return state


class FullPrecisionContractMixin:
    def __init__(self, *args, **kwargs):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(*args, **kwargs)
        initial = require_fp32(self.accelerator, self.args)
        self._precision_receipts = []
        self._precision_calls = dict(rollout=0, training=0)
        self._precision_contract = dict(initial=initial, overrides=precision_overrides(),
                                        checked_at_actual_actor_module=True)
        policy = self.accelerator.unwrap_model(self.model).policy
        self._precision_hook = policy.actor_module.register_forward_pre_hook(
            self._check_actor_precision)
        self._save_precision_contract()

    def _check_actor_precision(self, module, arguments):
        state = require_fp32(self.accelerator, self.args)
        obs = arguments[0]
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in obs.values()):
            raise ValueError('Actor observation is not float32')
        self._precision_calls['training' if module.training else 'rollout'] += 1
        if not any(row['training'] == module.training for row in self._precision_receipts):
            self._precision_receipts.append(dict(training=module.training, **state))

    def _save_precision_contract(self):
        atomic_json(Path(os.environ['YUANQI_RESUME_LOG_DIR']) / 'precision_contract.json',
            dict(self._precision_contract, actor_calls=dict(self._precision_calls),
                 first_call_per_mode=list(self._precision_receipts)))

    def load_checkpoint(self, *args, **kwargs):
        require_fp32(self.accelerator, self.args)
        return super().load_checkpoint(*args, **kwargs)

    def _get_rollout_data(self, *args, **kwargs):
        require_fp32(self.accelerator, self.args)
        result = super()._get_rollout_data(*args, **kwargs)
        self._save_precision_contract()
        return result

    def append_to_log_dict(self, log_dict):
        super().append_to_log_dict(log_dict)
        self._save_precision_contract()


class ConsistentPrecisionTrainer(FullPrecisionContractMixin, VerifiedFineTuneTrainer):
    """New opt-in training entry; rejects silently re-enabled AMP."""

