"""Capture a corrected FP32 zero-update gradient probe and its forward evidence."""

import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.consistent_precision import FullPrecisionContractMixin, require_fp32
from script.a3.gradient_probe import GradientProbeTrainer
from script.a3.zhy_likelihood_trainer import snapshot


def buffers(model):
    return {name: snapshot(value) for name, value in model.named_buffers()}


class ZHYGradientTrainer(FullPrecisionContractMixin, GradientProbeTrainer):
    """Reuse the official objective and forbidden-step hook; save CPU evidence."""

    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        model = self.accelerator.unwrap_model(self.model)
        self._gradient_loaded_buffers = buffers(model)
        self._gradient_parameter_names = [name for name, _ in model.named_parameters()]
        self._gradient_forwards = []
        return result

    def _get_rollout_data(self, *args, **kwargs):
        rollout = super()._get_rollout_data(*args, **kwargs)
        self._gradient_rollout_buffers = buffers(self.accelerator.unwrap_model(self.model))
        return rollout

    def _forward_model(self, model, mb_rollout_data):
        require_fp32(self.accelerator, self.args)
        live = self.accelerator.unwrap_model(model)
        before = buffers(live)
        results = super()._forward_model(model, mb_rollout_data)
        policy = results['policy_results']
        self._gradient_forwards.append(dict(
            indices=snapshot(mb_rollout_data['micro_batch_inds']),
            policy={key: snapshot(policy[key]) for key in (
                'action_mean', 'action_std', 'logprobs', 'entropy', 'aux_losses', 'aux_loss_coef')},
            value=snapshot(results['value_results']),
            buffers_before=before, buffers_after=buffers(live),
            precision=require_fp32(self.accelerator, self.args)))
        return results

    def _compute_loss(self, forward_results, mb_rollout_data):
        try:
            return super()._compute_loss(forward_results, mb_rollout_data)
        except SystemExit as exc:
            if exc.code != 0:
                raise
            output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
            if len(self._gradient_forwards) != 4 or self._optimizer_step_ordinal:
                raise ValueError('Requires four zero-update gradient forwards')
            model = self.accelerator.unwrap_model(self.model)
            path = output / 'forward_evidence.pt'
            torch.save(dict(
                forwards=self._gradient_forwards,
                parameter_names=self._gradient_parameter_names,
                loaded_buffers=self._gradient_loaded_buffers,
                rollout_buffers=self._gradient_rollout_buffers,
                final_buffers=buffers(model),
                objective=dict(cliprange=float(self.args.cliprange),
                    cliprange_value=float(self.args.cliprange_value),
                    vf_coef=float(self.args.vf_coef), entropy_coef=float(self.entropy_coef),
                    ppo_loss_coef=float(self.config.get('ppo_loss_coef', 1.0)),
                    aux_loss_scale=float(self.aux_loss_scale),
                    multi_critic_advantage_weights=snapshot(self.multi_critic_advantage_weights)),
                actual_actor_normalizer=model.policy.running_mean_std is not None,
                actual_critic_normalizer=model.value_model.running_mean_std is not None), path)
            self._save_precision_contract()
            atomic_json(output / 'forward_evidence.json', dict(
                sha256=sha256(path), bytes=path.stat().st_size,
                optimizer_steps=0, forward_count=4,
                scope='ACTUAL_FP32_FORWARDS_AND_BUFFERS_NOT_POLICY_ACCEPTANCE'))
            raise
