"""Capture the first clothing PPO batch before any validation or optimizer step."""

import os
from pathlib import Path

from omegaconf import OmegaConf
import torch

from script.a3.checkpoint_store import atomic_json, sha256, fsync_directory
from script.a3.cloth_constrained_trainer import ClothConstrainedTrainer
from script.a3.gradient_probe import parameter_digest
from script.a3.zhy_likelihood_trainer import snapshot, sampling_identity


class CapturedBatch(Exception):
    pass


def save_capture(value, path):
    with path.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def encoded(module):
    return dict(tokens=snapshot(module._last_encoded_tokens),
                latents=snapshot(module._last_encoded_latents))


class ClothLikelihoodProbeTrainer(ClothConstrainedTrainer):
    def _before_optimizer_step(self, *args, **kwargs):
        raise RuntimeError('E39 prohibits every optimizer.step')

    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        policy = self.accelerator.unwrap_model(self.model).policy
        self._capture_dir = Path(os.environ['YUANQI_RESUME_LOG_DIR'])
        self._capture_sampling = []
        self._capture_forward = policy.forward
        self._capture_digest = parameter_digest(list(policy.named_parameters()))
        policy.actor_module.cache_encoded_cpu = True

        def capture(obs, *arguments, **options):
            before = snapshot(obs)
            mean = self._capture_forward(obs, *arguments, **options)
            self._capture_sampling.append(dict(inputs=before, mean=snapshot(mean),
                                               encoded=encoded(policy.actor_module)))
            return mean

        policy.forward = capture
        return result

    def _get_rollout_data(self, *args, **kwargs):
        policy = self.accelerator.unwrap_model(self.model).policy
        policy.forward = self._capture_forward
        rollout = super()._get_rollout_data(*args, **kwargs)
        self._capture_rollout = snapshot(rollout)
        # Save the raw evidence before a diagnostic helper can reject it.
        save_capture(dict(sampling=self._capture_sampling, rollout=self._capture_rollout),
                     self._capture_dir / 'sampling.pt')
        identity = sampling_identity(self._capture_sampling, self._capture_rollout)
        atomic_json(self._capture_dir / 'sampling_identity.json', identity)
        self._capture_training_inputs = None
        def before_actor(module, arguments):
            self._capture_training_inputs = snapshot(arguments[0])
        self._capture_input_hook = policy.actor_module.register_forward_pre_hook(before_actor)
        return rollout

    def _compute_loss(self, forward_results, batch):
        if self.state.global_step or self._optimizer_step_ordinal or self.optimizer.state:
            raise ValueError('E39 was not intercepted before updates')
        policy = self.accelerator.unwrap_model(self.model).policy
        actual = dict(policy=snapshot(forward_results['policy_results']), batch=snapshot(batch),
            actor_inputs=self._capture_training_inputs, encoded=encoded(policy.actor_module))
        path = self._capture_dir / 'first_batch.pt'
        save_capture(actual, path)
        self._save_precision_contract()
        atomic_json(self._capture_dir / 'first_batch_saved.json', dict(
            sha256=sha256(path), bytes=path.stat().st_size, before_validation=True,
            optimizer_steps=0, global_step=0))
        self._capture_input_hook.remove()
        metadata = dict(env_config=OmegaConf.to_container(policy.env_config, resolve=True),
            algo_config=OmegaConf.to_container(policy.algo_config, resolve=True),
            cache_encoded_cpu_diagnostic=True, changes_action_computation=False,
            quantizer_class=type(policy.actor_module.quantizer).__module__ + '.' +
                            type(policy.actor_module.quantizer).__name__)
        atomic_json(self._capture_dir / 'runtime_metadata.json', metadata)
        indices = actual['batch']['micro_batch_inds']
        device = self.accelerator.device
        mini = {n: v.to(device) for n, v in actual['batch']['mb_obs_dict'].items()}
        full = {n: v.to(device) for n, v in self._capture_rollout['all_obs_dict'].items()}
        variants = {}
        with torch.no_grad():
            mean = policy.forward(mini, is_training=False)
            variants['mini_sequence_aux_false'] = dict(mean=snapshot(mean), encoded=encoded(policy.actor_module))
            for name, obs in (('mini_frames_aux_false', mini), ('full_frames_aux_false', full)):
                means, layers = [], []
                for tick in range(self._capture_rollout['actions'].shape[1]):
                    mean = policy.forward({n: v[:, tick:tick+1] for n, v in obs.items()}, is_training=False)
                    means.append(snapshot(mean))
                    layers.append(encoded(policy.actor_module))
                joined = torch.cat(means, dim=1)
                variants[name] = dict(mean=joined[indices] if name.startswith('full') else joined,
                                      frame_encoded=layers)
        save_capture(variants, self._capture_dir / 'variants.pt')
        self._assert_frozen()
        after = parameter_digest(list(policy.named_parameters()))
        if after != self._capture_digest:
            raise ValueError('E39 changed Actor parameters')
        self._save_precision_contract()
        atomic_json(self._capture_dir / 'capture_complete.json', dict(status='CAPTURED_ZERO_OPTIMIZER',
            optimizer_steps=0, global_step=0, parameters_before_sha256=self._capture_digest,
            parameters_after_sha256=after, source_frames=384, actual_batches=1,
            variant_calls=49, rollout_policy_steps=24, fresh_final_test=False,
            probability_contract_passed_not_required_for_capture=True,
            artifacts={n:dict(bytes=(self._capture_dir / n).stat().st_size,
                             sha256=sha256(self._capture_dir / n)) for n in
                ('sampling.pt', 'first_batch.pt', 'variants.pt', 'sampling_identity.json',
                 'runtime_metadata.json', 'precision_contract.json', 'first_batch_saved.json')},
            scope='DIAGNOSTIC_OBSERVER_CHANGES_TIMING_NOT_A_REPLAY_OF_E38_LEDGER'))
        raise CapturedBatch()

    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except CapturedBatch:
            if self.state.global_step or self._optimizer_step_ordinal or self.optimizer.state:
                raise ValueError('Zero-update capture contract violated')
            print('[E39] Captured first clothing batch and fixed-input controls; optimizer_steps=0', flush=True)
