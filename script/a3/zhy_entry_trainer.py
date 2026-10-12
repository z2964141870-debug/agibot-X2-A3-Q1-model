"""Capture the real first PPO actor call and replay it before any update."""

import os
from pathlib import Path

from omegaconf import OmegaConf
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.gradient_probe import parameter_digest
from script.a3.zhy_likelihood_trainer import snapshot
from script.a3.zhy_precision_trainer import ZHYFullPrecisionTrainer


def input_metadata(obs):
    return {key:dict(shape=list(value.shape), stride=list(value.stride()),
        dtype=str(value.dtype), device=str(value.device),
        contiguous=value.is_contiguous(), storage_offset=value.storage_offset())
        for key,value in obs.items()}


def actor_state(policy):
    module = policy.actor_module
    attributes = {}
    for name in ('encoders_to_iterate', '_active_decoders', 'rollout_decoders',
                 'rollout_encoder_exclusions', 'cache_encoded_cpu'):
        value = getattr(module, name, None)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        attributes[name] = snapshot(value)
    return dict(parameters=parameter_digest(list(policy.named_parameters())),
        buffers=parameter_digest(list(policy.named_buffers())),
        module_modes={name:part.training for name,part in policy.named_modules()},
        module_type=type(module).__module__ + '.' + type(module).__qualname__,
        attributes=attributes, actor_training=policy.training,
        actor_running_mean_std=policy.running_mean_std is not None)


class ZHYEntryTrainer(ZHYFullPrecisionTrainer):
    def _get_rollout_data(self, *args, **kwargs):
        rollout = super()._get_rollout_data(*args, **kwargs)
        policy = self.accelerator.unwrap_model(self.model).policy
        self._entry_sampling_actor = policy
        self._entry_rollout_actor_state = actor_state(policy)
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        atomic_json(output / 'runtime_model_config.json', dict(
            env_config=OmegaConf.to_container(policy.env_config, resolve=True),
            algo_config=OmegaConf.to_container(policy.algo_config, resolve=True)))
        return rollout

    def _forward_model(self, model, mb_rollout_data):
        if hasattr(self, '_entry_capture'):
            raise RuntimeError('Entry probe permits only the first training forward')
        policy = self.accelerator.unwrap_model(model).policy
        original = policy.forward
        calls = []
        live = {}
        before = actor_state(policy)
        batch_before = snapshot(mb_rollout_data['mb_obs_dict'])

        def traced(obs, *args, **kwargs):
            if calls:
                raise RuntimeError('Unexpected multiple Actor calls in first PPO forward')
            flags = dict(grad=torch.is_grad_enabled(),
                autocast=torch.is_autocast_enabled('cuda'),
                autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32)
            record = dict(inputs=snapshot(obs), args=snapshot(args), options=snapshot(kwargs),
                input_metadata=input_metadata(obs), flags=flags, state_before=actor_state(policy),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
            live.update(obs=obs, args=args, options=kwargs,
                        autocast_dtype=torch.get_autocast_dtype('cuda'))
            mean = original(obs, *args, **kwargs)
            record.update(mean=snapshot(mean), state_after=actor_state(policy))
            calls.append(record)
            return mean

        policy.forward = traced
        try:
            results = super()._forward_model(model, mb_rollout_data)
        finally:
            policy.forward = original
        if len(calls) != 1:
            raise RuntimeError('Actual Actor forward was not captured')
        entry = calls[0]
        indices = mb_rollout_data['micro_batch_inds']
        stored = {key:value[indices] for key,value in self._probe_rollout['all_obs_dict'].items()}
        distribution = results['policy_results']
        capture = dict(indices=snapshot(indices), entry=entry,
            wrapper_mean=snapshot(distribution['action_mean']),
            wrapper_std=snapshot(distribution['action_std']),
            wrapper_logprobs=snapshot(distribution['logprobs']),
            batch_input_before=batch_before,
            batch_input_after=snapshot(mb_rollout_data['mb_obs_dict']),
            live_input_after=snapshot(live['obs']),
            rollout=snapshot(self._probe_rollout), actor_is_sampling_actor=policy is self._entry_sampling_actor,
            rollout_actor_state=self._entry_rollout_actor_state,
            wrapper_actor_state_before=before, wrapper_actor_state_after=actor_state(policy),
            accelerator_mixed_precision=self.accelerator.mixed_precision,
            trainer_type=type(self).__module__ + '.' + type(self).__qualname__, replays={})
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        torch.save(capture, output / 'entry_before_replays.pt')

        # Restore the Actor-call RNG and precision for each immediate comparison.
        variants = (
            ('same_live_inputs_and_options', live['obs'], live['options']),
            ('stored_inputs_same_options', stored, live['options']),
            ('stored_inputs_without_episode_mask', stored,
             {key:value for key,value in live['options'].items() if key != 'episode_attnmask'}),
        )
        for label, obs, options in variants:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.set_rng_state(entry['cpu_rng'])
                torch.cuda.set_rng_state(entry['cuda_rng'])
                with torch.set_grad_enabled(entry['flags']['grad']), torch.autocast(
                    'cuda', enabled=entry['flags']['autocast'], dtype=live['autocast_dtype']):
                    mean = original(obs, *live['args'], **options)
                capture['replays'][label] = snapshot(mean)
                del mean
        capture['actor_state_after_replays'] = actor_state(policy)
        self._entry_capture = capture
        return results

    def _compute_loss(self, forward_results, mb_rollout_data):
        if self.state.global_step or self._optimizer_step_ordinal or self.optimizer.state:
            raise RuntimeError('Entry probe forbids optimizer updates')
        model = self.accelerator.unwrap_model(self.model)
        named = list(model.named_parameters())
        changed = [name for name,p in named if parameter_digest([(name,p)]) != self._loaded_hashes[name]]
        clamped = self._loaded_std.clamp(model.policy.algo_config.std_clamp_min,
                                         model.policy.algo_config.std_clamp_max)
        if changed and (changed != ['policy.std'] or not torch.equal(clamped, model.policy.std)):
            raise ValueError('Unexpected parameter changes before entry diagnosis')
        capture = self._entry_capture
        capture['parameter_changes_from_load'] = changed
        capture['mean_at_loss_entry'] = snapshot(forward_results['policy_results']['action_mean'])
        output = Path(os.environ['YUANQI_PROBE_OUTPUT'])
        path = output / 'entry_capture.pt'
        torch.save(capture, path)
        from script.a3.zhy_entry_probe import evaluate_capture
        result = evaluate_capture(capture)
        result.update(optimizer_steps=0, global_step=0, capture_sha256=sha256(path),
            capture_bytes=path.stat().st_size, backup_status='LOCAL_ONLY')
        atomic_json(output / 'entry_result.json', result)
        print('[YUANQI EntryProbe] first real Actor call captured; zero updates', flush=True)
        raise SystemExit(0)
