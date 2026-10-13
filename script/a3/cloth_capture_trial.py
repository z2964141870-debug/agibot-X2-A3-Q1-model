"""E39 independently captures clothing likelihood drift without optimization."""

import argparse
import os
from pathlib import Path
import time

from script.a3 import cloth_resource_trial as runtime
from script.a3.checkpoint_store import atomic_json, sha256

shared = runtime.shared
PRIOR, PRIOR_LOG, PRIOR_RUN = shared.OUT, shared.LOG, shared.RUN
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_capture_20261013_E39'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_capture_20261013_E39'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_capture_20261013_E39/R16_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_capture_20261013_E39.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_CAPTURE_20261013_E39'
shared.TRIAL = 'R16'
shared.MODULE = 'script.a3.cloth_capture_trial'
shared.TRAINER = 'script.a3.cloth_likelihood_probe.ClothLikelihoodProbeTrainer'
shared.CODE += ('script/a3/cloth_capture_trial.py', 'script/a3/cloth_likelihood_probe.py')
shared.STAGES = {'train': ('REGISTERED', 'CAPTURED'), 'verify-training': ('CAPTURED', 'COMPLETE')}


def register():
    state = shared.read(PRIOR / 'state.json')
    console = PRIOR_LOG / 'training/console.log'
    contract = shared.read(PRIOR_LOG / 'training/cloth_contract.json')
    if (state['status'] != 'STOPPED' or state['returncode'] != 1 or
            'Initial clothing likelihood contract failed' not in console.read_text() or
            contract['optimizer_steps'] or list(PRIOR_RUN.glob('model_step_00000[12].pt'))):
        raise ValueError('Requires preserved E38 first-batch zero-update rejection')
    old = shared.read(PRIOR / 'registration.json')
    for name, identity in {**old['code_sha256'], **old['input_sha256']}.items():
        if sha256(shared.ROOT / name) != identity:
            raise ValueError('E38 source/input changed before capture: ' + name)
    prepared = shared.read(PRIOR / 'prepare.json')
    for row in prepared['motions']:
        if sha256(shared.ROOT / row['path']) != row['sha256']:
            raise ValueError('E38 prepared motion changed')
    runtime.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment=old['experiment'], finalized_at=time.time(), updates=0,
        optimizer_steps=0, first_actual_batches=1, policy_rollout_steps=24,
        fixed_input_variant_calls=49, compute_loss_or_backward_allowed=False,
        main_variable='PERSISTENT_FIRST_BATCH_INPUT_LATENT_TOKEN_CAPTURE_ZERO_OPTIMIZER',
        prepared_motions_reused=True, encoder_cache_diagnostic=True)
    paths = [PRIOR / 'state.json', PRIOR / 'registration.json', PRIOR / 'prepare.json', console,
             PRIOR_LOG / 'training/cloth_contract.json', PRIOR_RUN / 'model_step_000000.json']
    paths += [shared.ROOT / row['path'] for row in prepared['motions']]
    for path in paths:
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)
    atomic_json(shared.OUT / 'prepare.json', prepared)
    for row in prepared['motions']:
        source = shared.ROOT / row['path']
        (shared.OUT / 'motionlib' / source.name).symlink_to(os.path.relpath(source, shared.OUT / 'motionlib'))
    print('E39 registered: one first-batch capture, 49 fixed-input forwards, zero optimizer/backward', flush=True)


def verify(job):
    import torch
    from script.a3.cloth_std_contract import canonical_policy_state
    from script.a3.likelihood_probe import compare
    from script.a3.zhy_likelihood_trainer import sampling_identity
    torch.set_num_threads(1)
    directory = shared.LOG / 'training'
    receipt = shared.read(directory / 'capture_complete.json')
    if (receipt['optimizer_steps'] or receipt['global_step'] or
            receipt['parameters_before_sha256'] != receipt['parameters_after_sha256']):
        raise ValueError('Capture performed updates')
    for name, row in receipt['artifacts'].items():
        if sha256(directory / name) != row['sha256'] or (directory / name).stat().st_size != row['bytes']:
            raise ValueError('Capture evidence changed')
    source = torch.load(shared.OFFICIAL, map_location='cpu', weights_only=False)
    pilot = torch.load(shared.RUN / 'model_step_000000.pt', map_location='cpu', weights_only=False)
    policy_source = canonical_policy_state(source['policy_state_dict'])
    for key, expected in (('policy_state_dict', policy_source), ('value_state_dict', source['value_state_dict'])):
        if pilot[key].keys() != expected.keys() or any(not torch.equal(v, pilot[key][n]) for n, v in expected.items()):
            raise ValueError('Step0 is not official runtime warm start')
    if pilot['optimizer_state_dict']['state'] or list(shared.RUN.glob('model_step_00000[12].pt')):
        raise ValueError('Unexpected optimizer state or new checkpoint')
    del source, pilot, policy_source
    sampled = torch.load(directory / 'sampling.pt', map_location='cpu', weights_only=False)
    actual = torch.load(directory / 'first_batch.pt', map_location='cpu', weights_only=False)
    variants = torch.load(directory / 'variants.pt', map_location='cpu', weights_only=False)
    identity = sampling_identity(sampled['sampling'], sampled['rollout'])
    if not identity['all_inputs_exact'] or not identity['sampled_means_match_storage']:
        raise ValueError('Sampling capture differs from training storage')
    batch, result = actual['batch'], actual['policy']
    indices = batch['micro_batch_inds']
    if any(not torch.equal(v, batch['mb_obs_dict'][n]) for n, v in actual['actor_inputs'].items()):
        raise ValueError('First Actor did not receive stored microbatch inputs')
    valid = ~batch['mb_padding_mask']
    rows = []
    for name, mean in [('actual_training_batch', result['action_mean'])] + [(n, v['mean']) for n, v in variants.items()]:
        rows.append(dict(variant=name, **compare(mean[valid], batch['mb_old_mu'][valid],
            result['action_std'][valid], batch['mb_old_sigma'][valid], batch['mb_actions'][valid],
            batch['mb_logprobs'][valid])))
    layers = {}
    for name in actual['encoded']['tokens']:
        tokens = [r['encoded']['tokens'].get(name) for r in sampled['sampling']]
        latents = [r['encoded']['latents'].get(name) for r in sampled['sampling']]
        if any(v is None or v.shape[0] != 16 for v in tokens + latents):
            layers[name] = dict(status='COMPACT_ENCODER_MASK_LAYOUT_REQUIRES_FURTHER_MAPPING')
            continue
        sample_tokens = torch.stack(tokens, dim=1)[indices]
        sample_latents = torch.stack(latents, dim=1)[indices]
        target = actual['encoded']['tokens'][name]
        if target.numel() != sample_tokens.numel():
            layers[name] = dict(status='ENCODER_MASK_COVERAGE_DIFFERS')
            continue
        target, latent = target.reshape_as(sample_tokens), actual['encoded']['latents'][name].reshape_as(sample_latents)
        delta = target != sample_tokens
        frame_delta = delta.flatten(2).any(-1)
        layers[name] = dict(status='COMPARED', changed_token_coordinates=int(delta.sum()),
            changed_token_frames=int(frame_delta.sum()), changed_frame_indices=frame_delta.nonzero().tolist(),
            latent_max_abs=float((latent-sample_latents).abs().max()),
            changed_token_max_abs=float((target-sample_tokens).abs().max()))
    difference = (result['action_mean']-batch['mb_old_mu']).abs().amax(-1)
    output = dict(status='VERIFIED_ZERO_OPTIMIZER_CAPTURE', experiment=job['experiment'],
        optimizer_steps=0, verified_official_step0=True, sampling_identity=identity,
        actual_actor_inputs_match_storage=True, valid_frames=int(valid.sum()),
        micro_batch_environment_indices=indices.tolist(), changed_action_frames=(difference > 1e-4).nonzero().tolist(),
        rows=rows, encoder_layers=layers, raw_capture_receipt=receipt,
        probability_gate_relaxed=False, improvement_proven=False, root_cause_claim='NOT_YET_CAUSAL_ISOLATION',
        fresh_final_test=False, deploy_allowed=False)
    atomic_json(shared.OUT / 'training_verification.json', output)


shared.verify_training = verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='train')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else runtime.run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
