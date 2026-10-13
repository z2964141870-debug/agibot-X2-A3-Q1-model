"""E41 changes exactly one saved token with a fixed official decoder."""

import argparse
import hashlib
import math
import time

from script.a3 import cloth_token_audit as saved
from script.a3.checkpoint_store import atomic_json, sha256

shared = saved.shared
PRIOR = shared.OUT
STEP0 = shared.ROOT / 'data/training/a3_zhy_cloth_capture_20261013_E39/R16_001_s0/model_step_000000.pt'
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_intervention_20261013_E41'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_intervention_20261013_E41'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_intervention_20261013_E41/NO_TRAINING'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_intervention_20261013_E41.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_INTERVENTION_20261013_E41'
shared.MODULE = 'script.a3.cloth_token_intervention'
shared.CODE += ('script/a3/cloth_token_intervention.py',)


def register():
    prior = shared.read(PRIOR / 'training_verification.json')
    if (prior['status'] != 'VERIFIED_SAVED_TOKEN_MAPPING' or
            prior['encoder_layers']['a3_fast']['changed_coordinates'] != 1):
        raise ValueError('Requires E40 one-coordinate saved token evidence')
    saved.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment=prior['experiment'], finalized_at=time.time(), model_loads=1,
        new_forward_calls=2, new_policy_rollouts=0, optimizer_steps=0,
        main_variable='ONE_TOKEN_COORDINATE_INTERVENTION_FIXED_INPUT_DECODER_BATCH',
        action_match_max_abs=1e-5, ratio_deviation_max=.001, batch_shape=[4, 24],
        changed_coordinate=[3, 2, 1, 24], old_token=-.4375, replacement_token=-.375)
    for path in (PRIOR / 'training_verification.json', PRIOR / 'state.json', STEP0, STEP0.with_suffix('.json')):
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)


def intervene(job):
    import torch
    from omegaconf import OmegaConf
    from gear_sonic.trl.utils.common import custom_instantiate
    from script.a3.likelihood_probe import compare
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = torch.load(saved.CAPTURE / 'first_batch.pt', map_location='cpu', weights_only=False)
    sampled = torch.load(saved.CAPTURE / 'sampling.pt', map_location='cpu', weights_only=False)
    meta = shared.read(saved.CAPTURE / 'runtime_metadata.json')
    config = OmegaConf.create(meta)
    actor = custom_instantiate(config.algo_config.actor, env_config=config.env_config,
                              algo_config=config.algo_config, _resolve=False)
    checkpoint = torch.load(STEP0, map_location='cpu', weights_only=False)
    actor.load_state_dict(checkpoint['policy_state_dict'], strict=True)
    del checkpoint
    actor.requires_grad_(False)
    actor.eval()
    def digest():
        result = hashlib.sha256()
        for name, value in actor.state_dict().items():
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()
    before = digest()
    mini = actual['batch']['mb_obs_dict']
    indices = actual['batch']['micro_batch_inds']
    dims = meta['env_config']['obs']['group_obs_dims']['tokenizer']
    names = meta['env_config']['obs']['group_obs_names']['tokenizer']
    offset = sum(math.prod(dims[n]) for n in names[:names.index('encoder_index')])
    columns = list(meta['algo_config']['actor']['backbone']['encoder_sample_probs'])
    def mask(obs, name):
        return obs['tokenizer'][..., offset+columns.index(name)].bool()
    original = torch.zeros(4, 24, 2, 32)
    gold = torch.zeros_like(original)
    for name in ('g1', 'a3_fast'):
        active = mask(mini, name)
        original[active] = saved.inflate(actual['encoded']['tokens'][name], active)[active]
        pieces = [saved.inflate(row['encoded']['tokens'][name], mask(row['inputs'], name))
                  for row in sampled['sampling']]
        selected = torch.cat(pieces, dim=1)[indices]
        gold[active] = selected[active]
    changed = (original != gold).nonzero().tolist()
    if changed != [job['changed_coordinate']]:
        raise ValueError('Intervention would change more than the registered coordinate')
    corrected = original.clone()
    coordinate = tuple(job['changed_coordinate'])
    if float(original[coordinate]) != job['old_token'] or float(gold[coordinate]) != job['replacement_token']:
        raise ValueError('Registered token values changed')
    corrected[coordinate] = job['replacement_token']
    if not torch.equal(corrected, gold):
        raise ValueError('One-coordinate intervention did not reconstruct sampling tokens')
    actor.to('cuda')
    obs = {name:value.to('cuda') for name, value in actual['actor_inputs'].items()}
    means = {}
    with torch.no_grad():
        for name, tokens in (('training_tokens', original), ('one_coordinate_corrected', corrected)):
            means[name] = actor.actor_module.forward_with_external_tokens(obs, tokens.to('cuda')).cpu()
    torch.cuda.synchronize()
    after = digest()
    if before != after:
        raise ValueError('Intervention changed model state')
    batch, result = actual['batch'], actual['policy']
    valid = ~batch['mb_padding_mask']
    rows = [dict(condition=name, **compare(mean[valid], batch['mb_old_mu'][valid],
        result['action_std'][valid], batch['mb_old_sigma'][valid],
        batch['mb_actions'][valid], batch['mb_logprobs'][valid])) for name,mean in means.items()]
    wrong_reconstruction = float((means['training_tokens']-result['action_mean']).abs().max())
    passed = (wrong_reconstruction <= 1e-5 and rows[0]['action_max_abs'] > 1e-4 and
        rows[0]['max_abs_ratio_minus_one'] > .001 and rows[1]['action_max_abs'] <= 1e-5 and
        rows[1]['max_abs_ratio_minus_one'] <= .001)
    path = shared.OUT / 'decoder_intervention.pt'
    with path.open('xb') as stream:
        torch.save(dict(means=means, old_mean=batch['mb_old_mu'], old_std=batch['mb_old_sigma'],
            new_std=result['action_std'], actions=batch['mb_actions'], logprobs=batch['mb_logprobs'],
            valid=valid, original_tokens=original, corrected_tokens=corrected), stream)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    # Verify the probability arithmetic independently after reloading the saved outputs on CPU.
    restored = torch.load(path, map_location='cpu', weights_only=False)
    for row in rows:
        name = row['condition']
        recomputed = compare(restored['means'][name][valid], restored['old_mean'][valid],
            restored['new_std'][valid], restored['old_std'][valid], restored['actions'][valid],
            restored['logprobs'][valid])
        if any(recomputed[key] != value for key,value in row.items() if key != 'condition'):
            raise ValueError('Saved-output independent likelihood arithmetic differs')
    atomic_json(shared.OUT / 'training_verification.json', dict(
        status='VERIFIED_TOKEN_INTERVENTION' if passed else 'REJECTED_TOKEN_INTERVENTION',
        intervention_passed=passed, changed_coordinate=job['changed_coordinate'], rows=rows,
        training_mean_reconstruction_max_abs=wrong_reconstruction,
        state_before_sha256=before, state_after_sha256=after,
        model_loads=1, decoder_forward_calls=2, encoder_forward_calls=0, quantizer_forward_calls=0,
        optimizer_steps=0, new_policy_rollouts=0, saved_outputs_sha256=sha256(path),
        independently_reloaded_cpu_probability_verified=True,
        scope='ONE_BATCH_NUMERICAL_ACTION_CAUSE_NOT_FALL_CAUSE_OR_POLICY_IMPROVEMENT',
        probability_gate_relaxed=False, deploy_allowed=False))


shared.verify_training = intervene

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='verify-training')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else saved.runtime.run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
