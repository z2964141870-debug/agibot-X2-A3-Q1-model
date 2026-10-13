"""E40 maps saved compact encoder outputs without models or new forward calls."""

import argparse
import math
import time

from script.a3 import cloth_resource_trial as runtime
from script.a3.checkpoint_store import atomic_json, sha256

shared = runtime.shared
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_capture_20261013_E39'
CAPTURE = shared.ROOT / 'logs/a3_zhy_cloth_capture_20261013_E39/training'
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_tokens_20261013_E40'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_tokens_20261013_E40'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_tokens_20261013_E40/NO_TRAINING'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_tokens_20261013_E40.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_TOKENS_20261013_E40'
shared.TRIAL = 'NO_TRAINING'
shared.MODULE = 'script.a3.cloth_token_audit'
shared.CODE += ('script/a3/cloth_token_audit.py',)
shared.STAGES = {'verify-training': ('REGISTERED', 'COMPLETE')}


def register():
    prior = shared.read(PRIOR / 'training_verification.json')
    if prior['status'] != 'VERIFIED_ZERO_OPTIMIZER_CAPTURE' or prior['optimizer_steps']:
        raise ValueError('Requires complete zero-update E39')
    runtime.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment=prior['experiment'], finalized_at=time.time(), updates=0,
        optimizer_steps=0, model_loads=0, new_forward_calls=0, new_policy_rollouts=0,
        main_variable='SAVED_ENCODER_COMPACT_MASK_MAPPING_NOT_NEW_INFERENCE')
    for path in [PRIOR / 'training_verification.json', PRIOR / 'state.json'] + [CAPTURE / n for n in
            ('sampling.pt', 'first_batch.pt', 'variants.pt', 'runtime_metadata.json', 'capture_complete.json')]:
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)


def inflate(values, mask):
    import torch
    if int(mask.sum()) != values.shape[0]:
        raise ValueError('Compact encoder row count does not match mask')
    full = torch.full((*mask.shape, *values.shape[1:]), float('nan'), dtype=values.dtype)
    full.reshape(-1, *values.shape[1:])[mask.flatten()] = values
    return full


def audit(job):
    import torch
    torch.set_num_threads(1)
    # A known noncontiguous example catches treating compact rows as full environments.
    mask = torch.tensor([[False, True], [True, False]])
    toy = inflate(torch.tensor([[3.], [7.]]), mask)
    if float(toy[0, 1, 0]) != 3 or float(toy[1, 0, 0]) != 7 or not torch.isnan(toy[0, 0, 0]):
        raise ValueError('Noncontiguous mask positive control failed')
    try:
        inflate(torch.zeros(3, 1), mask)
    except ValueError:
        pass
    else:
        raise ValueError('Mask/count mismatch negative control failed')
    recorded = shared.read(CAPTURE / 'capture_complete.json')
    for name in ('sampling.pt', 'first_batch.pt', 'variants.pt', 'runtime_metadata.json'):
        if sha256(CAPTURE / name) != recorded['artifacts'][name]['sha256']:
            raise ValueError('Saved E39 capture changed')
    sampled = torch.load(CAPTURE / 'sampling.pt', map_location='cpu', weights_only=False)
    actual = torch.load(CAPTURE / 'first_batch.pt', map_location='cpu', weights_only=False)
    variants = torch.load(CAPTURE / 'variants.pt', map_location='cpu', weights_only=False)
    meta = shared.read(CAPTURE / 'runtime_metadata.json')
    dims = meta['env_config']['obs']['group_obs_dims']['tokenizer']
    order = meta['env_config']['obs']['group_obs_names']['tokenizer']
    offset = sum(math.prod(dims[n]) for n in order[:order.index('encoder_index')])
    column_order = list(meta['algo_config']['actor']['backbone']['encoder_sample_probs'])
    def native(obs, name):
        return obs['tokenizer'][..., offset + column_order.index(name)].bool()
    indices = actual['batch']['micro_batch_inds']
    mini = actual['batch']['mb_obs_dict']
    full_obs = sampled['rollout']['all_obs_dict']
    action_changed = (actual['policy']['action_mean']-actual['batch']['mb_old_mu']).abs().amax(-1) > 1e-4
    layers = {}
    for name in ('g1', 'a3_fast'):
        rows, latents = [], []
        for record in sampled['sampling']:
            m = native(record['inputs'], name)
            rows.append(inflate(record['encoded']['tokens'][name], m))
            latents.append(inflate(record['encoded']['latents'][name], m))
        gold, gold_latent = torch.cat(rows, dim=1)[indices], torch.cat(latents, dim=1)[indices]
        active = native(mini, name)
        target = inflate(actual['encoded']['tokens'][name], active)
        target_latent = inflate(actual['encoded']['latents'][name], active)
        delta = (target != gold) & active[..., None, None]
        coords = delta.nonzero().tolist()
        changed_frames = delta.flatten(2).any(-1)
        details = []
        for coord in coords:
            b, t, token, channel = coord
            key = tuple(coord)
            details.append(dict(mini_environment=b, global_environment=int(indices[b]), tick=t,
                token=token, channel=channel, sampled_token=float(gold[key]),
                training_token=float(target[key]), sampled_latent=float(gold_latent[key]),
                training_latent=float(target_latent[key]), action_changed=bool(action_changed[b, t])))
        controls = []
        for variant_name, value in variants.items():
            if 'encoded' in value:
                candidate = inflate(value['encoded']['tokens'][name], active)
            else:
                obs = full_obs if variant_name.startswith('full') else mini
                pieces = [inflate(layer['tokens'][name], native({n:v[:,t:t+1] for n,v in obs.items()}, name))
                          for t, layer in enumerate(value['frame_encoded'])]
                candidate = torch.cat(pieces, dim=1)
                if variant_name.startswith('full'):
                    candidate = candidate[indices]
            differences = (candidate != gold) & active[..., None, None]
            controls.append(dict(variant=variant_name, changed_coordinates=int(differences.sum()),
                                 changed_frames=int(differences.flatten(2).any(-1).sum())))
        layers[name] = dict(native_frames=int(active.sum()), changed_coordinates=len(coords),
            changed_frames=int(changed_frames.sum()), details=details, controls=controls,
            latent_max_abs=float((target_latent-gold_latent)[active].abs().max()),
            changed_tokens_have_changed_actions=bool(torch.all(~changed_frames | action_changed)),
            all_changed_actions_have_changed_tokens=bool(torch.all(~action_changed | changed_frames)))
    output = dict(status='VERIFIED_SAVED_TOKEN_MAPPING', experiment=job['experiment'],
        mapping_controls=2, encoder_index_offset=offset, encoder_column_order=column_order,
        batch_environment_indices=indices.tolist(), encoder_layers=layers,
        optimizer_steps=0, model_loads=0, new_forward_calls=0, new_policy_rollouts=0,
        scope='SAVED_TOKEN_CHANGE_AND_ACTION_COLOCATION_NOT_DECODER_CAUSAL_INTERVENTION',
        original_probability_gate_relaxed=False, improvement_proven=False, deploy_allowed=False)
    atomic_json(shared.OUT / 'training_verification.json', output)


shared.verify_training = audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='verify-training')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else runtime.run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
