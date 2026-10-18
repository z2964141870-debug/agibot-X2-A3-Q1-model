"""Successful-only saved expert/optimizer check; never used after STOPPED."""

import json

from script.a3.humanplus_specialist_protocol import (
    DECODER, MINIBATCHES, UPDATES, require_training_receipt, serialized_owners, validate_config,
)


def verify():
    import torch
    import yaml
    from script.a3 import humanplus_specialist_completion as trial
    from script.a3.checkpoint_store import sha256, validate_payload
    from script.a3.humanplus_saved_probability import initial_probability
    from script.a3.cloth_std_contract import canonical_policy_state
    state = trial.read(trial.RESULT)
    if (state['status'] != 'RUNNING' or state['current_stage'] != 'verify-training'
            or [r['stage'] for r in state['stages']] != []):
        raise ValueError('Only completed successful expert training may load saved models')
    trial.require_source()
    directory = trial.SOURCE_LOG / 'train'
    receipt = trial.read(directory / 'specialist_contract.json')
    require_training_receipt(receipt)
    torch.set_num_threads(1)
    original = torch.load(trial.OFFICIAL, map_location='cpu', weights_only=False)
    actor = canonical_policy_state(original['policy_state_dict'])
    records = []
    for step in (0, UPDATES):
        path = trial.checkpoint(step)
        sidecar = trial.read(path.with_suffix('.json'))
        if sidecar['bytes'] != path.stat().st_size or sidecar['sha256'] != sha256(path):
            raise ValueError('Saved expert identity changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Saved expert step changed')
        source = dict(original, policy_state_dict=actor)
        for kind in ('policy_state_dict', 'value_state_dict'):
            live, old = payload[kind], source[kind]
            if live.keys() != old.keys() or any(
                    v.shape != old[n].shape or v.dtype != old[n].dtype or not torch.isfinite(v).all()
                    for n, v in live.items()):
                raise ValueError('Expert network architecture or precision changed')
            if step == 0 and any(not torch.equal(v, old[n]) for n, v in live.items()):
                raise ValueError('Expert fresh035 did not match Actor/Critic')
        changes = [n for n, v in payload['policy_state_dict'].items() if not torch.equal(v, actor[n])]
        if any(not n.startswith(DECODER) for n in changes) or (step and not changes):
            raise ValueError('Expert changed frozen Actor or did not learn any decoder parameter')
        owners = trial.read(path.parent / 'specialist_owners.json')
        if owners['ownership'] != receipt['ownership']:
            raise ValueError('Saved optimizer owners changed')
        optimizer = payload['optimizer_state_dict']
        mapping = serialized_owners(optimizer['param_groups'], owners['ownership'])
        if {str(k): v for k, v in mapping.items()} != owners['serialized_id_names']:
            raise ValueError('Saved optimizer ID ordering changed')
        expected = set(mapping) if step else set()
        if set(optimizer['state']) != expected:
            raise ValueError('Missing or extra Adam moments')
        for identifier, moments in optimizer['state'].items():
            if set(moments) != {'step', 'exp_avg', 'exp_avg_sq'} or float(moments['step']) != MINIBATCHES * step:
                raise ValueError('Saved Adam state or step count changed')
            shape = owners['ownership']['shapes'][mapping[identifier]]
            for key in ('exp_avg', 'exp_avg_sq'):
                value = moments[key]
                if list(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Saved Adam moment ABI changed')
            if (moments['exp_avg_sq'] < 0).any():
                raise ValueError('Negative Adam variance')
        options = [{k: v for k, v in g.items() if k not in ('params', 'lr', 'initial_lr')}
                   for g in optimizer['param_groups']]
        if json.loads(json.dumps(options)) != owners['optimizer_options']:
            raise ValueError('Saved Adam options changed')
        records.append(dict(step=step, sha256=sidecar['sha256'], bytes=sidecar['bytes'],
                            changed_actor_names=changes, optimizer_parameters=len(mapping)))
        del payload
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    probability = torch.load(directory / 'initial_probability.pt', map_location='cpu', weights_only=False)
    batch = probability['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('Actual sampled token environment identity changed')
    check = initial_probability(probability['corrected'], batch)
    config = yaml.safe_load((trial.RUN / 'config.yaml').read_text())
    validate_config(config, trial.OUT / 'motionlib')
    old = yaml.safe_load((trial.ROOT / trial.read(trial.PLAN)['recipe_source']).read_text())
    for field in ('rewards', 'events', 'terminations', 'actions'):
        if config['manager_env'][field] != old['manager_env'][field]:
            raise ValueError('Specialist altered original reward/physics/reset recipe: ' + field)
    audit = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    if [r['attempt_optimizer_step'] for r in audit] != list(range(1, MINIBATCHES * UPDATES + 1)):
        raise ValueError('Missing or extra applied optimizer steps')
    return dict(status='COMPLETE', checkpoints=records, initial_probability=check,
        complete_training_receipt=receipt, reward_reset_recipe_preserved=True,
        saved_expert_verified=True, deploy_allowed=False)
