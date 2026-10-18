"""Successful D01 final-only checkpoint audit; never run on a stopped trial."""

import json

from script.a3.humanplus_finite_verification import recipe
from script.a3.humanplus_head_protocol import (
    HEAD_SHAPES, changed_actor, digest, serialized_owners, transactions,
)

from script.a3.humanplus_replay_protocol import forward_accounting, validate_receipt


def verify_optimizer(torch, payload, sidecar, contract, step):
    if (sidecar['ownership'] != contract or sidecar['owner_sha256'] != digest(contract)
            or sidecar['step'] != step):
        raise ValueError('Saved ownership contract changed')
    state = payload['optimizer_state_dict']
    options = [{k: v for k, v in g.items() if k not in ('params', 'lr', 'initial_lr')}
               for g in state['param_groups']]
    if json.loads(json.dumps(options)) != sidecar['optimizer_options']:
        raise ValueError('Saved Adam options differ from owner sidecar')
    mapping = serialized_owners(state['param_groups'], contract)
    if {str(k): v for k, v in mapping.items()} != sidecar['serialized_id_names']:
        raise ValueError('Saved Adam integer ID-to-name mapping changed')
    if set(state['state']) != (set(mapping) if step else set()):
        raise ValueError('Missing or additional Adam moments')
    for identifier, moments in state['state'].items():
        if set(moments) != {'step', 'exp_avg', 'exp_avg_sq'} or float(moments['step']) != 4 * step:
            raise ValueError('Adam moments or commit counter changed')
        shape = contract['shapes'][mapping[identifier]]
        for key in ('exp_avg', 'exp_avg_sq'):
            value = moments[key]
            if list(value.shape) != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Adam moment shape, precision or finiteness changed')
        if (moments['exp_avg_sq'] < 0).any():
            raise ValueError('Negative second Adam moment')
    return mapping


def verify():
    import torch
    import yaml
    from script.a3 import humanplus_replay_loss as trial
    from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
    from script.a3.humanplus_saved_probability import initial_probability
    from script.a3.cloth_std_contract import canonical_policy_state

    state = trial.read(trial.RESULT)
    if state['status'] != 'RUNNING' or state['current_stage'] != 'verify-training':
        raise ValueError('Only registered successful-training verification may load saved PT')
    if [r['stage'] for r in state['stages']] != ['train']:
        raise ValueError('Successful train stage required before any model audit')
    directory = trial.LOG / 'train'
    head = trial.read(directory / 'head_contract.json')
    contract = head['ownership']
    if head['owner_sha256'] != digest(contract) or len(head['steps']) != 32:
        raise ValueError('Incomplete exact-head runtime evidence')
    torch.set_num_threads(1)
    original = torch.load(trial.OFFICIAL, map_location='cpu', weights_only=False)
    official = canonical_policy_state(original['policy_state_dict'])
    records = []
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        record = trial.read(path.with_suffix('.json'))
        sidecar = trial.read(path.parent / 'head_owners.json')
        if (record['sha256'] != sha256(path) or record['bytes'] != path.stat().st_size
                or sidecar['checkpoint_sha256'] != record['sha256']):
            raise ValueError('Checkpoint identity changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Saved step mismatch')
        actor = payload['policy_state_dict']
        critic = payload['value_state_dict']
        if actor.keys() != official.keys() or critic.keys() != original['value_state_dict'].keys():
            raise ValueError('Original Actor/Critic architecture changed')
        names = [n for n, v in actor.items() if not torch.equal(v, official[n])]
        changed_actor(names, step)
        if any(tuple(actor[n].shape) != shape for n, shape in HEAD_SHAPES.items()):
            raise ValueError('Saved output-head ABI changed')
        for key, source in (('policy_state_dict', official), ('value_state_dict', original['value_state_dict'])):
            if any(v.shape != source[n].shape or v.dtype != source[n].dtype
                   or (v.is_floating_point() and v.dtype != torch.float32) or not torch.isfinite(v).all()
                   for n, v in payload[key].items()):
                raise ValueError('Saved network shape, precision or finiteness changed')
        if step == 0 and any(not torch.equal(v, original['value_state_dict'][n]) for n, v in critic.items()):
            raise ValueError('Step0 Critic differs from fresh035')
        verify_optimizer(torch, payload, sidecar, contract, step)
        records.append(dict(record, path=str(path.relative_to(trial.ROOT)), changed_actor_names=names,
                            owner_sha256=digest(contract)))
        del payload, actor, critic
    finite = trial.read(directory / 'finite_contract.json')
    cloth = trial.read(directory / 'cloth_contract.json')
    batches = trial.read(directory / 'sampled_token_contract.json')['batches']
    audit = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    rows = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    ledger = transactions(rows, digest(contract))
    if (ledger['rejected'] is not None or len(ledger['commits']) != 32
            or finite['proposals'] != 32 or finite['committed'] != 32
            or finite['rows'] != ledger['commits'] or finite['evaluations'] != ledger['evaluations']
            or len(batches) != 32 or len(cloth['optimizer_steps']) != 32
            or [r['attempt_optimizer_step'] for r in audit] != list(range(1, 33))
            or [r['commit'] for r in head['steps']] != list(range(1, 33))
            or any(not r['decoder_gradient_enabled'] for r in batches)
            or any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in cloth['optimizer_steps'])):
        raise ValueError('Incomplete exact-head optimizer, token or teacher coverage')
    for row, commit in zip(head['steps'], ledger['commits']):
        if (row['proposal'] != row['commit'] or row['owner_sha256'] != digest(contract)
                or row['gradients'] != commit['head_gradients'] or not row['frozen_actor_unchanged']):
            raise ValueError('Head gradient or frozen-state receipt mismatch')
    repair = torch.load(directory / 'initial_token_repair.pt', map_location='cpu', weights_only=False)
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    batch = repair['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('Actual sampled tokens changed')
    probability = initial_probability(repair['corrected'], batch)
    recipe(yaml.safe_load((trial.RUN / 'config.yaml').read_text()), yaml.safe_load(trial.OLD_CONFIG.read_text()))
    routes = trial.read(directory / 'reference_route_coverage.json')['rollouts']
    if (len(routes) != 8 or [r['update'] for r in routes] != list(range(1, 9))
            or any(r['transitions'] != 384 or sum(r['routes'].values()) != 384 for r in routes)):
        raise ValueError('Actual rollout budget differs')
    replay = trial.read(directory / 'replay_loss_contract.json')
    if replay['design_sha256'] != sha256(trial.DESIGN):
        raise ValueError('D02 replay design identity changed')
    replay_evidence = validate_receipt(replay, trial.read(trial.DESIGN), 32, rows)
    result = dict(status='COMPLETE', replay_regularizer=replay_evidence, scope='VERIFIED_HEAD_TRAINING_NOT_EFFECT', checkpoints=records,
        optimizer_steps=32, updates=8, initial_probability=probability, finite_contract=finite,
        head_contract=head, actual_optimizer_steps=audit, routes=routes,
        derived_outer_actor_decoder_calls=forward_accounting(8, 32, 32, ledger['evaluations']),
        call_count_basis='PINNED_SOURCE_PATHS_AND_COMPLETED_JSON_RECEIPTS_NOT_KERNEL_INSTRUMENTATION',
        recipe_unchanged_except_registered_storage_and_actor_trainable_scope=True,
        deploy_allowed=False, final_effect_step=8)
    atomic_json(trial.OUT / 'training_verification.json', result)
    return result


