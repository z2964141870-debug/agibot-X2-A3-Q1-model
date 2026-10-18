"""Independent tensor inspection of HP12 checkpoints and transaction receipts."""

import copy
import json


def recipe(new, old):
    left, right = copy.deepcopy(new), copy.deepcopy(old)
    for tree in (left, right):
        tree['manager_env']['commands']['motion']['motion_lib_cfg']['motion_file'] = '<MOTION_DIRECTORY>'
        for name in ('output_dir', 'logging_dir'):
            tree['algo']['trl'][name] = '<STORAGE>'
    if left['manager_env'] != right['manager_env'] or left['algo'] != right['algo']:
        raise ValueError('Environment or loss/optimizer recipe changed beyond declared storage')


def verify():
    import torch
    import yaml
    from script.a3 import humanplus_finite as trial
    from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
    from script.a3.cloth_constrained_trainer import DECODER, initial_probability
    from script.a3.cloth_std_contract import canonical_policy_state
    from script.a3.finite_retention_protocol import accept, SCALES
    torch.set_num_threads(1)
    original = torch.load(trial.OFFICIAL, map_location='cpu', weights_only=False)
    official = canonical_policy_state(original['policy_state_dict'])
    records = []
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        record = trial.read(path.with_suffix('.json'))
        if record['sha256'] != sha256(path) or record['bytes'] != path.stat().st_size:
            raise ValueError('Checkpoint size/SHA changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if validate_payload(payload) != step:
            raise ValueError('Saved step mismatch')
        actor = payload['policy_state_dict']
        if actor.keys() != official.keys():
            raise ValueError('Actor key set changed')
        names = [n for n, v in actor.items() if not torch.equal(v, official[n])]
        if step == 0:
            critic, target = payload['value_state_dict'], original['value_state_dict']
            if (names or payload['optimizer_state_dict']['state'] or critic.keys() != target.keys()
                    or any(not torch.equal(v, target[n]) for n, v in critic.items())):
                raise ValueError('Not fresh035 Actor/Critic/emptyAdam')
            del critic, target
        else:
            if not names or any(not n.startswith(DECODER) for n in names):
                raise ValueError('Actor changes escaped selected decoder')
            if {int(v['step']) for v in payload['optimizer_state_dict']['state'].values()} != {4 * step}:
                raise ValueError('Adam counters differ from committed transactions')
        records.append(dict(record, path=str(path.relative_to(trial.ROOT)), changed_actor_names=names))
        del payload, actor
    directory = trial.LOG / 'train'
    finite, cloth = (trial.read(directory / n) for n in ('finite_contract.json', 'cloth_contract.json'))
    batches = trial.read(directory / 'sampled_token_contract.json')['batches']
    audit = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    transactions = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    if (finite['proposals'] != 32 or finite['committed'] != 32 or len(finite['rows']) != 32
            or len(transactions) != 64 or len(audit) != 32 or len(batches) != 32
            or len(cloth['optimizer_steps']) != 32
            or [r['attempt_optimizer_step'] for r in audit] != list(range(1, 33))):
        raise ValueError('Incomplete optimizer/transaction/batch coverage')
    for i, row in enumerate(finite['rows']):
        candidates = row['candidates']
        pending, committed = transactions[2*i:2*i+2]
        if (row != committed or pending['status'] != 'ACCEPTED_PENDING_COMMIT'
                or committed['status'] != 'COMMITTED' or row['proposal'] != i+1
                or pending['parameters_and_optimizer_now_sha256'] != pending['parameters_and_optimizer_before_sha256']
                or pending['candidates'] != candidates
                or [r['scale'] for r in candidates] != list(SCALES[:len(candidates)])
                or not candidates[-1]['accepted'] or any(r['accepted'] for r in candidates[:-1])
                or not accept(row['before']['objective'], candidates[-1])):
            raise ValueError('Finite candidate decision/commit audit failed')
    if (any(not r['decoder_gradient_enabled'] for r in batches)
            or any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in cloth['optimizer_steps'])):
        raise ValueError('Teacher or decoder-gradient protection failed')
    repair = torch.load(directory / 'initial_token_repair.pt', map_location='cpu', weights_only=False)
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    batch = repair['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('Actual sampled tokens changed')
    probability = initial_probability(repair['corrected'], batch)
    recipe(yaml.safe_load((trial.RUN / 'config.yaml').read_text()),
           yaml.safe_load(trial.OLD_CONFIG.read_text()))
    routes = trial.read(directory / 'reference_route_coverage.json')['rollouts']
    if (len(routes) != 8 or [r['update'] for r in routes] != list(range(1, 9))
            or any(sum(r['routes'].values()) != 384 for r in routes)
            or sum(sum(r['routes'].values()) for r in routes) != 3072):
        raise ValueError('Actual rollout budget differs')
    result = dict(status='COMPLETE', scope='VERIFIED_TRAINING_NOT_EFFECT', checkpoints=records,
        optimizer_steps=32, updates=8, initial_probability=probability, finite_contract=finite,
        original_contract=cloth, actual_optimizer_steps=audit, routes=routes,
        recipe_unchanged_except_registered_storage_and_optimizer_mechanism=True,
        deploy_allowed=False, final_effect_step=8)
    atomic_json(trial.OUT / 'training_verification.json', result)
    return result
