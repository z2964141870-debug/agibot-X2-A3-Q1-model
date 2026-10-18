"""Tensor-free contracts for the independent D01 output-head trial."""

import hashlib
import json
import math

from script.a3.finite_retention_protocol import SCALES, accept

HEAD_SHAPES = {
    'actor_module.decoders.g1_dyn.module.12.weight': (29, 512),
    'actor_module.decoders.g1_dyn.module.12.bias': (29,),
}
HEAD_KEYS = tuple(HEAD_SHAPES)
HEAD_PARAMETERS = 14877
CALL_BUDGET = dict(rollout=192, training_batch=96, initial_teacher=4,
                   committed_teacher=128, finite_candidates=1728, total=2148)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def restrict_head(policy):
    parameters = dict(policy.named_parameters())
    if any(n not in parameters or tuple(parameters[n].shape) != shape
           for n, shape in HEAD_SHAPES.items()):
        raise ValueError('Exact original 512-to-29 output head required')
    if 'std' not in parameters or not any(n.startswith('actor_module.encoders.') for n in parameters):
        raise ValueError('Official Actor codec and std required')
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in HEAD_SHAPES)
    return dict(trainable=[n for n in parameters if n in HEAD_SHAPES],
                frozen=[n for n in parameters if n not in HEAD_SHAPES])


def ownership(wrapper, optimizer):
    parameters = list(wrapper.named_parameters(remove_duplicate=False))
    if len({n for n, _ in parameters}) != len(parameters) or len({id(p) for _, p in parameters}) != len(parameters):
        raise ValueError('Aliased or duplicate wrapper parameter')
    named = dict(parameters)
    actor = dict(wrapper.policy.named_parameters())
    if any(n not in actor or tuple(actor[n].shape) != shape for n, shape in HEAD_SHAPES.items()):
        raise ValueError('Head shape changed')
    if {n for n, p in actor.items() if p.requires_grad} != set(HEAD_KEYS):
        raise ValueError('Actor trainability escaped exact head whitelist')
    critic_parameters = dict(wrapper.value_model.named_parameters())
    critic = set(critic_parameters)
    if (not critic or any(not p.requires_grad for p in critic_parameters.values())
            or getattr(wrapper, 'disc_model', None) is not None):
        raise ValueError('Original Critic and no discriminator required')
    expected = {'policy.' + n for n in HEAD_KEYS} | {'value_model.' + n for n in critic}
    if any(not n.startswith(('policy.', 'value_model.')) for n in named):
        raise ValueError('Unexpected additional model owner')
    if {n for n, p in parameters if p.requires_grad} != expected:
        raise ValueError('Unexpected wrapper trainability')
    by_id = {id(p): n for n, p in parameters}
    groups, ids = [], []
    for group in optimizer.param_groups:
        members = []
        for parameter in group['params']:
            name = by_id.get(id(parameter))
            if name not in expected or not parameter.requires_grad:
                raise ValueError('Frozen or foreign optimizer owner')
            ids.append(id(parameter))
            members.append(name)
        groups.append(members)
    flat = [n for group in groups for n in group]
    if len(ids) != len(set(ids)) or set(flat) != expected:
        raise ValueError('Missing or duplicate optimizer owner')
    return dict(groups=groups, shapes={n: list(named[n].shape) for n in sorted(expected)},
                actor_keys=list(HEAD_KEYS), actor_parameters=HEAD_PARAMETERS,
                critic_keys=sorted(critic))


def serialized_owners(param_groups, contract):
    groups = contract['groups']
    if len(param_groups) != len(groups):
        raise ValueError('Saved optimizer group count changed')
    mapping = {}
    for group, names in zip(param_groups, groups):
        if len(group['params']) != len(names):
            raise ValueError('Saved optimizer group length changed')
        for identifier, name in zip(group['params'], names):
            if type(identifier) is not int or identifier in mapping:
                raise ValueError('Duplicate or invalid serialized Adam ID')
            mapping[identifier] = name
    flat = [n for group in groups for n in group]
    expected = {'policy.' + n for n in HEAD_KEYS} | {'value_model.' + n for n in contract['critic_keys']}
    if len(flat) != len(set(flat)) or set(flat) != expected or set(contract['shapes']) != expected:
        raise ValueError('Serialized owner names incomplete or escaped scope')
    if any(contract['shapes']['policy.' + n] != list(shape) for n, shape in HEAD_SHAPES.items()):
        raise ValueError('Serialized head shape changed')
    return mapping


def changed_actor(names, step):
    if (step == 0 and names) or (step != 0 and (not names or not set(names) <= set(HEAD_KEYS))):
        raise ValueError('Actor changes escaped original output head or were empty')


def forward_accounting(rollouts, batches, commits, evaluations):
    counts = dict(rollout=24 * rollouts, training_batch=3 * batches,
                  initial_teacher=4, committed_teacher=4 * commits,
                  finite_candidates=6 * evaluations)
    counts['total'] = sum(counts.values())
    if any(counts[k] > CALL_BUDGET[k] for k in counts):
        raise ValueError('Original outer Actor/decoder call budget exceeded')
    return counts


def transactions(rows, owner_sha256):
    """Validate completed pairs and an optional final scientific rejection."""
    commits, rejected, evaluations, candidate_checks = [], None, 0, 0
    index = 0
    while index < len(rows):
        row = rows[index]
        proposal = len(commits) + 1
        if proposal > 32 or row['proposal'] != proposal or row['owner_sha256'] != owner_sha256:
            raise ValueError('Proposal budget or ownership identity changed')
        gradients = row['head_gradients']
        if set(gradients) != set(HEAD_KEYS) or any(
                gradients[n]['shape'] != list(shape)
                or not math.isfinite(gradients[n]['max_abs']) or gradients[n]['max_abs'] < 0
                or len(gradients[n]['sha256']) != 64
                for n, shape in HEAD_SHAPES.items()):
            raise ValueError('Missing or invalid exact-head gradient evidence')
        if not accept(row['before']['objective'], row['before']):
            raise ValueError('Live proposal starts outside original protection')
        candidates = row['candidates']
        if not candidates or [c['scale'] for c in candidates] != list(SCALES[:len(candidates)]):
            raise ValueError('Candidate scale order or budget changed')
        for candidate in candidates:
            accepted = False if candidate.get('zero_displacement', False) else accept(row['before']['objective'], candidate)
            if candidate['accepted'] != accepted:
                raise ValueError('Candidate decision differs from fixed gates')
        checks = sum(not c.get('zero_displacement', False) for c in candidates)
        if row['evaluations'] != 1 + checks:
            raise ValueError('Evaluation count differs from actual candidates')
        if row['parameters_and_optimizer_now_sha256'] != row['parameters_and_optimizer_before_sha256']:
            raise ValueError('Pending or rejected proposal changed live state')
        evaluations += 1 + checks
        candidate_checks += checks
        if row['status'] == 'REJECTED_NO_LIVE_UPDATE':
            if index != len(rows) - 1 or len(candidates) != len(SCALES) or any(c['accepted'] for c in candidates):
                raise ValueError('Rejection must exhaust exactly the fixed scales and terminate')
            rejected = row
            break
        if row['status'] != 'ACCEPTED_PENDING_COMMIT' or index + 1 >= len(rows):
            raise ValueError('Unpaired pending transaction')
        commit = rows[index + 1]
        if (commit['status'] != 'COMMITTED' or commit['proposal'] != proposal
                or commit['owner_sha256'] != owner_sha256
                or not candidates[-1]['accepted'] or any(c['accepted'] for c in candidates[:-1])
                or any(commit[k] != row[k] for k in ('before', 'candidates', 'evaluations',
                    'parameters_and_optimizer_before_sha256', 'nominal_group_lrs', 'head_gradients'))):
            raise ValueError('Committed transaction differs from first accepted proposal')
        commits.append(commit)
        index += 2
    if evaluations > 288 or candidate_checks > 256:
        raise ValueError('Finite forward budget exceeded')
    return dict(commits=commits, rejected=rejected, proposals=len(commits) + int(rejected is not None),
                evaluations=evaluations, candidate_checks=candidate_checks)
