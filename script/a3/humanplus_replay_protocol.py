"""Tensor-free D02 layout, provenance and source-derived call accounting."""

import math

from script.a3.humanplus_head_protocol import forward_accounting as head_accounting

INPUT_ORDER = 'command_multi_future_then_ori6d_multi_future_then_proprioception'
COMMAND = 'command_multi_future_a3_fast_nonflat'
ORIENTATION = 'motion_anchor_ori_a3_fast_b_mf_nonflat'
CALL_BUDGET = dict(rollout=192, training_batch=96, initial_teacher=4,
                   committed_teacher=128, finite_candidates=1728,
                   replay_initial=4, replay_candidates=288, total=2440)


def slices(specs, total, encoder_names, proprioception):
    if list(encoder_names) != ['g1', 'teleop', 'smpl', 'a3_fast'] or list(proprioception) != ['actor_obs']:
        raise ValueError('Pinned native routing or proprioception changed')
    result, cursor = {}, 0
    for name, start, end, dims in specs:
        if name in result or start != cursor or end - start != math.prod(dims):
            raise ValueError('Invalid or overlapping tokenizer layout')
        result[name] = (start, end, tuple(dims))
        cursor = end
    if cursor != total or any(result.get(n, (None, None, None))[2] != shape for n, shape in
            ((COMMAND, (10, 58)), (ORIENTATION, (10, 6)), ('encoder_index', (4,)))):
        raise ValueError('Pinned 10-frame a3_fast schema changed')
    if 'command_num_frames' in result:
        raise ValueError('Variable-frame tokenizer is outside the fixed D02 bridge')
    return result


def validate_selection(design):
    examples = design['loss_examples']
    if (examples['input_order'] != INPUT_ORDER or examples['actor_input_dim'] != 1570
            or examples['action_dim'] != 29 or examples['total_frames'] != 384
            or len(examples['selection']) != 4):
        raise ValueError('D02 example ABI changed')
    seen = set()
    for row in examples['selection']:
        n = row['full_policy_steps']
        expected = [i * (n - 1) // 95 for i in range(96)]
        if (row['key'] in seen or row['selected_policy_ticks'] != expected
                or row['count'] != 96 or len(set(expected)) != 96):
            raise ValueError('D02 fixed uniform sample selection changed')
        seen.add(row['key'])
    return examples['selection']


def forward_accounting(rollouts, batches, commits, evaluations):
    result = head_accounting(rollouts, batches, commits, evaluations)
    result.pop('total')
    result.update(replay_initial=4, replay_candidates=evaluations)
    result['total'] = sum(result.values())
    if any(result[k] > CALL_BUDGET[k] for k in result):
        raise ValueError('D02 outer Actor/decoder budget exceeded')
    return result


def validate_receipt(receipt, design, batches, rows):
    selection = validate_selection(design)
    identities = [dict(row, **design['read_only_sources'][row['source']]) for row in selection]
    if (receipt['sources'] != identities or receipt['frames'] != 384
            or receipt['input_order'] != INPUT_ORDER or not receipt['preclip_labels']
            or receipt['native_route'] != [0, 0, 0, 1] or receipt['initial_calls'] != 4
            or receipt['coefficient'] != 1. or not receipt['legacy_hard_guard_unchanged']
            or receipt['label_tolerance'] != 1e-5
            or not 0 <= receipt['fresh035_label_max_abs'] <= 1e-5
            or receipt['roles'] != 'DEVELOPMENT_REGULARIZATION_NOT_HELD_OUT'
            or len(receipt['batches']) != batches):
        raise ValueError('Incomplete or changed replay regularizer evidence')
    pending = [row for row in rows if row['status'] != 'COMMITTED']
    if len(pending) != batches:
        raise ValueError('Replay loss and finite batch counts differ')
    for index, (batch, row) in enumerate(zip(receipt['batches'], pending)):
        if (batch['commit_before'] != index or batch['quarter'] != index % 4
                or batch['frames'] != 96 or not math.isfinite(batch['loss']) or batch['loss'] < 0
                or not math.isclose(batch['loss'], row['before']['replay_retention_loss'],
                                    rel_tol=1e-6, abs_tol=1e-7)):
            raise ValueError('Live loss differs from fixed replay quarter')
        for candidate in (row['before'], *row['candidates']):
            if candidate.get('zero_displacement', False):
                continue
            if (candidate['replay_quarter'] != index % 4
                    or not math.isfinite(candidate['replay_retention_loss'])
                    or candidate['replay_retention_loss'] < 0):
                raise ValueError('Candidate replay objective evidence changed')
    return dict(replay_frames=384, replay_batches=batches,
                fresh035_replay_label_max_abs=receipt['fresh035_label_max_abs'],
                legacy_hard_guard_unchanged=True)
