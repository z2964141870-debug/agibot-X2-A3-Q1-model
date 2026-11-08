"""Read train-only saved states and measure objectives without updating a policy."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import INPUT_FIELDS, batch, causal_records
from student_coverage_data import catalog
from student_coverage_fit import evaluate_cells
from student_fit import module_sha
from student_models import action_distillation_loss
from student_onpolicy_pair_fit import warm_student
from student_retained_sequence_data import sample_windows
from student_retained_sequence_fit import increment_loss


def load_train(path):
    path = Path(path)
    if path.name not in ('train_data.npz', 'train_transitions.npz'):
        raise ValueError('Only registered training arrays may be decoded')
    with np.load(path, allow_pickle=False) as data:
        result = {k:data[k] for k in (*INPUT_FIELDS, 'frame')}
        if 'terminal_after_action' in data.files:
            result['terminal_after_action'] = data['terminal_after_action']
    return result


def train_data(origins, pure, candidate, jobs, summary):
    if len(origins) != 3 or summary.get('status') != 'SAVED_SEQUENCE_AUDIT_PASSED':
        raise ValueError('Require the three original training arrays and saved sequence audit')
    sources = catalog(candidate)
    ids = {r['relative']:i for i,r in sources.items()}
    owned = [candidate['original_jobs'], candidate['original_jobs'], candidate['new_collection_jobs'], jobs]
    parts, windows, offset = [], {'old': [], 'pure': []}, 0
    for origin, (raw, collection) in enumerate(zip([*origins, pure], owned)):
        if len({j['relative'] for j in collection}) != len(collection) or any(
                j['relative'] not in ids or any(j[k] != sources[ids[j['relative']]][k]
                    for k in ('split', 'group_id')) for j in collection):
            raise ValueError('Registered collection source groups or splits changed')
        data = causal_records(raw)
        mapping = {i:ids[j['relative']] for i,j in enumerate(collection) if j['split'] == 'train'}
        if set(map(int, data['clip_id'])) != set(mapping):
            raise ValueError('Training source missing, added or reassigned across splits')
        frame, clip = np.asarray(raw['frame']), data['clip_id']
        if (frame.shape != clip.shape or not np.issubdtype(frame.dtype, np.integer)
                or len(set(zip(map(int, clip), map(int, frame)))) != len(frame)):
            raise ValueError('Preserve unique integer training source/frame identities')
        part = {k:v.copy() for k,v in data.items()}
        part['clip_id'] = np.asarray([mapping[int(i)] for i in clip], dtype=np.int64)
        part['origin_id'] = np.full(len(clip), origin, dtype=np.int64)
        rows = [r for r in summary['records'] if r['origin'] == origin and r['split'] == 'train']
        if sum(r['records'] for r in rows) != len(frame) or len({r['clip_id'] for r in rows}) != len(rows):
            raise ValueError('Saved training sequence coverage changed')
        lookup = {(int(c), int(f)):i for i,(c,f) in enumerate(zip(clip, frame))}
        for row in rows:
            local_id = row['clip_id']
            if (local_id not in mapping or row['relative'] != sources[mapping[local_id]]['relative']
                    or row['records'] != int(np.count_nonzero(clip == local_id))):
                raise ValueError('Saved sequence crossed a training source mapping')
            for start in row['four_label_starts']:
                indices = np.asarray([lookup[(local_id, start+2*i)] for i in range(4)], dtype=np.int64)
                if (not np.array_equal(frame[indices], start+np.arange(4)*2)
                        or row['failure_frame'] is not None and int(frame[indices[-1]])+1 >= row['failure_frame']
                        or 'terminal_after_action' in raw and np.any(raw['terminal_after_action'][indices])):
                    raise ValueError('Sequence crosses actual time or terminal boundary')
                windows['old' if origin < 3 else 'pure'].append(indices+(offset if origin < 3 else 0))
        parts.append(part)
        if origin < 3:
            offset += len(clip)
    old = {k:np.concatenate([p[k] for p in parts[:3]]) for k in (*INPUT_FIELDS, 'origin_id')}
    pure = parts[3]
    expected = {i for i,r in sources.items() if r['split'] == 'train'}
    if set(map(int, old['clip_id'])) != expected or any(sources[int(i)]['split'] != 'train' for i in pure['clip_id']):
        raise ValueError('Training coverage differs from the registered source catalog')
    return old, pure, sources, {k:np.asarray(v, dtype=np.int64).reshape(-1, 4) for k,v in windows.items()}


def group_mean(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['group_id'], {}).setdefault(row['source_clip_id'], []).append(row['action_mse'])
    if not groups:
        raise ValueError('Cannot average an absent distribution')
    return float(np.mean([np.mean([np.mean(v) for v in clips.values()]) for clips in groups.values()]))


def gradient_record(losses, parameters, displacements, retention_weight, increment_weight):
    vectors, values = {}, {}
    active = [(k,v) for k,v in losses.items() if v is not None]
    for index, (name, loss) in enumerate(active):
        gradient = torch.autograd.grad(loss, parameters, retain_graph=index < len(active)-1)
        vector = torch.cat([g.detach().flatten().cpu().double() for g in gradient])
        if not torch.isfinite(vector).all() or not torch.isfinite(loss):
            raise ValueError('Nonfinite diagnostic objective or gradient')
        vectors[name], values[name] = vector, float(loss.detach().cpu())
    vectors['point_total'] = .5*(vectors['old_point']+vectors['pure_point'])+retention_weight*vectors['retention']
    vectors['delta_total'] = vectors['point_total']+increment_weight*vectors['increment']
    names = list(losses)+['point_total', 'delta_total']
    norms = {k:float(torch.linalg.vector_norm(v)) for k,v in vectors.items()}
    gram = [[float(torch.dot(vectors[a], vectors[b])) if a in vectors and b in vectors else None
        for b in names] for a in names]
    cosine = [[(gram[i][j]/(norms[a]*norms[b]) if a in norms and b in norms
        and min(norms[a], norms[b]) > 1e-15 else None) for j,b in enumerate(names)] for i,a in enumerate(names)]
    return dict(objectives=names, losses={k:values.get(k) for k in losses},
        gradient_norms={k:norms.get(k) for k in names}, gram=gram, cosine=cosine,
        zero_or_absent_gradient_cosine=None, autograd_calls=len(active),
        dot_actual_final_displacement={arm:{k:float(torch.dot(v, d)) for k,v in vectors.items()}
            for arm,d in displacements.items()})


def audit(native, payloads, origins, pure, candidate, jobs, summary, settings, deadline_ns):
    if list(payloads) != ['initial', 'retained_point_last', 'retained_delta_last']:
        raise ValueError('Only the registered actual initial and two last policies are inspected')
    old, pure, sources, windows = train_data(origins, pure, candidate, jobs, summary)
    rng, schedules, digest = np.random.default_rng(settings['seed']), [], hashlib.sha256()
    for index in range(settings['minibatches']):
        pair = [sample_windows(data, windows[kind], sources, rng, settings['windows_per_distribution'],
            index*settings['windows_per_distribution']) for kind,data in (('old', old), ('pure', pure))]
        for indices in pair:
            digest.update(indices.astype('<i8', copy=False).tobytes())
        schedules.append(pair)
    reference = payloads['initial']
    initial_vector = torch.cat([v.flatten().double() for v in reference['adapter'].values()])
    displacements = {name:torch.cat([v.flatten().double() for v in p['adapter'].values()])-initial_vector
        for name,p in payloads.items() if name != 'initial'}
    counts = dict(student_forward_batches=0, student_forward_rows=0, autograd_calls=0)
    teacher_sha, references, reports = module_sha(native.policy), [], []
    for name,payload in payloads.items():
        student = warm_student(native, payload).eval()
        device, present = next(student.parameters()).device, native.present
        before = {k:module_sha(getattr(student, k)) for k in ('encoder', 'decoder', 'adapter')}
        def forward(history, goal):
            if (time.monotonic_ns() >= deadline_ns-10_000_000_000
                    or counts['student_forward_batches'] >= settings['maximum_forward_batches']
                    or counts['student_forward_rows']+len(history) > settings['maximum_forward_rows']):
                raise RuntimeError('Bounded read-only objective audit exhausted its registered allowance')
            counts['student_forward_batches'] += 1
            counts['student_forward_rows'] += len(history)
            return student(history, goal)
        old_risk, old_rows = evaluate_cells(forward, old, sources, present, device)
        pure_risk, pure_rows = evaluate_cells(forward, pure, sources, present, device)
        matched = [r for r in old_rows if r['state_origin'] < 2 and r['source_clip_id'] in set(pure['clip_id'])]
        old_matched = group_mean(matched)
        metrics = dict(full_old_group_mse=old_risk, pure_group_mse=pure_risk,
            matched_old_group_mse=old_matched, combined_matched_pure_mse=.5*(old_matched+pure_risk),
            origin_group_mse={str(i):group_mean([r for r in old_rows if r['state_origin'] == i]) for i in range(3)},
            fixed_r1_source_mse=float(np.mean([r['action_mse'] for r in old_rows if r['state_origin'] == 0])),
            fixed_mixed_source_mse=float(np.mean([r['action_mse'] for r in old_rows if r['state_origin'] == 1])))
        probes = []
        for index, (old_indices, pure_indices) in enumerate(schedules):
            predictions, targets = [], []
            for data,indices in ((old, old_indices), (pure, pure_indices)):
                goal, history, target = batch(data, indices.reshape(-1), device)
                predictions.append(forward(history, goal).reshape(-1, 4, 29))
                targets.append(target.reshape(-1, 4, 29))
            a,b = predictions
            if name == 'initial':
                references.append(a.detach().clone())
            losses = dict(old_point=action_distillation_loss(a.reshape(-1, 29), targets[0].reshape(-1, 29), present))
            origin_ids = old['origin_id'][old_indices.reshape(-1)]
            for origin in range(3):
                take = torch.as_tensor(origin_ids == origin, device=device)
                losses['old_origin_'+str(origin)] = (action_distillation_loss(a.reshape(-1, 29)[take],
                    targets[0].reshape(-1, 29)[take], present) if bool(take.any()) else None)
            losses.update(pure_point=action_distillation_loss(b.reshape(-1, 29), targets[1].reshape(-1, 29), present),
                retention=(a[..., present]-references[index][..., present]).square().mean(),
                increment=increment_loss(torch.cat(predictions), torch.cat(targets), present))
            record = gradient_record(losses, list(student.trainable_parameters()),
                displacements if name == 'initial' else {}, settings['retention_weight'], settings['increment_weight'])
            counts['autograd_calls'] += record['autograd_calls']
            if counts['autograd_calls'] > settings['maximum_autograd_calls']:
                raise RuntimeError('Autograd allowance exhausted')
            probes.append(dict(minibatch=index, **record))
        if (before != {k:module_sha(getattr(student,k)) for k in before}
                or teacher_sha != module_sha(native.policy)
                or any(p.grad is not None for p in student.parameters())):
            raise ValueError('Read-only audit changed weights or accumulated parameter gradients')
        reports.append(dict(policy=name, train_metrics=metrics, train_cells=old_rows+pure_rows, gradient_probes=probes,
            policy_weights_unchanged=True, gradients_wrt='four adapter tensors only'))
    return dict(status='TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED', policies=reports,
        counts=counts, training_records=dict(old=len(old['clip_id']), pure=len(pure['clip_id'])),
        training_windows={k:len(v) for k,v in windows.items()}, schedule_sha256=digest.hexdigest(),
        schedule_scope='first registered four minibatches of completed retained-sequence fitting',
        validation_arrays_decoded=0, validation_gradients=0, test_sources_used=False, optimizer_updates=0,
        new_physical_steps=0, new_rendered_frames=0, human_real_future_frames=0,
        policy_inputs=['f0', 'history'], future_state_used_as_model_input=False,
        first_order_dots_are_local_approximations=True, causal_mechanism_proven=False,
        natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None, user_accepted_checkpoint=None)
