"""Retain old outputs in both arms; isolate teacher action-increment matching."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import batch, validation
from student_coverage_fit import evaluate_cells
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss
from student_onpolicy_pair_fit import eligible, warm_student
from student_retained_sequence_data import sample_windows, sequence_data


ARMS = ('retained_point', 'retained_delta')


def increment_loss(prediction, target, present):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (4, 29):
        raise ValueError('Use four same-source labels spaced40ms apart')
    error = (prediction[:, 1:]-prediction[:, :-1])-(target[:, 1:]-target[:, :-1])
    return error[..., present].square().mean()


def fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, output, deadline_ns, provenance):
    required = {'seed', 'maximum_updates', 'windows_per_distribution', 'learning_rate',
        'gradient_norm_limit', 'validation_every_updates', 'retention_weight', 'increment_weight'}
    if (set(settings) != required or settings['maximum_updates'] < 1
            or settings['windows_per_distribution'] < 1 or settings['validation_every_updates'] < 1
            or settings['maximum_updates'] % settings['validation_every_updates']
            or not np.isfinite(list(settings.values())).all()
            or any(settings[k] <= 0 for k in ('learning_rate', 'gradient_norm_limit', 'retention_weight', 'increment_weight'))):
        raise ValueError('Require finite fixed paired sequence settings')
    old, matched, current, sources, windows = sequence_data(origins, pure, candidate, jobs, summary)
    teacher_sha, decoder_sha = module_sha(native.policy), module_sha(native.policy.decoder)
    if teacher_sha != payload['frozen_teacher_policy_sha256'] or decoder_sha != payload['decoder_sha256']:
        raise ValueError('Warm student belongs to a different teacher')
    reference = warm_student(native, payload).eval().requires_grad_(False)
    device = next(reference.parameters()).device
    present = native.present.to(device)
    with torch.no_grad():
        retained = torch.cat([reference(h, g) for g, h, _ in
            (batch(old['train'], np.arange(i, min(i+128, len(old['train']['clip_id']))), device)
             for i in range(0, len(old['train']['clip_id']), 128))]).detach()
    retained_sha, runs = module_sha(reference.adapter), []
    for arm in ARMS:
        folder = Path(output)/arm
        folder.mkdir(exist_ok=False)
        torch.manual_seed(settings['seed'])
        rng = np.random.default_rng(settings['seed'])
        student = warm_student(native, payload).train()
        encoder_sha, initial_sha = module_sha(student.encoder), module_sha(student.adapter)
        optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings['learning_rate'])
        trace, initial_anchors, best, digest, exposures = [], None, None, hashlib.sha256(), {}
        started = time.monotonic_ns()

        def evaluate(update, metrics=None):
            nonlocal initial_anchors, best
            old_risk, _ = evaluate_cells(student, old['validation'], sources, present, device)
            matched_risk, _ = evaluate_cells(student, matched['validation'], sources, present, device)
            pure_risk, pure_rows = evaluate_cells(student, current['validation'], sources, present, device)
            anchors = [old_risk]+[validation(student, x['validation'], present, device)[0] for x in origins[:2]]
            risk = .5*(matched_risk+pure_risk)
            if not np.isfinite([risk, *anchors]).all():
                raise ValueError('Nonfinite retained-sequence validation')
            if initial_anchors is None:
                initial_anchors = anchors
            allowed = eligible(initial_anchors, anchors)
            if best is None or allowed and risk < best['risk']:
                best = dict(update=update, risk=risk, anchors=anchors, state=cpu_state(student.adapter))
            trace.append(dict(update=update, validation_risk=risk, old_matched_validation_risk=matched_risk,
                pure_validation_risk=pure_risk, pure_validation=pure_rows, full_old_validation_risk=old_risk,
                fixed_anchor_r1_risk=anchors[1], fixed_anchor_mixed_risk=anchors[2],
                fixed_anchor_non_regression=allowed, metrics=metrics,
                schedule_sha256=digest.hexdigest(), source_sample_counts=dict(exposures),
                elapsed_seconds=(time.monotonic_ns()-started)/1e9))
            save_json(folder/'training_trace.json', trace)
            save_json(Path(output)/'training_progress.json', dict(arm=arm, **trace[-1]))

        evaluate(0)
        for update in range(1, settings['maximum_updates']+1):
            if time.monotonic_ns() >= deadline_ns-30_000_000_000:
                raise RuntimeError('Do not truncate one paired arm to meet wall time')
            count = settings['windows_per_distribution']
            chosen = [sample_windows(data['train'], windows[kind]['train'], sources, rng, count, (update-1)*count)
                for kind, data in (('old', old), ('pure', current))]
            values, predictions = [], []
            for kind, data, indices in zip(('old', 'pure'), (old, current), chosen):
                digest.update(indices.astype('<i8', copy=False).tobytes())
                for clip in data['train']['clip_id'][indices[:, 0]]:
                    key = kind+':'+str(int(clip))
                    exposures[key] = exposures.get(key, 0)+1
                goal, history, target = batch(data['train'], indices.reshape(-1), device)
                values.append(target.reshape(count, 4, 29))
                predictions.append(student(history, goal).reshape(count, 4, 29))
            point = action_distillation_loss(torch.cat(predictions).reshape(-1, 29),
                torch.cat(values).reshape(-1, 29), present)
            initial = retained[torch.as_tensor(chosen[0], device=device)]
            retention = (predictions[0][..., present]-initial[..., present]).square().mean()
            delta = increment_loss(torch.cat(predictions), torch.cat(values), present)
            loss = point+settings['retention_weight']*retention
            if arm == 'retained_delta':
                loss = loss+settings['increment_weight']*delta
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
                settings['gradient_norm_limit'], error_if_nonfinite=True)
            optimizer.step()
            if update % settings['validation_every_updates'] == 0:
                evaluate(update, dict(loss=float(loss.detach().cpu()), point_loss=float(point.detach().cpu()),
                    retention_loss=float(retention.detach().cpu()), increment_loss=float(delta.detach().cpu()),
                    gradient_norm=float(gradient.detach().cpu())))
        counters = [int(v['step'].item()) for v in optimizer.state.values()]
        if len(counters) != 4 or counters != [settings['maximum_updates']]*4:
            raise ValueError('Four residual Adam counters must match actual updates')
        if (module_sha(native.policy) != teacher_sha or module_sha(student.encoder) != encoder_sha
                or module_sha(student.decoder) != decoder_sha or module_sha(reference.adapter) != retained_sha):
            raise ValueError('Frozen teacher, base or retention reference changed')
        from upper_mps_forecast import audit_cpu_tensors, cpu_tree
        common = dict(schema='q1_f0_retained_sequence_v1', variant=arm, arm='causal_history',
            interface='f0_plus_history', human_real_future_frames=0, history_frames=10, history_frame_dim=226,
            robot_history_features=930, adapter_hidden_width=128, residual_rule='z+(1-abs(z))*tanh(adapter(history))',
            current_statistics=payload['current_statistics'], base_encoder=cpu_state(student.encoder),
            base_encoder_sha256=encoder_sha, decoder_sha256=decoder_sha, frozen_teacher_policy_sha256=teacher_sha,
            initial_adapter_sha256=initial_sha, actual_updates=settings['maximum_updates'], actual_adam_steps=counters,
            selected_update=best['update'], selected_validation_risk=best['risk'],
            selected_fixed_anchor_risks=best['anchors'], initial_fixed_anchor_risks=initial_anchors,
            selection_split='validation', test_used_for_selection=False, normalization_refitted=False,
            provenance=provenance, sequence_label_spacing_ms=40, sequence_span_ms=120,
            saved_state_open_loop_fitting=True, new_physical_steps=0, user_accepted_checkpoint=None)
        for name, saved in (('selected.pt', dict(common, adapter=best['state'], checkpoint_update=best['update'])),
                ('last.pt', dict(common, adapter=cpu_state(student.adapter), checkpoint_update=settings['maximum_updates'],
                    optimizer=cpu_tree(torch, optimizer.state_dict())))):
            audit_cpu_tensors(torch, saved)
            with (folder/name).open('xb') as stream:
                torch.save(saved, stream)
            audit_cpu_tensors(torch, torch.load(folder/name, map_location='cpu', weights_only=True))
        result = {k:v for k,v in common.items() if k != 'base_encoder'}
        result.update(status='TRAINED_PENDING_CLOSED_LOOP', initial_validation_risk=trace[0]['validation_risk'],
            initial_pure_validation_risk=trace[0]['pure_validation_risk'], final_pure_validation_risk=trace[-1]['pure_validation_risk'],
            final_validation_risk=trace[-1]['validation_risk'],
            selected_adapter_unchanged=all(torch.equal(v, payload['adapter'][k]) for k,v in best['state'].items()),
            schedule_sha256=digest.hexdigest(), source_sample_counts=exposures,
            trainable_parameters=sum(p.numel() for p in student.trainable_parameters()),
            elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9)
        save_json(folder/'fit_result.json', result)
        runs.append(result)
    for key in ('initial_adapter_sha256', 'initial_validation_risk', 'initial_fixed_anchor_risks',
            'schedule_sha256', 'source_sample_counts', 'trainable_parameters'):
        if runs[0][key] != runs[1][key]:
            raise ValueError('Paired retention control differs: '+key)
    result = dict(status='RETAINED_SEQUENCE_PAIR_TRAINED_PENDING_CLOSED_LOOP', arms=runs,
        matched_initialization_capacity_and_source_slots=True, actual_optimizer_updates=2*settings['maximum_updates'],
        sequence_windows={k:{s:len(v) for s,v in values.items()} for k,values in windows.items()},
        retention_weight=settings['retention_weight'], increment_weight=settings['increment_weight'],
        old_records={s:len(v['clip_id']) for s,v in old.items()}, pure_records={s:len(v['clip_id']) for s,v in current.items()},
        new_physical_steps=0, new_rendered_frames=0, human_real_future_frames=0, test_sources_read=False,
        natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None, user_accepted_checkpoint=None)
    save_json(Path(output)/'fit_result.json', result)
    return result
