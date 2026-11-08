"""Matched real-versus-zero human latent, with the original point/retention loss."""

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import batch, validation
from student_conditioning_models import ARMS, conditioned_student, zero_human_branch
from student_coverage_fit import evaluate_cells
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss
from student_onpolicy_pair_fit import eligible, warm_student
from student_retained_sequence_data import sample_windows, sequence_data


def initial_audit(native, payload, old, current, seed):
    reference = warm_student(native, payload).eval().requires_grad_(False)
    device = next(reference.parameters()).device
    values = [batch(data['train'], np.arange(min(16, len(data['train']['clip_id']))), device)
        for data in (old, current)]
    goal, history, target = (torch.cat(parts) for parts in zip(*values))
    cpu_reference = copy.deepcopy(reference).cpu()
    with torch.no_grad():
        expected = reference(history, goal)
        cpu_expected = cpu_reference(history.cpu(), goal.cpu())
    rows, states = [], []
    for arm in ARMS:
        torch.manual_seed(seed)
        student = conditioned_student(native, payload, arm).eval()
        before = module_sha(student)
        states.append(before)
        with torch.no_grad():
            prediction = student(history, goal)
            cpu_prediction = copy.deepcopy(student).cpu()(history.cpu(), goal.cpu())
        if not torch.equal(prediction, expected) or not torch.equal(cpu_prediction, cpu_expected):
            raise ValueError('The added zero-output branch changed the actual initial policy')
        difference = float((prediction.cpu()-cpu_prediction).abs().max())
        if not np.isfinite(difference) or difference > 2e-5:
            raise ValueError('Initial CPU/device action parity failed')
        parameters = tuple(student.trainable_parameters())
        loss = action_distillation_loss(student(history, goal), target, native.present.to(device))
        gradients = torch.autograd.grad(loss, parameters)
        if (len(gradients) != 8 or any(not torch.isfinite(g).all() for g in gradients)
                or module_sha(student) != before
                or any(p.requires_grad for m in (student.encoder, student.decoder) for p in m.parameters())):
            raise ValueError('Only eight residual tensors may receive finite diagnostic gradients')
        rows.append(dict(variant=arm, initial_model_sha256=before, initial_actions_exact_reference=True,
            cpu_actions_exact_reference=True, maximum_cpu_device_action_difference=difference,
            train_probe_rows=len(history), gradient_norms=[float(g.double().norm().cpu()) for g in gradients],
            actual_autograd_calls=1, optimizer_updates=0, human_real_future_frames=0))
    if states[0] != states[1]:
        raise ValueError('Paired conditioning modules have different initial weights')
    return dict(status='CONDITIONING_INITIAL_ACTION_PARITY_PASSED', arms=rows,
        training_only_probe=True, maximum_action_tolerance=2e-5, actual_autograd_calls=2,
        matched_initial_model_sha256=states[0], new_physical_steps=0)


def fit_pair(native, origins, pure, candidate, jobs, summary, payload, settings, output, deadline_ns, provenance):
    required = {'seed', 'maximum_updates', 'windows_per_distribution', 'learning_rate',
        'gradient_norm_limit', 'validation_every_updates', 'retention_weight'}
    if (set(settings) != required or settings['maximum_updates'] < 1
            or settings['windows_per_distribution'] < 1 or settings['validation_every_updates'] < 1
            or settings['maximum_updates'] % settings['validation_every_updates']
            or not np.isfinite(list(settings.values())).all()
            or any(settings[k] <= 0 for k in ('learning_rate', 'gradient_norm_limit', 'retention_weight'))):
        raise ValueError('Require the fixed point-plus-retention paired settings')
    if any((Path(output)/arm).exists() for arm in ARMS):
        raise FileExistsError('Conditioning training outputs are single-use')
    old, matched, current, sources, windows = sequence_data(origins, pure, candidate, jobs, summary)
    teacher_sha, decoder_sha = module_sha(native.policy), module_sha(native.policy.decoder)
    if teacher_sha != payload['frozen_teacher_policy_sha256'] or decoder_sha != payload['decoder_sha256']:
        raise ValueError('Initial student belongs to another frozen teacher')
    parity = initial_audit(native, payload, old, current, settings['seed'])
    save_json(Path(output)/'initial_action_audit.json', parity)
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
        student = conditioned_student(native, payload, arm).train()
        encoder_sha, initial_sha = module_sha(student.encoder), module_sha(student.adapter)
        human_initial_sha = module_sha(student.human_adapter)
        if module_sha(student) != parity['matched_initial_model_sha256']:
            raise ValueError('Training initialization differs from the actual action parity probe')
        parameters = tuple(student.trainable_parameters())
        optimizer = torch.optim.Adam(parameters, lr=settings['learning_rate'])
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
                raise ValueError('Nonfinite conditioning validation')
            if initial_anchors is None:
                initial_anchors = anchors
            allowed = eligible(initial_anchors, anchors)
            if best is None or allowed and risk < best['risk']:
                best = dict(update=update, risk=risk, anchors=anchors,
                    adapter=cpu_state(student.adapter), human_adapter=cpu_state(student.human_adapter))
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
                raise RuntimeError('Do not truncate one conditioning arm to meet wall time')
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
                values.append(target)
                predictions.append(student(history, goal).reshape(count, 4, 29))
            point = action_distillation_loss(torch.cat(predictions).reshape(-1, 29), torch.cat(values), present)
            initial = retained[torch.as_tensor(chosen[0], device=device)]
            retention = (predictions[0][..., present]-initial[..., present]).square().mean()
            loss = point+settings['retention_weight']*retention
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_norm_limit'], error_if_nonfinite=True)
            optimizer.step()
            if update % settings['validation_every_updates'] == 0:
                evaluate(update, dict(loss=float(loss.detach().cpu()), point_loss=float(point.detach().cpu()),
                    retention_loss=float(retention.detach().cpu()), gradient_norm=float(gradient.detach().cpu())))
        counters = [int(v['step'].item()) for v in optimizer.state.values()]
        if len(counters) != 8 or counters != [settings['maximum_updates']]*8:
            raise ValueError('Eight residual Adam counters must match actual updates')
        if (module_sha(native.policy) != teacher_sha or module_sha(student.encoder) != encoder_sha
                or module_sha(student.decoder) != decoder_sha or module_sha(reference.adapter) != retained_sha):
            raise ValueError('Frozen teacher, base or retention reference changed')
        from upper_mps_forecast import audit_cpu_tensors, cpu_tree
        common = dict(schema='q1_f0_human_conditioned_residual_v1', variant=arm, arm='causal_history',
            interface='f0_plus_history', human_real_future_frames=0, history_frames=10, history_frame_dim=226,
            robot_history_features=930, adapter_hidden_width=128, human_adapter_hidden_width=128,
            conditioning_latent_dim=native.policy.cfg.latent, human_conditioning='frozen_causal_latent' if arm == 'real_human' else 'zeros',
            residual_rule='z+(1-abs(z))*tanh(robot_adapter(history)+human_adapter(real_z_or_zero))',
            current_statistics=payload['current_statistics'], base_encoder=cpu_state(student.encoder),
            base_encoder_sha256=encoder_sha, decoder_sha256=decoder_sha, frozen_teacher_policy_sha256=teacher_sha,
            initial_adapter_sha256=initial_sha, initial_human_adapter_sha256=human_initial_sha,
            actual_updates=settings['maximum_updates'], actual_adam_steps=counters,
            selected_update=best['update'], selected_validation_risk=best['risk'],
            selected_fixed_anchor_risks=best['anchors'], initial_fixed_anchor_risks=initial_anchors,
            selection_split='validation', test_used_for_selection=False, normalization_refitted=False,
            provenance=provenance, sequence_label_spacing_ms=40, sequence_span_ms=120,
            saved_state_open_loop_fitting=True, new_physical_steps=0, user_accepted_checkpoint=None)
        for name, saved in (('selected.pt', dict(common, adapter=best['adapter'], human_adapter=best['human_adapter'], checkpoint_update=best['update'])),
                ('last.pt', dict(common, adapter=cpu_state(student.adapter), human_adapter=cpu_state(student.human_adapter),
                    checkpoint_update=settings['maximum_updates'], optimizer=cpu_tree(torch, optimizer.state_dict())))):
            audit_cpu_tensors(torch, saved)
            with (folder/name).open('xb') as stream:
                torch.save(saved, stream)
            audit_cpu_tensors(torch, torch.load(folder/name, map_location='cpu', weights_only=True))
        result = {k:v for k,v in common.items() if k != 'base_encoder'}
        result.update(status='TRAINED_PENDING_CLOSED_LOOP', initial_validation_risk=trace[0]['validation_risk'],
            initial_pure_validation_risk=trace[0]['pure_validation_risk'], final_pure_validation_risk=trace[-1]['pure_validation_risk'],
            final_validation_risk=trace[-1]['validation_risk'],
            selected_policy_unchanged=zero_human_branch(best['human_adapter'])
                and all(torch.equal(v, payload['adapter'][k]) for k,v in best['adapter'].items()),
            selected_human_branch_output_zero=zero_human_branch(best['human_adapter']),
            schedule_sha256=digest.hexdigest(), source_sample_counts=exposures,
            trainable_parameters=sum(p.numel() for p in parameters), trainable_tensors=len(parameters),
            elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9)
        save_json(folder/'fit_result.json', result)
        runs.append(result)
    for key in ('initial_adapter_sha256', 'initial_human_adapter_sha256', 'initial_validation_risk',
            'initial_fixed_anchor_risks', 'schedule_sha256', 'source_sample_counts', 'trainable_parameters', 'trainable_tensors'):
        if runs[0][key] != runs[1][key]:
            raise ValueError('Matched conditioning control differs: '+key)
    result = dict(status='HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP', arms=runs,
        matched_initialization_capacity_and_source_slots=True, actual_optimizer_updates=2*settings['maximum_updates'],
        initial_action_audit=parity, retention_weight=settings['retention_weight'],
        sequence_windows={k:{s:len(v) for s,v in values.items()} for k,values in windows.items()},
        old_records={s:len(v['clip_id']) for s,v in old.items()}, pure_records={s:len(v['clip_id']) for s,v in current.items()},
        new_physical_steps=0, new_rendered_frames=0, human_real_future_frames=0, test_sources_read=False,
        natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None, user_accepted_checkpoint=None)
    save_json(Path(output)/'fit_result.json', result)
    return result
