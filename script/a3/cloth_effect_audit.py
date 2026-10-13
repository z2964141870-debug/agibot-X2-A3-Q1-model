"""E43 decomposes saved matched E42 outcomes without new model or rollout calls."""

import argparse
import time

from script.a3 import cloth_resource_trial as runtime
from script.a3.checkpoint_store import atomic_json, sha256

shared = runtime.shared
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_cached_20261013_E42'
NAMES = shared.ROOT / 'data/experiments/a3_zhy_smpl_joint_audit_20261013_E28/result.json'
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_effect_audit_20261013_E43'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_effect_audit_20261013_E43'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_effect_audit_20261013_E43/NO_TRAINING'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_effect_audit_20261013_E43.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_EFFECT_AUDIT_20261013_E43'
shared.TRIAL = 'NO_TRAINING'
shared.MODULE = 'script.a3.cloth_effect_audit'
shared.CODE += ('script/a3/cloth_effect_audit.py',)
shared.STAGES = {'verify-training': ('REGISTERED', 'COMPLETE')}


def directories(kind):
    return ((shared.BASELINE / 'evaluation' if kind == 'cloth' else shared.OLD / 'official_oracle'),
            PRIOR / ('evaluation_' + kind))


def register():
    state = shared.read(PRIOR / 'state.json')
    effect = shared.read(PRIOR / 'effect_verification.json')
    if state['status'] != 'COMPLETE' or effect['development_gate_passed'] or len(effect['rows']) != 10:
        raise ValueError('Requires complete rejected E42 matched development results')
    runtime.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment='A3_ZHY_CLOTH_CACHED_20261013_E42', finalized_at=time.time(),
        updates=0, optimizer_steps=0, model_loads=0, new_forward_calls=0,
        new_policy_rollouts=0, simulator_steps=0, cuda_allowed=False,
        main_variable='SAVED_MATCHED_COMMON_PREFALL_JOINT_GROUP_AND_INPUT_DECOMPOSITION',
        causal_intervention=False, prior_rows=effect['rows'],
        descriptive_joint_error_rad=.35, descriptive_sustained_frames=5,
        reference_arrays_atol=0., initial_input_atol=0., action_clip=20.,
        interpretation='DESCRIPTIVE_NOT_PROOF_OF_FAILURE_CAUSE_OR_NEXT_TRAINING_BENEFIT')
    files = [PRIOR / n for n in ('state.json', 'registration.json', 'effect_verification.json',
                                 'training_verification.json')]
    files.append(NAMES)
    for row in effect['rows']:
        for directory in directories(row['kind']):
            files += [directory / (row['motion'] + '.' + suffix) for suffix in
                      ('metrics.json', 'timeseries.json', 'inputs.npz')]
    job['input_sha256'].update({str(p.relative_to(shared.ROOT)): sha256(p) for p in files})
    atomic_json(shared.OUT / 'registration.json', job)


def first_run(mask, length):
    start = None
    for tick, active in enumerate(mask):
        start = tick if active and start is None else start if active else None
        if start is not None and tick - start + 1 >= length:
            return dict(start_tick=start, label_s=start / 50.,
                        post_step_state_s=(start + 1) / 50., confirmed_tick=tick)
    return None


def window_errors(old, new, indices, names):
    import numpy as np
    a, b = old[indices], new[indices]
    if not len(a):
        raise ValueError('Empty comparison window')
    old_mse, new_mse = np.mean(a * a, axis=0), np.mean(b * b, axis=0)
    groups = {}
    for group in ('legs', 'waist', 'arms'):
        selected = [i for i, r in enumerate(names) if r['group'] == group]
        groups[group] = dict(baseline_rmse_rad=float(np.sqrt(old_mse[selected].mean())),
            candidate_rmse_rad=float(np.sqrt(new_mse[selected].mean())),
            extra_total_mse_contribution=float((new_mse[selected] - old_mse[selected]).sum() / 29))
    joints = [dict(joint=r['joint'], group=r['group'], baseline_rmse_rad=float(np.sqrt(old_mse[i])),
        candidate_rmse_rad=float(np.sqrt(new_mse[i])),
        extra_total_mse_contribution=float((new_mse[i] - old_mse[i]) / 29)) for i, r in enumerate(names)]
    joints.sort(key=lambda r: r['extra_total_mse_contribution'], reverse=True)
    return dict(frames=len(a), baseline_rmse_rad=float(np.sqrt(old_mse.mean())),
        candidate_rmse_rad=float(np.sqrt(new_mse.mean())),
        ratio=float(np.sqrt(new_mse.mean() / old_mse.mean())) if old_mse.mean() else None,
        groups=groups, largest_extra_mse_joints=joints[:8])


def audit(job):
    import numpy as np
    # Exclusion uses the earlier fall tick, and sustained events report their start.
    if first_run([False, True, True, True], 3)['start_tick'] != 1:
        raise ValueError('Sustained-event control failed')
    toy = [{'joint':str(i), 'group':'legs' if i < 12 else 'waist' if i < 15 else 'arms'} for i in range(29)]
    a, b = np.ones((3, 29)), np.ones((3, 29))
    b[2] = 100.
    if window_errors(a, b, slice(0, 2), toy)['ratio'] != 1.:
        raise ValueError('Common prefall exclusion control failed')
    names = sorted(shared.read(NAMES)['motions'][0]['named_joints'], key=lambda r:r['index'])
    if len(names) != 29 or [r['index'] for r in names] != list(range(29)):
        raise ValueError('Saved 29-joint mapping changed')
    rows = []
    for prior in job['prior_rows']:
        traces, metrics, observations, raw = [], [], [], []
        for directory in directories(prior['kind']):
            traces.append(shared.read(directory / (prior['motion'] + '.timeseries.json'))['motions'][0])
            metrics.append(shared.read(directory / (prior['motion'] + '.metrics.json'))['motions'][0])
            with np.load(directory / (prior['motion'] + '.inputs.npz'), allow_pickle=False) as capture:
                observations.append(capture['observations'])
                raw.append(capture['actions'])
        n = prior['steps']
        errors, falls, clipping = [], [], []
        for index, (trace, metric) in enumerate(zip(traces, metrics)):
            ticks = np.asarray(trace['policy_tick'])
            q, target = np.asarray(trace['q_state_29']), np.asarray(trace['reference_q_29'])
            if (q.shape != (n, 29) or target.shape != (n, 29) or
                    not np.array_equal(ticks, np.arange(n)) or metric['num_policy_steps'] != n):
                raise ValueError('Saved matched coverage rejected')
            if (observations[index].ndim != 2 or observations[index].shape[1] != 1570 or
                    not len(observations[index]) or raw[index].shape != (len(observations[index]), 29)):
                raise ValueError('Saved capture ABI rejected')
            if not all(np.isfinite(v).all() for v in (q, target, observations[index], raw[index])):
                raise ValueError('Nonfinite saved arrays')
            expected = np.clip(raw[index].astype(np.float64), -job['action_clip'], job['action_clip'])
            recorded = np.asarray(trace['raw_action_29'])[:len(expected)]
            np.testing.assert_array_equal(expected, recorded)
            clipping.append(int((np.abs(raw[index]) > job['action_clip']).any(axis=1).sum()))
            falling = np.flatnonzero((np.asarray(trace['root_height']) < .45) |
                (np.abs(trace['root_roll_deg']) > 60.) | (np.abs(trace['root_pitch_deg']) > 60.))
            fall = int(falling[0]) if len(falling) else None
            if metric['fall_tick'] != fall or metric['fall'] != (fall is not None):
                raise ValueError('Saved fall reconstruction failed')
            falls.append(fall)
            errors.append(q - target)
            np.testing.assert_allclose(np.sqrt(np.mean(errors[-1] ** 2)),
                metric['tracking']['all_29_rmse'], rtol=0, atol=1e-12)
        initial_difference = float(np.max(np.abs(observations[0][0] - observations[1][0])))
        if initial_difference > job['initial_input_atol']:
            raise ValueError('Matched initial policy input changed')
        reference_keys = ('reference_q_29', 'ref_root_pos_w', 'ref_root_quat_wxyz',
                          'ref_body_pos_w', 'ref_body_pos_anchor')
        for key in reference_keys:
            np.testing.assert_array_equal(traces[0][key], traces[1][key])
        end = min([n] + [fall for fall in falls if fall is not None])
        full = window_errors(*errors, slice(None), names)
        pre = window_errors(*errors, slice(0, end), names)
        if abs(sum(g['extra_total_mse_contribution'] for g in pre['groups'].values()) -
                (pre['candidate_rmse_rad'] ** 2 - pre['baseline_rmse_rad'] ** 2)) > 1e-12:
            raise ValueError('Group error contributions do not add up')
        knee_events = {}
        for i, r in enumerate(names):
            if 'knee' in r['joint']:
                knee_events[r['joint']] = [first_run(np.abs(e[:end, i]) > job['descriptive_joint_error_rad'],
                    job['descriptive_sustained_frames']) for e in errors]
        root_world = [float(np.asarray(t['root_pos_error_m'])[:end].mean()) for t in traces]
        first_action = np.asarray(traces[1]['raw_action_29'][0]) - np.asarray(traces[0]['raw_action_29'][0])
        item = dict(kind=prior['kind'], motion=prior['motion'], full=full, common_prefall=pre,
            common_prefall_end_exclusive_tick=end, falls=falls, initial_policy_input_max_abs=initial_difference,
            reference_arrays_exact=True, captured_input_frames=[len(x) for x in observations],
            clipping_frames_in_capture=clipping, common_prefall_root_error_mean_m=root_world,
            knee_error_sustained_events_before_either_fall=knee_events,
            initial_same_state_action_max_abs=float(np.abs(first_action).max()),
            initial_same_state_action_largest_joints=[dict(joint=names[i]['joint'], delta=float(first_action[i]))
                for i in np.argsort(np.abs(first_action))[::-1][:5]])
        if prior['kind'] == 'cloth' and 'zzx' in prior['motion']:
            focus = np.arange(n) / 50.
            item['fixed_4_4_to_4_9_s'] = window_errors(*errors, (focus >= 4.4) & (focus <= 4.9), names)
        rows.append(item)
        atomic_json(shared.OUT / 'partial.json', dict(rows=rows))
    summaries = {}
    for kind in ('cloth', 'official'):
        selected = [r for r in rows if r['kind'] == kind]
        summaries[kind] = {window:{key:float(np.mean([r[window][key] for r in selected])) for key in
            ('baseline_rmse_rad', 'candidate_rmse_rad')} for window in ('full', 'common_prefall')}
        for value in summaries[kind].values():
            value['mean_rmse_ratio'] = value['candidate_rmse_rad'] / value['baseline_rmse_rad']
    shared.unchanged(job)
    atomic_json(shared.OUT / 'training_verification.json', dict(
        status='VERIFIED_SAVED_MATCHED_EFFECT_DECOMPOSITION', rows=rows, summaries=summaries,
        controls=2, matched_initial_inputs_and_reference_verified=True,
        total_matched_steps=sum(r['full']['frames'] for r in rows),
        optimizer_steps=0, model_loads=0, new_forward_calls=0, new_policy_rollouts=0,
        simulator_steps=0, fresh_final_test=False, deploy_allowed=False,
        scope=job['interpretation']))


shared.verify_training = audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='verify-training')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else runtime.run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
