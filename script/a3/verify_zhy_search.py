"""Rebuild E02 causal tokens and recompute rollout metrics independently."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.joint_forecast_replay import load_predictor
from script.a3.reference_contract import load_sim
from script.a3.zhy_search import OUT, OLD, read
from script.a3.zhy_trial import health


def verify_causal(directory, source, sigma, model, permutation, sim):
    name = Path(source['path']).stem
    with np.load(directory / f'{name}.inputs.npz', allow_pickle=False) as stored:
        observations, actions = stored['observations'].copy(), stored['actions'].copy()
    with np.load(directory / f'{name}.joint.npz', allow_pickle=False) as stored:
        arrivals, anchors = stored['rows'].copy(), stored['robot_anchor_quat_wxyz'].copy()
    trace = read(directory / f'{name}.timeseries.json')['motions'][0]
    metric = read(directory / f'{name}.metrics.json')['motions'][0]
    _, quaternion, q, _ = sim.load_a3_flat_csv(Path(source['path']), source_fps=30, frame_stride=4)
    times = np.arange(len(q))/30
    resampled = np.arange(0., times[-1]-1e-12, .02)
    clean = np.column_stack([np.interp(resampled, times, q[:, j]) for j in range(29)])[:, permutation]
    observed = q+np.random.default_rng(0).normal(0, sigma, q.shape)
    n = len(observations)
    if observations.shape != (len(clean), 1570) or actions.shape != (n, 29) or anchors.shape != (n, 4):
        raise ValueError('Actual ABI/full duration/anchor shape mismatch')
    if not all(np.isfinite(v).all() for v in (observations, actions, anchors, arrivals)):
        raise ValueError('Nonfinite policy inputs')
    np.testing.assert_allclose(actions, trace['raw_action_29'], atol=1e-6, rtol=1e-6)
    maximum, bases = 0., []
    for tick in range(n):
        native = tick*3//5
        base = native*5//3
        if native >= len(q)-1:
            raise ValueError('Unexpected EOF: separate contract required')
        bases.append(base)
        current = observed[native]
        if native < 5:
            nodes = np.repeat(current[None], 11, axis=0)
        else:
            velocity = (current-observed[native-1])*30
            future = current+np.arange(1, 11)[:, None]/50*velocity
            if model is not None:
                x = (observed[native-5:native]-current).reshape(145)
                correction = ((x-model['mean'])/model['std']) @ model['weights']+model['bias']
                future += correction.reshape(10, 29)
            nodes = np.r_[current[None], future]
        support = np.r_[times[:native], times[native]+np.arange(11)/50]
        values = np.r_[observed[:native], nodes]
        requested = (base+np.arange(11))/50
        predicted = np.column_stack([np.interp(requested, support, values[:, j]) for j in range(29)])[:, permutation]
        token = np.r_[predicted[:10].reshape(-1), (np.diff(predicted, axis=0)*50).reshape(-1)]
        if native == 0:
            root = np.repeat(quaternion[:1], 10, axis=0)
        else:
            seen = Rotation.from_quat(quaternion[native-1:native+1, [1, 2, 3, 0]])
            past = requested[:10] <= times[native]+1e-10
            root = np.empty((10, 4))
            if past.any():
                root[past] = Slerp(times[native-1:native+1], seen)(np.minimum(requested[:10][past], times[native])).as_quat()[:, [3, 0, 1, 2]]
            if (~past).any():
                rate = (seen[-1]*seen[-2].inv()).as_rotvec()*30
                root[~past] = (Rotation.from_rotvec((requested[:10][~past]-times[native])[:, None]*rate)*seen[-1]).as_quat()[:, [3, 0, 1, 2]]
        expected = np.r_[token, sim.root_ori_diff_6d(anchors[tick], root).reshape(-1)].astype(np.float32)
        np.testing.assert_allclose(observations[tick, :640], expected, atol=2e-6, rtol=2e-6)
        expected_arrival = [tick/50, base/50, tick/50-base/50, native/30, native, 0, max(0., base/50+.2-native/30)]
        np.testing.assert_allclose(arrivals[tick], expected_arrival, atol=1e-10, rtol=0)
        maximum = max(maximum, float(np.abs(observations[tick, :640]-expected).max()))
    target = clean[np.asarray(bases)]
    np.testing.assert_allclose(trace['reference_q_29'], target, atol=1e-9, rtol=0)
    rmse = float(np.sqrt(np.mean((np.asarray(trace['q_state_29'])-target)**2)))
    if not math.isclose(rmse, metric['tracking']['all_29_rmse'], rel_tol=1e-9):
        raise ValueError('Independent source-based joint metric failed')
    return dict(condition=directory.name, motion=name, reference_tokens_verified=n,
                max_abs_difference=maximum, independent_joint_rmse_rad=rmse,
                inputs_sha256=sha256(directory / f'{name}.inputs.npz'))


def verify_metrics(directory):
    manifest = read(directory / 'explicit_manifest.json')
    results = []
    for source in manifest['protocol']['motions']:
        name = Path(source['path']).stem
        status = manifest['runs'][name]
        if status['status'] != 'passed' or sha256(directory / f'{name}.metrics.json') != status['metrics_sha256']:
            raise ValueError('Unverified or altered motion metric')
        if sha256(Path(source['path'])) != source['sha256']:
            raise ValueError('Source changed since evaluation')
        trace = read(directory / f'{name}.timeseries.json')['motions'][0]
        metric = read(directory / f'{name}.metrics.json')['motions'][0]
        n = metric['num_policy_steps']
        if not np.array_equal(trace['policy_tick'], np.arange(n)):
            raise ValueError('Missing policy steps')
        q, ref = np.asarray(trace['q_state_29']), np.asarray(trace['reference_q_29'])
        actions = np.asarray(trace['raw_action_29'])
        if q.shape != (n, 29) or ref.shape != q.shape or actions.shape != q.shape or not all(np.isfinite(v).all() for v in (q, ref, actions)):
            raise ValueError('Invalid rollout states or actions')
        rmse = float(np.sqrt(np.mean((q-ref)**2)))
        if not math.isclose(rmse, metric['tracking']['all_29_rmse'], rel_tol=1e-9):
            raise ValueError('Joint metric recomputation failed')
        root = np.linalg.norm(np.asarray(trace['sim_root_pos_w'])-np.asarray(trace['ref_root_pos_w']), axis=-1)
        if not np.isfinite(root).all():
            raise ValueError('Nonfinite root positions')
        np.testing.assert_allclose(root, trace['root_pos_error_m'], atol=1e-12, rtol=1e-10)
        published = next(r for r in read(directory / 'explicit_summary.json')['motions'] if Path(r['motion']).stem == name)
        if published['windows']['full_rollout']['policy_steps'] != n:
            raise ValueError('Published step count mismatch')
        for frame, sim_key, ref_key in (('world', 'sim_body_pos_w', 'ref_body_pos_w'),
                                       ('anchor_local', 'sim_body_pos_anchor', 'ref_body_pos_anchor')):
            error = np.linalg.norm(np.asarray(trace[sim_key])-np.asarray(trace[ref_key]), axis=-1)
            names = trace['body_names']
            groups = dict(torso=['torso_Link'], wrists=['left_wrist_yaw_Link', 'right_wrist_yaw_Link'],
                          legs=[b for b in names if any(t in b for t in ('hip_', 'knee_', 'ankle_'))])
            for group, bodies in groups.items():
                actual = float(error[:, [names.index(b) for b in bodies]].mean())
                expected = published['windows']['full_rollout'][f'{group}_{frame}_mean_m']
                if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError('Body metric recomputation failed')
        results.append(dict(condition=directory.name, motion=name, policy_steps=n,
                            metrics_sha256=sha256(directory / f'{name}.metrics.json'),
                            timeseries_sha256=sha256(directory / f'{name}.timeseries.json')))
    return results


def main():
    job, state = read(OUT / 'job.json'), read(OUT / 'state.json')
    if state['status'] != 'EVALUATION_COMPLETE' or health()['boot_id'] != job['boot_id']:
        raise RuntimeError('Independent verification requires completed same-boot runs')
    with np.load(OLD / 'training_windows.npz', allow_pickle=False) as stored:
        permutation = stored['permutation'].copy()
    sim = load_sim()
    metrics, tokens = [], []
    for task in job['tasks']:
        directory = OUT / task['key']
        metrics.extend(verify_metrics(directory))
        if not task['mode'] or task.get('smoke'):
            continue
        model = load_predictor(job['adapters'][f"{task['objective']}_200"]['path']) if 'objective' in task else None
        for source in job['diagnostic_motions']:
            tokens.append(verify_causal(directory, source, task['sigma'], model, permutation, sim))
    for objective in ('mse', 'latent'):
        for key in ('observations', 'actions'):
            with np.load(OUT / f'smoke_{objective}/001_walk_front_slow.inputs.npz', allow_pickle=False) as new:
                with np.load(OUT / 'smoke_cv/001_walk_front_slow.inputs.npz', allow_pickle=False) as old:
                    np.testing.assert_allclose(new[key], old[key], atol=1e-6, rtol=1e-6)
    result = dict(status='passed', experiment=job['experiment'], motions=len(metrics),
        full_motions=sum(not r['condition'].startswith('smoke') for r in metrics),
        actual_reference_tokens_verified=sum(r['reference_tokens_verified'] for r in tokens),
        zero_adapter_matches_matched_cv=True, metric_rows=metrics, token_rows=tokens,
        proprio_independently_rebuilt=False, initialization_causal_verified=False,
        scope='CAUSAL_REFERENCE_TOKENS_AND_JOINT_ROOT_BODY_METRICS_NOT_LIVE_OR_ROBOT_ACCEPTANCE',
        code_sha256=sha256(Path(__file__)), backup_status='LOCAL_ONLY')
    atomic_json(OUT / 'independent_verification.json', result)
    state.update(status='VERIFIED', verified_at=__import__('time').time())
    atomic_json(OUT / 'state.json', state)
    print(json.dumps({k:result[k] for k in ('status', 'motions', 'full_motions', 'actual_reference_tokens_verified')}))


if __name__ == '__main__':
    main()
