"""Bounded saved-input diagnostics: no model, mocap payload, physics or fitting."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

from forecast_data import source_support
from stream_contract import DT_NS
from stream_events import EventSpec, EventTimeline


BASE = Path('/opt/data/private/ZHY')
DATA = BASE/'data/supermimic_for_q1'
ROOTS = (DATA/'Q1_LOW_LATENCY_CKPT32000_STUDENT_R1_20261107',
         DATA/'Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107')
SPLITS = ('train', 'validation')
BLOCKS = {'root_history': (0, 3), 'body_positions': (3, 72),
          'body_rotations': (72, 216), 'absolute_height': (216, 217),
          'relative_torso': (217, 223)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(record):
    require(isinstance(record.get('path'), str) and type(record.get('bytes')) is int
            and record['bytes'] > 0 and isinstance(record.get('sha256'), str)
            and len(record['sha256']) == 64 and set(record['sha256']) <= set('0123456789abcdef'),
            'Malformed saved input identity')
    require(identity(record['path']) == record, 'Saved input identity changed: '+record['path'])
    return record


def resources(initial):
    require(Path.cwd().resolve() == BASE and Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            == '90395f47-4035-4785-858e-b3d05c44a738', 'Wrong server root or boot')
    require(all(not Path('/proc', str(pid)).exists() for pid in (54868, 55672, 55677, 55678, 55679)),
            'Original paused training PID is present')
    require(read(DATA/'MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104/e1024/train/guard.json')['status']
            == 'STOPPED_SIGNAL', 'Original training is no longer in its explicit paused state')
    cg = Path('/sys/fs/cgroup/memory')
    total, limit = (int((cg/name).read_text()) for name in ('memory.usage_in_bytes', 'memory.limit_in_bytes'))
    oom = dict(line.split() for line in (cg/'memory.oom_control').read_text().splitlines())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,temperature.gpu',
                                  '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
    require(gpu[0].strip() == 'GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4'
            and int(gpu[1]) < 18*1024 and int(gpu[2]) < 80, 'GPU identity/resource boundary failed')
    require(total < (22 if initial else 24)*(1 << 30) and limit == 34292629504
            and oom.get('oom_kill') == '0' and oom.get('under_oom') == '0'
            and shutil.disk_usage(BASE).free >= 100*(1 << 30), 'Memory/OOM/disk boundary failed')
    return dict(memory_total=total, memory_limit=limit, oom=oom, gpu=gpu,
                disk_free=shutil.disk_usage(BASE).free)


def load_arrays(path, names):
    with np.load(path, allow_pickle=False) as source:
        require(set(names) <= set(source.files), 'Missing saved audit arrays')
        result = {name:source[name] for name in names}
    count = len(result['frame'])
    shapes = dict(f0=(count, 10, 226), history=(count, 930), teacher_action=(count, 29),
                  actual_state=(count, 79), clip_id=(count,), frame=(count,))
    require(0 < count <= 4096 and all(result[k].shape == v for k, v in shapes.items()),
            'Unexpected saved shapes or sample bound')
    require(all(np.isfinite(a).all() for a in result.values()), 'Nonfinite saved arrays')
    require(all(np.issubdtype(result[k].dtype, np.integer) for k in ('frame', 'clip_id')),
            'Noninteger source/sample identity')
    return result


def expected_flags(frame, frames, spec):
    timeline = EventTimeline(frames, EventSpec(**spec))
    control = frame*DT_NS
    eligible = [(p.capture_ns, max(p.capture_ns, source_support(p.source_index)['dependency_end_ns']),
                 p.arrival_ns) for p in timeline.packets if max(0, frame-19) <= p.sequence <= frame]
    result = []
    for target_frame in range(frame-9, frame+1):
        target = target_frame*DT_NS
        support = [capture for capture, dependency, arrival in eligible
                   if capture <= target and dependency <= target and max(arrival, dependency) <= control]
        capture = max(support) if support else None
        if capture is None or target-capture > 10*DT_NS:
            result.append((0., 0., 0.))
        else:
            result.append((1., float(capture == target), (target-capture)/1e9))
    return np.asarray(result)


def validate_rows(data, split, jobs, traces, origin):
    keys = set()
    valid, exact, age = np.moveaxis(data['f0'][..., 223:], -1, 0)
    require(np.isin(valid, (0., 1.)).all() and np.isin(exact, (0., 1.)).all()
            and np.all(valid[:, -1] == 1.) and np.all(exact <= valid)
            and np.all(age >= 0.) and np.all(age <= .200001)
            and np.all(data['f0'][valid == 0.] == 0.), 'Invalid history flags, age or missing geometry')
    for row, (clip_id, frame) in enumerate(zip(data['clip_id'], data['frame'])):
        clip_id, frame = int(clip_id), int(frame)
        require(0 <= clip_id < len(jobs) and (clip_id, frame) not in keys, 'Duplicate or unknown source row')
        keys.add((clip_id, frame))
        job, trace = jobs[clip_id], traces[clip_id]
        require(job['split'] == split and (10 if origin == 0 else 0) <= frame < job['maximum_steps']
                and frame % 2 == 0 and frame < len(trace['canonical_action']), 'Split or terminal boundary crossed')
        label = trace['canonical_action' if origin == 0 else 'teacher_action'][frame]
        require(np.array_equal(data['actual_state'][row], trace['actual'][frame])
                and np.allclose(data['teacher_action'][row], label, atol=2e-5, rtol=2e-5),
                'Saved state or privileged label does not match its actual physical trace')
        flags = expected_flags(frame, job['maximum_steps']+10, job['packet_event'])
        require(np.allclose(data['f0'][row, :, 223:], flags, atol=1e-8, rtol=1e-6),
                'Saved flags/ages violate the published packet/dependency clock')
    require(set(map(int, data['clip_id'])) == {i for i,j in enumerate(jobs) if j['split'] == split},
            'One collection source is missing')
    return dict(rows=len(keys), state_and_label_trace_match=True, saved_flags_clock_match=True,
                raw_pose_geometry_recomputed=False, live_clock_certified=False)


def normalization_metrics(values, statistics, train_min, train_max):
    geometry = values[..., :223][values[..., 223] == 1.].astype(np.float64)
    require(len(geometry) > 0, 'No valid history geometry')
    z = (geometry-np.asarray(statistics['mean']))/np.asarray(statistics['scale'])
    clipped = np.abs(z) > 10.
    outside = (geometry < train_min-1e-6) | (geometry > train_max+1e-6)
    valid = values[..., 223] == 1.
    ages = values[..., 225][valid]*1000
    feature_rates = clipped.mean(0)
    return dict(valid_history_fraction=float(valid.mean()),
        exact_among_valid_fraction=float(values[..., 224][valid].mean()),
        current_age_ms={'median':float(np.median(values[:, -1, 225])*1000),
                        'p95':float(np.percentile(values[:, -1, 225], 95)*1000),
                        'max':float(values[:, -1, 225].max()*1000)},
        history_age_p95_ms=float(np.percentile(ages, 95)),
        abs_z_p95=float(np.percentile(np.abs(z), 95)), abs_z_max=float(np.abs(z).max()),
        clipped_coordinate_fraction=float(clipped.mean()),
        history_rows_with_clipping_fraction=float(clipped.any(-1).mean()),
        outside_original_train_range_fraction=float(outside.mean()),
        blocks={k:dict(clipped_fraction=float(clipped[:, a:b].mean()),
                       outside_train_range_fraction=float(outside[:, a:b].mean()))
                for k,(a,b) in BLOCKS.items()},
        top_clipped_features=[dict(index=int(i), clipped_fraction=float(feature_rates[i]))
                              for i in np.argsort(-feature_rates, kind='stable')[:12]])


def normalized_input(data, statistics):
    f0 = data['f0'].astype(np.float32)
    z = np.clip((f0[..., :223]-np.asarray(statistics['mean'], dtype=np.float32)) /
                np.asarray(statistics['scale'], dtype=np.float32), -10., 10.)
    z[f0[..., 223] != 1.] = 0.
    human = np.concatenate((z, f0[..., 223:]), -1).reshape(len(f0), -1)
    # A declared distance proxy gives each actual decoder input block equal RMS weight.
    return np.concatenate((human/np.sqrt(2*human.shape[1]),
                           data['history']/np.sqrt(2*data['history'].shape[1])), -1)


def sampled_indices(data, count):
    return np.concatenate([indices[np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)]
        for source in np.unique(data['clip_id'])
        for indices in [np.flatnonzero(data['clip_id'] == source)]])


def recorded_residual(residual, split, source):
    rows = [row for row in residual['rows'] if row.get('split') == split
            and row.get('phase') == 'all' and row.get('source_clip_id') == int(source)]
    require(len(rows) == 1, 'Historical residual is missing or duplicates this source')
    return rows


def neighbor_proxy(origins, statistics, present):
    bank, labels, source_ids = [], [], []
    for data in origins:
        indices = sampled_indices(data['train'], 32)
        bank.append(normalized_input(data['train'], statistics)[indices])
        labels.append(data['train']['teacher_action'][indices][:, present])
        source_ids.append(data['train']['clip_id'][indices])
    bank, labels, source_ids = np.concatenate(bank), np.concatenate(labels), np.concatenate(source_ids)
    rows = []
    for origin, data in enumerate(origins):
        for split in SPLITS:
            query = sampled_indices(data[split], 8)
            inputs = normalized_input(data[split], statistics)
            for source in np.unique(data[split]['clip_id']):
                indices = query[data[split]['clip_id'][query] == source]
                candidates = np.flatnonzero(source_ids != source) if split == 'train' else np.arange(len(bank))
                distances, errors = [], []
                for i in indices:
                    ds = np.sum((bank[candidates]-inputs[i])**2, axis=1)
                    nearest = candidates[int(np.argmin(ds))]
                    distances.append(float(np.sqrt(ds.min())))
                    errors.append(float(np.mean((labels[nearest]-data[split]['teacher_action'][i, present])**2)))
                rows.append(dict(origin=origin, split=split, source_clip_id=int(source), queries=len(indices),
                    input_distance_median=float(np.median(distances)),
                    nearest_label_mse_mean=float(np.mean(errors))))
    return dict(training_bank_rows=len(bank), rows=rows, train_queries_leave_whole_source_out=True,
        metric='equal RMS weights for normalized human history and normalized robot history',
        interpretation='finite-sample coverage/label proxy; not causality, a trained policy, or an irreducible error bound')


def execute(args):
    started = time.monotonic()
    require(args.output == DATA/'Q1_LOW_LATENCY_CKPT32000_INPUT_AUDIT01_20261107'
            and not args.output.exists(), 'Use the fresh independent saved-data audit root')
    admitted = resources(True)
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    require(identity(args.manifest)['sha256'] == args.manifest_sha256
            and len(args.evaluation_commit) == 40, 'Unpublished audit manifest or commit')
    code = [checked(row) for row in read(args.manifest)['files']]
    plan = read(args.plan)
    records = [checked(row) for row in plan['inputs']]
    require(plan['schema'] == 'q1_saved_causal_input_coverage_audit_v1'
            and sum(r['bytes'] for r in records) < 100_000_000, 'Audit source or input budget changed')
    split = read(BASE/'data/manifests/q1_teleop_forecast_split_20261106.json')
    statistics = read(DATA/'Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107/current_statistics.json')
    require(statistics['fitted_split'] == 'train' and statistics['clipping'] == 10.
            and np.all(np.asarray(statistics['scale']) >= .01), 'Use original frozen train-only normalization')
    origins, collections, jobs, inputs = [], [], None, list(records)+code+[identity(args.manifest)]
    for origin, root in enumerate(ROOTS):
        current_jobs = read(root/'collection_plan.json')['jobs']
        require(jobs is None or current_jobs == jobs, 'State origins have different source jobs')
        jobs = current_jobs
        collection = read(root/'collection_result.json')
        require(collection['status'] == 'PASSED' and collection['test_sources_read'] is False,
                'Use complete saved actual-state collections only')
        traces = {}
        for clip in collection['clips']:
            record = checked(clip['trace'])
            require(Path(record['path']) == root/('collection_%02d.npz' % clip['clip_id']), 'Trace path escaped its source')
            with np.load(record['path'], allow_pickle=False) as trace:
                traces[clip['clip_id']] = {name:trace[name] for name in
                    (('actual', 'canonical_action') if origin == 0 else ('actual', 'canonical_action', 'teacher_action'))}
            inputs.append(record)
        data = {}
        receipt = read(root/'dataset_receipt.json')
        for name in SPLITS:
            require(receipt['files'][name] in records, 'Dataset is absent from registered audit inputs')
            data[name] = load_arrays(root/(name+'_data.npz'),
                ('f0', 'history', 'teacher_action', 'actual_state', 'frame', 'clip_id'))
        checks = {name:validate_rows(data[name], name, jobs, traces, origin) for name in SPLITS}
        origins.append(data)
        collections.append(dict(origin=origin, checks=checks, clips=collection['clips']))
    require(len(inputs) < 100 and sum(r['bytes'] for r in inputs) < 100_000_000, 'Trace/input budget exceeded')
    ownership = {r['relative']:r for r in split['clips']}
    require(len(ownership) == len(split['clips']) and all(j['relative'] in ownership
        and all(j[k] == ownership[j['relative']][k] for k in ('split', 'group_id', 'human')) for j in jobs),
        'Collection jobs differ from original complete-source ownership')
    train = origins[0]['train']['f0']
    valid_train = train[..., :223][train[..., 223] == 1.]
    lower, upper = valid_train.min(0), valid_train.max(0)
    coverage = {}
    for name in SPLITS:
        corpus = [r for r in split['clips'] if r['split'] == name]
        owned = [j for j in jobs if j['split'] == name]
        coverage[name] = dict(available_source_groups=len({r['group_id'] for r in corpus}),
            available_clips=len(corpus), collected_source_groups=len({j['group_id'] for j in owned}),
            collected_clips=len(owned), maximum_requested_steps_per_clip=max(j['maximum_steps'] for j in owned),
            uncollected_source_groups=sorted({r['group_id'] for r in corpus}-{j['group_id'] for j in owned}),
            corpus_frames=sum(r['frames'] for r in corpus), requested_steps=sum(j['maximum_steps'] for j in owned))
    residual = read(BASE/'logs/session_records/q1_student_state_residuals_saved_20261107.json')
    require(residual['stage'] == 'SAVED_AGGREGATE_ACTION_RESIDUALS' and residual['status'] == 'PASSED',
            'Reuse the previously completed residual diagnosis')
    rows = []
    for origin, data in enumerate(origins):
        for name in SPLITS:
            values = data[name]
            for source in np.unique(values['clip_id']):
                indices = np.flatnonzero(values['clip_id'] == source)
                job = jobs[int(source)]
                metric = normalization_metrics(values['f0'][indices], statistics, lower, upper)
                saved = None if origin == 0 else recorded_residual(residual, name, source)
                rows.append(dict(origin=origin, split=name, source_clip_id=int(source), group_id=job['group_id'],
                    relative=job['relative'], samples=len(indices), first_frame=int(values['frame'][indices].min()),
                    last_frame=int(values['frame'][indices].max()), first_200ms_samples=int(np.sum(values['frame'][indices] < 10)),
                    first_second_samples=int(np.sum(values['frame'][indices] < 50)), metrics=metric,
                    historical_action_residual=saved))
    result = dict(stage='SAVED_CAUSAL_INPUT_COVERAGE_AUDIT', status='PASSED', utc=datetime.now(timezone.utc).isoformat(),
        evaluation_commit=args.evaluation_commit, resources_before=admitted, resources_after=resources(False),
        source_coverage=coverage, source_rows=rows, collection_checks=collections,
        nearest_neighbor_proxy=neighbor_proxy(origins, statistics, plan['present_joint_indices']),
        source_inputs=inputs, source_integrity='PASSED', normalization_refitted=False,
        new_model_inference_calls=0, new_physical_transitions=0, actual_optimizer_updates=0,
        raw_human_or_test_payload_read=False, existing_action_residuals_recomputed=False,
        sparse_coverage_is_proven_failure_cause=False, hardware_latency_ms=None,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None,
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    require(all(identity(r['path']) == r for r in inputs), 'Audit sources changed during reading')
    require(time.monotonic()-started < 280 and result['peak_rss_bytes'] < 1 << 30, 'CPU audit budget exceeded')
    args.output.mkdir(exist_ok=False)
    path = args.output/'input_audit.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    require(path.stat().st_size < 64 << 20, 'Audit output bound exceeded')
    print(json.dumps(dict(status=result['status'], output=identity(path),
                          elapsed_seconds=result['elapsed_seconds'], coverage=coverage)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--evaluation-commit', required=True)
    execute(parser.parse_args())
