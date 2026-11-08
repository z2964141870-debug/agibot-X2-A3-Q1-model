"""Read saved source sequences without running a policy, physics or training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np

from student_input_audit import identity, read, require, resources


BASE = Path('/opt/data/private/ZHY')
DATA = BASE/'data/supermimic_for_q1'
RECOVERY = DATA/'Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_RESUME1_20261108'
OUTPUT = DATA/'Q1_LOW_LATENCY_SAVED_SEQUENCE_AUDIT01_20261108'
ORIGINS = ('STUDENT_R1_20261107', 'STUDENT_R2_AGGREGATE01_20261107',
           'STUDENT_R3_COVERAGE01_20261107', 'F0_ONPOLICY_LABELS01_20261108')
LEDGER = BASE/'logs/session_records/q1_low_latency_gpu_cost_20261107.json'
INPUT_FIELDS = ('f0', 'history')


def windows(frames, failure_frame, *, width=4, stride=2, input_ready=None):
    """Use recorded label times; exclude a terminating action, even at the end."""
    frames = np.asarray(frames)
    require(frames.ndim == 1 and np.issubdtype(frames.dtype, np.integer)
            and width >= 2 and stride >= 1 and len(np.unique(frames)) == len(frames)
            and np.all(frames >= 0), 'Invalid source-local sequence indices')
    require(failure_frame is None or type(failure_frame) is int and failure_frame > 0,
            'Invalid physical terminal boundary')
    lookup = {int(frame): i for i, frame in enumerate(frames)}
    result = []
    for start in sorted(lookup):
        wanted = [start+i*stride for i in range(width)]
        if (all(frame in lookup for frame in wanted)
                and (failure_frame is None or wanted[-1]+1 < failure_frame)
                and (input_ready is None or np.all(input_ready[start:wanted[-1]+1]))):
            result.append([lookup[frame] for frame in wanted])
    return np.asarray(result, dtype=np.int64).reshape(-1, width)


def rms(values):
    return None if not values.size else float(np.sqrt(np.mean(values.astype(np.float64)**2)))


def source_summary(values, trace, job, *, origin):
    frames = values['frame']
    count = len(frames)
    require(count > 0 and np.issubdtype(frames.dtype, np.integer)
            and np.all(frames % 2 == 0) and len(np.unique(frames)) == count,
            'Require unique saved even-frame labels')
    shapes = dict(f0=(count, 10, 226), history=(count, 930), teacher_action=(count, 29),
                  actual_state=(count, 79))
    require(all(values[k].shape == shape and np.isfinite(values[k]).all()
                for k, shape in shapes.items()), 'Invalid saved source arrays')
    steps = len(trace['canonical_action'])
    require(trace['canonical_action'].shape == (steps, 29)
            and trace['actual'].shape == (steps+1, 79)
            and np.all(frames < steps), 'Saved sample exceeds its physical trace')
    failure = int(trace['failure_frame']) if 'failure_frame' in trace else job['failure_frame']
    failure = None if failure is None or failure == -1 else failure
    require(failure is None or failure == steps, 'Trace continued after physical termination')
    require(np.array_equal(values['actual_state'], trace['actual'][frames]), 'Source state pairing changed')
    label = 'teacher_action' if origin in (1, 3) else 'canonical_action'
    require(trace[label].shape == (steps, 29)
            and np.allclose(values['teacher_action'], trace[label][frames], atol=2e-5, rtol=2e-5),
            'Privileged labels differ from their saved source trace')
    if 'input_ready' in trace:
        require(np.all(trace['input_ready'][frames]), 'A saved sequence contains unavailable input')
    valid, exact, age = (values['f0'][..., 223+i] for i in range(3))
    require(np.isin(valid, (0., 1.)).all() and np.isin(exact, (0., 1.)).all()
            and np.all(exact <= valid) and np.all(valid[:, -1] == 1.)
            and np.all(age[exact == 1.] == 0.)
            and np.all(age >= 0.) and np.all(age <= .200001)
            and np.all(values['f0'][valid == 0.] == 0.), 'Invalid saved causal visibility')
    require(set(INPUT_FIELDS).isdisjoint(('teacher_action', 'next_state', 'terminal_after_action')),
            'Privileged targets must not enter student inputs')
    terminal = np.asarray([failure == int(frame)+1 for frame in frames], dtype=bool)
    if 'next_state' in values:
        require(np.array_equal(values['next_state'], trace['actual'][frames+1])
                and values['terminal_after_action'].dtype == np.bool_
                and np.array_equal(values['terminal_after_action'], terminal),
                'Saved next state or terminal flag is not the actual 20ms transition')
    ready = trace.get('input_ready')
    require(ready is None or ready.shape == (steps,), 'Invalid physical input-availability trace')
    four = windows(frames, failure, input_ready=ready)
    adjacent = windows(frames, failure, width=2, input_ready=ready)
    order = np.argsort(frames, kind='stable')
    sorted_frames = frames[order]
    gaps = np.diff(sorted_frames)
    controls = trace['canonical_action'][frames]
    teacher_delta = values['teacher_action'][adjacent[:, 1]]-values['teacher_action'][adjacent[:, 0]]
    mismatch = values['teacher_action']-controls
    student_delta = controls[adjacent[:, 1]]-controls[adjacent[:, 0]]
    dot = np.sum(teacher_delta.astype(np.float64)*student_delta, axis=-1)
    norms = np.linalg.norm(teacher_delta, axis=-1)*np.linalg.norm(student_delta, axis=-1)
    supported = norms > 1e-8
    mismatch_terminal = mismatch[np.asarray([failure is not None and 0 < failure-int(f) <= 10
                                            for f in frames])]
    return dict(records=count, physical_steps=steps, failure_frame=failure,
        first_frame=int(sorted_frames[0]), last_frame=int(sorted_frames[-1]),
        label_stride_control_steps=2, label_spacing_ms=40, next_state_spacing_ms=20,
        adjacent_nonterminal_pairs=len(adjacent), four_label_windows=len(four),
        four_label_window_span_ms=120, gap_count=int(np.count_nonzero(gaps != 2)),
        gap_control_steps=[int(v) for v in gaps if v != 2],
        terminating_label_rows=int(terminal.sum()),
        four_label_starts=[int(frames[row[0]]) for row in four],
        teacher_delta_rms_normalized_action=rms(teacher_delta),
        teacher_behavior_mismatch_rms_normalized_action=rms(mismatch),
        teacher_behavior_mismatch_last_200ms_rms=rms(mismatch_terminal),
        delta_direction_supported_pairs=int(supported.sum()),
        opposite_action_delta_pairs=int(np.count_nonzero((dot < 0.) & supported)),
        opposite_deltas_are_not_proof_of_harm=True,
        state_and_label_trace_match=True, saved_visibility_flags_valid=True,
        raw_pose_reconstruction_performed=False)


def audit():
    started = time.monotonic()
    initial = resources(True)
    prior = read(RECOVERY/'result.json')
    guard = read(RECOVERY/'recovery.guard.json')
    require(prior['status'] == 'COMPLETED' and prior['source_integrity'] == 'PASSED'
            and prior['cost_settlement']['state'] == 'SETTLED'
            and guard['cleanup']['worker_reaped'] is True
            and guard['cleanup']['worker_return_code'] == 0,
            'Wait for original recovery completion and cleanup')
    ledger_before = identity(LEDGER)
    protected = read(RECOVERY/'authorization.json')['protected_inputs']
    require(len(protected) == 1172 and all(identity(r['path']) == r for r in protected),
            'A protected published source changed')
    ledger = read(LEDGER)
    require(not any(r['state'] == 'RESERVED' for r in ledger['runs']), 'Prior GPU reservation remains')
    controls = [identity(RECOVERY/n) for n in ('result.json', 'recovery.guard.json', 'authorization.json')]
    controls += [identity(Path(__file__).resolve())]
    output = []
    source_owners, group_owners = {}, {}
    for origin, suffix in enumerate(ORIGINS):
        root = DATA/('Q1_LOW_LATENCY_CKPT32000_'+suffix)
        collection, plan = read(root/'collection_result.json'), read(root/'collection_plan.json')
        require(collection.get('test_sources_read') is False and plan['test_sources_read'] is False,
                'Do not open held-out source datasets')
        clips = {r['clip_id']: r for r in collection['clips']}
        jobs = plan['jobs']
        require(set(clips) == set(range(len(jobs))), 'Source metadata is incomplete')
        for name in ('collection_result.json', 'collection_plan.json', 'result.json'):
            controls.append(identity(root/name))
        for split in ('train', 'validation'):
            path = root/(split+('_transitions.npz' if origin == 3 else '_data.npz'))
            controls.append(identity(path))
            with np.load(path, allow_pickle=False) as source:
                values = {k:source[k] for k in source.files
                          if k in ('f0', 'history', 'teacher_action', 'actual_state', 'clip_id',
                                   'frame', 'next_state', 'terminal_after_action')}
            ids = values['clip_id']
            require(np.issubdtype(ids.dtype, np.integer) and len(ids) <= 8192
                    and set(map(int, ids)) == {i for i, j in enumerate(jobs) if j['split'] == split},
                    'Owned source/split identities changed')
            for clip_id in sorted(set(map(int, ids))):
                job, clip = jobs[clip_id], clips[clip_id]
                require(job['split'] == clip['split'] == split and job['clip'] == clip['clip'],
                        'Collection source or split was relabeled')
                for field, owners in (('relative', source_owners), ('group_id', group_owners)):
                    key = job[field]
                    require(owners.setdefault(key, split) == split, 'A source group crosses splits')
                trace_record = clip['trace']
                require(identity(trace_record['path']) == trace_record, 'Saved physical trace changed')
                controls.append(trace_record)
                with np.load(trace_record['path'], allow_pickle=False) as saved:
                    trace = dict(saved)
                selected = ids == clip_id
                row = source_summary({k:v[selected] for k, v in values.items()}, trace, clip, origin=origin)
                output.append(dict(origin=origin, split=split, clip_id=clip_id, clip=job['clip'],
                    relative=job['relative'], group_id=job['group_id'], **row))
                require(time.monotonic()-started < 240, 'Saved audit240s body bound')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < (1 << 30), 'Saved audit1GiB RSS bound')
    require(all(identity(r['path']) == r for r in protected+controls)
            and identity(LEDGER) == ledger_before, 'Saved inputs or GPU cost ledger changed')
    final = resources(False)
    require(final['oom'] == initial['oom'], 'OOM counters changed during saved analysis')
    return dict(schema='q1_saved_sequence_audit_v1', status='SAVED_SEQUENCE_AUDIT_PASSED',
        observed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
        records=output, protected_sources_unchanged=len(protected), inputs=controls,
        source_group_splits_disjoint=True, sequence_width=4, sequence_label_stride=2,
        sequence_span_ms=120, student_input_fields=list(INPUT_FIELDS),
        teacher_future_available_only_as_target=True, raw_pose_reconstruction_performed=False,
        actual_new_policy_forwards=0, actual_new_physical_steps=0, actual_new_optimizer_updates=0,
        new_rendered_frames=0, gpu_cost_ledger_unchanged=ledger_before,
        resources_initial=initial, resources_final=final,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        test_sources_read=False, hardware_end_to_end_latency_ms=None,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    require(args.output == OUTPUT and not args.output.exists(), 'Saved audit output is single-use')
    args.output.mkdir(exist_ok=False)
    result = audit()
    path = args.output/'summary.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    require(path.stat().st_size < 64*(1 << 20), 'Saved output64MiB bound')
    print(json.dumps(dict(status=result['status'], sources=len(result['records']),
        elapsed_seconds=result['elapsed_seconds'], summary=identity(path)), allow_nan=False))
