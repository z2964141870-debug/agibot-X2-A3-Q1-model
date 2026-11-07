"""Plan broader owned-source collection, reusing every original sampled prefix."""

import argparse
import hashlib
import json
from pathlib import Path

from student_rollouts import SPLIT_SHA, collection_jobs


def build(plan, previous):
    rows = {r['relative']:r for r in plan['clips']}
    if len(rows) != len(plan['clips']):
        raise ValueError('Repeated corpus source')
    ownership, old = {}, {}
    for row in rows.values():
        if row['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown source ownership')
        prior = ownership.setdefault(row['group_id'], row['split'])
        if prior != row['split']:
            raise ValueError('Source group crosses splits')
    for job in previous:
        row = rows[job['relative']]
        if (job['split'] not in ('train', 'validation') or job['split'] != row['split']
                or job['group_id'] != row['group_id'] or job['human'] != row['human']):
            raise ValueError('Prior source does not match the original owned corpus')
        if job['group_id'] in old:
            raise ValueError('Original collector must have exactly one clip per selected group')
        old[job['group_id']] = job
    selected, new_jobs, exclusions = [], [], []
    for split in ('train', 'validation'):
        groups = sorted({r['group_id'] for r in rows.values() if r['split'] == split})
        for group in groups:
            available = [r for r in rows.values() if r['group_id'] == group and r['frames'] >= 40]
            available.sort(key=lambda r:hashlib.sha256(('2026110704:'+r['relative']).encode()).hexdigest())
            chosen = []
            if group in old:
                chosen.append(rows[old[group]['relative']])
            chosen.extend(r for r in available if r['relative'] not in {x['relative'] for x in chosen})
            chosen = chosen[:2]
            if not chosen:
                exclusions.append(dict(split=split, group_id=group, reason='no clip with at least40 frames'))
            for row in chosen:
                reused = group in old and row['relative'] == old[group]['relative']
                selected.append(dict(split=split, group_id=group, relative=row['relative'],
                                     reused_original_prefix=reused))
                if reused:
                    continue
                maximum = min(384, row['frames']-10)
                index = len(new_jobs)
                new_jobs.append(dict(split=split, group_id=group, relative=row['relative'],
                    clip=Path(row['relative']).stem, human=row['human'], maximum_steps=maximum,
                    record_stride=2, record_from_first_available_frame=True,
                    packet_event=dict(kind=('nominal', 'dropout', 'late_burst')[index % 3],
                        first_frame=max(1, min(maximum-3, int(maximum*.4))),
                        duration_frames=min(8, maximum-3), transport_frames=index % 3),
                    physical_reference_event='nominal', seed=0))
    if (len({r['relative'] for r in selected}) != len(selected)
            or len({Path(r['relative']).stem for r in selected}) != len(selected)):
        raise ValueError('Ambiguous source path or policy library clip name')
    return dict(schema='q1_source_coverage_candidate_plan_v1', status='METADATA_PLAN_ONLY',
        source_split_sha256=SPLIT_SHA, source_split_unchanged=True,
        proposed_single_main_variable='broader source/trajectory coverage at frozen normalization and decoder',
        original_jobs=previous, selected_sources=selected, new_collection_jobs=new_jobs,
        excluded_groups=exclusions,
        coverage={split:dict(available_groups=len({r['group_id'] for r in rows.values() if r['split'] == split}),
            planned_groups=len({r['group_id'] for r in selected if r['split'] == split}),
            planned_clips=sum(r['split'] == split for r in selected),
            new_clips=sum(r['split'] == split for r in new_jobs)) for split in ('train', 'validation')},
        maximum_new_physical_steps=sum(j['maximum_steps'] for j in new_jobs),
        maximum_planned_new_records=sum((j['maximum_steps']+1)//2 for j in new_jobs),
        reused_original_datasets=['R1 actual teacher states', 'AGGREGATE01 actual mixed states'],
        original_sampled_prefixes_replayed=False, raw_human_payload_read=False, test_payload_read=False,
        proposed_teacher_iteration=32000, proposed_human_real_future_frames=0,
        proposed_normalization='original R2 train-only statistics frozen',
        proposed_frozen_modules=['original teacher', 'shared decoder', 'world model'],
        proposed_fit=dict(warm_start='original R2 selected encoder', learning_rate=.0001,
            maximum_updates=3000, batch_size=128, validation_every_updates=500,
            selection_split='validation', selection_includes_update0=True,
            weighting='equal source groups, then equal source clips and state origins where available'),
        required_before_dispatch=['INPUT_AUDIT01 actual PASSED result and input SHA verification',
            'evidence-based decision after reviewing normalization and coverage',
            'independent collector/worker/fit implementation and runtime contracts',
            'registered source and resource admission, published code and nonblocking original GPU lock',
            'original cost ledger reservation and remaining first-round budget'],
        proposed_phase_seconds=1740, original_phase_protection_seconds=1800,
        proposed_output_bytes=1073741824, first_round_gpu_seconds_limit=14400,
        prior_settled_gpu_seconds=1606, prior_r3_settled_gpu_seconds=485,
        new_training_updates=0, new_physical_transitions=0, training_dispatched=False,
        causal_failure_explanation_proven=False, user_accepted_checkpoint=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.split.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_SHA:
        raise ValueError('Use the original unchanged complete-source split')
    plan = json.loads(raw)
    result = build(plan, collection_jobs(plan))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status', 'coverage', 'maximum_new_physical_steps',
        'maximum_planned_new_records', 'training_dispatched')}, indent=2))
