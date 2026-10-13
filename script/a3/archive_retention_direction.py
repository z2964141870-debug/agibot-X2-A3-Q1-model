"""One-shot, standard-library-only E46 artifact identity and completion receipt."""

import json
from pathlib import Path

from script.a3.checkpoint_store import atomic_json, sha256


def main():
    root = Path(__file__).resolve().parents[2]
    if root != Path('/opt/data/private/hechengzhu/ZHY'):
        raise RuntimeError('Archive original ZHY evidence only')
    out = root / 'data/experiments/a3_zhy_retention_direction_20261013_E46'
    log = root / 'logs/a3_zhy_retention_direction_20261013_E46'
    target = root / 'data/manifests/a3_retention_direction_result_files_20261013_E46.json'
    if target.exists():
        raise FileExistsError('Preserve original archive receipt')
    read = lambda p: json.loads(p.read_text())
    state, job, result = (read(out / name) for name in ('state.json', 'registration.json', 'result.json'))
    manifest = root / 'data/manifests/a3_zhy_retention_direction_20261013_E46.json'
    receipt = read(manifest)
    if state['status'] != 'COMPLETE' or receipt['status'] != 'COMPLETE' or receipt['result'] != result:
        raise ValueError('Incomplete or inconsistent diagnostic record')
    if result['counters'] != dict(policy_states=2, actor_forwards=12, autograd_grad_calls=10):
        raise ValueError('Unexpected actual diagnostic coverage')
    for row in result['rows']:
        for a, b in (('actor_state_before_sha256', 'actor_state_after_sha256'),
                     ('optimizer_before_sha256', 'optimizer_after_sha256'),
                     ('inputs_before_sha256', 'inputs_after_sha256')):
            if row[a] != row[b]:
                raise ValueError('Unexpected mutation')
        if not row['gradient_sum']['passed']:
            raise ValueError('Invalid objective decomposition')
    for name, expected in {**job['input_sha256'], **job['code_sha256']}.items():
        if sha256(root / name) != expected:
            raise ValueError('Original registration identity changed: ' + name)
    files = [out / n for n in ('registration.json', 'state.json', 'result.json', 'progress.json',
                              'step0_direction.json', 'step4_direction.json')]
    files += [manifest, log / 'health.jsonl', log / 'entry_preflight.jsonl',
              log / 'probe_entry_preflight.jsonl', log / 'probe/console.log']
    rows = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha256(p)) for p in files]
    atomic_json(target, dict(experiment=job['experiment'], status='COMPLETE_ARCHIVED',
        registered_head=job['project_head'], files=rows, health=receipt['health'],
        counters=result['counters'], all_registered_inputs_and_code_unchanged=True,
        main_answer=result['main_answer'], baseline_improvement_proven=False,
        backup_status='LOCAL_ONLY'))
    print(json.dumps(dict(status='COMPLETE_ARCHIVED', files=len(rows))))


if __name__ == '__main__':
    main()
