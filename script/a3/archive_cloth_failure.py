"""Archive E45's stopped branch without loading tensors or evaluating a policy."""

from collections import Counter
import json
from pathlib import Path
import re

from script.a3.checkpoint_store import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45'
LOG = ROOT / 'logs/a3_zhy_cloth_failure_20261013_E45/training'
RUN = ROOT / 'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0'
RECEIPT = ROOT / 'data/manifests/a3_zhy_cloth_failure_stop_20261013_E45.json'


def read(path):
    return json.loads(path.read_text())


def section(text, key):
    """Compare generated YAML sections as text; do not interpret arbitrary YAML."""
    lines = text.splitlines(keepends=True)
    start = lines.index(key + ':\n')
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r'^[A-Za-z_][A-Za-z_0-9]*:', lines[i])), len(lines))
    return ''.join(lines[start:end])


def main():
    if ROOT != Path('/opt/data/private/hechengzhu/ZHY'):
        raise RuntimeError('Archive only the actual ZHY evidence')
    if RECEIPT.exists():
        raise FileExistsError('Preserve the original stop receipt')
    state, job = read(OUT / 'state.json'), read(OUT / 'registration.json')
    if state['status'] != 'STOPPED':
        raise ValueError('Expected stopped E45; no live experiment archival')
    if (OUT / 'training_verification.json').exists() or list(OUT.glob('checkpoint_*')):
        raise ValueError('Unexpected verification/evaluation after stop')
    contract = read(LOG / 'cloth_contract.json')
    audit = [json.loads(line) for line in (LOG / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    steps = contract['optimizer_steps']
    if (len(audit) != len(steps) or not steps or
            [r['attempt_optimizer_step'] for r in audit] != list(range(1, len(steps) + 1))):
        raise ValueError('Optimizer evidence is incomplete')
    message = 'Official action anchor drift budget exceeded; stop without retry'
    if (message not in (LOG / 'console.log').read_text() or
            not (steps[-1]['teacher_action_rmse'] > .05 or steps[-1]['teacher_action_max_abs'] > .5)):
        raise ValueError('Teacher stop reason not established')
    batches = read(LOG / 'sampled_token_contract.json')['batches']
    if len(batches) != len(steps):
        raise ValueError('Actual batch coverage differs')
    routes = read(LOG / 'reference_route_coverage.json')['rollouts']
    totals = Counter()
    for row in routes:
        totals.update(row['routes'])
    checkpoints = []
    for path in sorted(RUN.glob('milestone_*/*.json')):
        record = read(path)
        model = path.parent / record['file']
        if model.stat().st_size != record['bytes'] or sha256(model) != record['sha256']:
            raise ValueError('Saved file identity changed')
        checkpoints.append(dict(record, path=str(model.relative_to(ROOT)),
                                verification_scope='BYTES_SHA_ONLY_NO_MODEL_LOAD'))
    old_path = ROOT / 'data/training/a3_zhy_cloth_curve_20261013_E44/R18_001_s0/config.yaml'
    old, new = old_path.read_text(), (RUN / 'config.yaml').read_text()
    old_env, new_env = section(old, 'manager_env'), section(new, 'manager_env')
    new_env = new_env.replace('a3_zhy_cloth_failure_20261013_E45', 'a3_zhy_cloth_curve_20261013_E44')
    recipe = dict(algo_exact_text=section(old, 'algo') == section(new, 'algo'),
                  manager_env_exact_except_motion_directory=old_env == new_env,
                  interpretation='STATIC_GENERATED_CONFIG_COMPARISON_NOT_TRAINING_VERIFICATION')
    paths = [OUT / n for n in ('registration.json', 'state.json', 'prepare.json', 'training_launch.json')]
    paths += [LOG / n for n in ('cloth_contract.json', 'reference_route_coverage.json',
              'sampled_token_contract.json', 'actual_optimizer_steps.jsonl', 'console.log')]
    paths += [RUN / 'config.yaml', Path(__file__).resolve()]
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha256(p)) for p in paths]
    cumulative = read(ROOT / 'data/manifests/a3_zhy_cloth_failure_20261013_E45.json')
    atomic_json(RECEIPT, dict(experiment=job['experiment'], trial=job['trial'],
        status='STOPPED_ORIGINAL_TEACHER_DRIFT_PROTECTION', controller_state=state,
        observed_failure=message, completed_full_updates=audit[-1]['trainer_global_step'],
        partial_update=audit[-1]['trainer_global_step'] + 1,
        completed_optimizer_steps=len(audit), sampled_transitions=sum(totals.values()),
        route_totals=dict(totals), loaded_training_sequences=job['training_sequences'],
        per_motion_sample_counts_recorded=False, development_roles=job['development_roles'],
        initial_probability=contract['initial_probability'],
        initial_anchor_max_abs=contract['initial_anchor_max_abs'],
        optimizer_steps=steps, actual_optimizer_steps=audit,
        frozen_digest_unchanged=all(r['frozen_parameter_digest'] == contract['frozen_parameter_digest'] for r in steps),
        config_comparison=recipe, health=cumulative['health'], files=rows,
        saved_checkpoints=checkpoints, requested_effect_checkpoints=job['effect_checkpoints'],
        new_effect_rollouts=0, training_verification_not_run=True, model_effect_not_measured=True,
        backup_status='LOCAL_ONLY', baseline='official035', no_auto_resume=True,
        corpus_branch_closed=True,
        next_action='SEPARATE_RETENTION_OPTIMIZATION_CONFLICT_STUDY_NO_MORE_CORPUS_OR_EXPOSURE',
        conclusion_boundary='Adding the known failing development motion did not avoid the same original teacher stop; closed-loop benefit and gradient conflict are not measured.'))
    print(json.dumps(dict(receipt=str(RECEIPT.relative_to(ROOT)), optimizer_steps=len(audit),
        checkpoint_steps=[r['step'] for r in checkpoints], config_comparison=recipe)))


if __name__ == '__main__':
    main()
