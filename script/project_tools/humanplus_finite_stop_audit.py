"""Audit the existing HP12 rejection receipts without loading stopped models."""

import json

from script.a3 import humanplus_finite as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.finite_retention_protocol import SCALES, accept


def main():
    plan = trial.checked()
    output = trial.PLAN.with_name(trial.NAME + '_stop_verification.json')
    if output.exists():
        raise FileExistsError('Existing stop audit must not repeat')
    result = trial.read(trial.RESULT)
    health_audit = trial.read(trial.PLAN.with_name(trial.NAME + '_train_audit.json'))
    directory = trial.LOG / 'train'
    stop = trial.read(directory / 'finite_stop.json')
    rows = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    commits = [r for r in rows if r['status'] == 'COMMITTED']
    pending = [r for r in rows if r['status'] == 'ACCEPTED_PENDING_COMMIT']
    rejected = rows[-1]
    if (result['status'] != 'STOPPED' or result['stages'] or health_audit['verified']
            or health_audit['after_issues'] or health_audit['remaining_own_group_processes']
            or health_audit['fatal_matches'] or health_audit['oom_matches']
            or len(rows) != 55 or len(commits) != 27 or len(pending) != 27
            or rejected['status'] != 'REJECTED_NO_LIVE_UPDATE' or rejected['proposal'] != 28
            or stop['proposals'] != 28 or stop['committed'] != 27
            or stop['committed_before_failure'] or not stop['uncommitted_batch_buffers_restored']):
        raise ValueError('Unexpected stop, transaction or cleanup evidence')
    for i, (decision, commit) in enumerate(zip(pending, commits), 1):
        if (rows[2*i-2] != decision or rows[2*i-1] != commit
                or decision['proposal'] != i or commit['proposal'] != i
                or decision['parameters_and_optimizer_now_sha256'] != decision['parameters_and_optimizer_before_sha256']
                or decision['candidates'] != commit['candidates']
                or [r['scale'] for r in commit['candidates']] != list(SCALES[:len(commit['candidates'])])
                or any(r['accepted'] for r in commit['candidates'][:-1])
                or not accept(commit['before']['objective'], commit['candidates'][-1])):
            raise ValueError('Committed transaction differs from fixed method')
    candidates = rejected['candidates']
    if ([r['scale'] for r in candidates] != list(SCALES)
            or any(r['accepted'] or accept(rejected['before']['objective'], r) for r in candidates)
            or rejected['parameters_and_optimizer_now_sha256'] != rejected['parameters_and_optimizer_before_sha256']):
        raise ValueError('Rejected proposal changed live parameters or violated scale budget')
    routes = trial.read(directory / 'reference_route_coverage.json')['rollouts']
    optimizer = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    finite = trial.read(directory / 'finite_contract.json')
    cloth = trial.read(directory / 'cloth_contract.json')
    batches = trial.read(directory / 'sampled_token_contract.json')['batches']
    if (len(routes) != 7 or any(sum(r['routes'].values()) != 384 for r in routes)
            or [r['attempt_optimizer_step'] for r in optimizer] != list(range(1, 28))
            or finite['committed'] != 27 or finite['rows'] != commits
            or len(cloth['optimizer_steps']) != 27 or len(batches) != 28
            or any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in cloth['optimizer_steps'])):
        raise ValueError('Actual budget or teacher receipts inconsistent')
    saved = {}
    for step in (0, 2, 4):
        path = trial.checkpoint(step)
        receipt = trial.read(path.with_suffix('.json'))
        if receipt['bytes'] != path.stat().st_size or receipt['sha256'] != sha256(path):
            raise ValueError('Saved evidence checkpoint identity mismatch')
        saved[str(step)] = receipt
    if trial.checkpoint(8).exists() or any((trial.OUT / stage).exists() for stage in ('cloth', 'official')):
        raise ValueError('Unreached checkpoint or prohibited early effect evaluation exists')
    proof = dict(status='VERIFIED_STOPPED_RECEIPTS_NOT_TRAINING_OR_EFFECT_ACCEPTANCE',
        at_utc=trial.runtime.stamp(), experiment=plan['experiment'], optimizer_proposals=28,
        optimizer_commits=27, complete_updates=6, partial_update=7, partial_update_commits=3,
        sampled_transitions=2688, sampled_rollouts=7,
        candidate_evaluations=sum(not r.get('zero_displacement', False) for row in commits + [rejected]
                                  for r in row['candidates']),
        complete_native_candidate_checks=finite['evaluations'] + rejected['evaluations'],
        last_live_teacher_rmse=rejected['before']['teacher_rmse'],
        smallest_candidate_teacher_rmse=candidates[-1]['teacher_rmse'],
        smallest_candidate_teacher_margin_above_limit=candidates[-1]['teacher_rmse'] - .05,
        every_rejected_candidate_improved_objective=all(r['objective'] <= rejected['before']['objective'] + 1e-7 for r in candidates),
        every_rejected_candidate_violated_rmse=all(r['teacher_rmse'] > .05 for r in candidates),
        every_rejected_candidate_passed_maxabs=all(r['teacher_max_abs'] <= .5 for r in candidates),
        rejected_live_parameter_optimizer_digest_unchanged=True,
        uncommitted_batch_buffers_restored=True, checkpoints=saved, final_step8_exists=False,
        final_effect_rollouts=0, standalone_model_verification_runs=0, tensors_deserialized=0,
        machine_fault_evidenced=False, policy_improvement_proven=False,
        no_resume_no_budget_extension_no_early_evaluation=True,
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        train_audit_sha256=sha256(trial.PLAN.with_name(trial.NAME + '_train_audit.json')),
        auditor_sha256=sha256(trial.ROOT / 'script/project_tools/humanplus_finite_stop_audit.py'),
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(output, proof)
    print(json.dumps({k: v for k, v in proof.items() if k != 'checkpoints'}, indent=2))


if __name__ == '__main__':
    main()
