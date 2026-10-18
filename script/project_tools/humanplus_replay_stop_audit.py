"""D01 stopped-receipt audit with derived counts; never deserialize stopped PT."""

import json

from script.a3 import humanplus_replay_loss as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_head_protocol import digest, transactions
from script.a3.humanplus_replay_protocol import forward_accounting, validate_receipt


def main():
    plan = trial.checked()
    output = trial.PLAN.with_name(trial.NAME + '_stop_verification.json')
    if output.exists():
        raise FileExistsError('Stopped receipt audit must not repeat')
    result = trial.read(trial.RESULT)
    health_audit = trial.read(trial.PLAN.with_name(trial.NAME + '_train_audit.json'))
    directory = trial.LOG / 'train'
    head = trial.read(directory / 'head_contract.json')
    if head['owner_sha256'] != digest(head['ownership']):
        raise ValueError('Head ownership digest changed')
    rows = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    ledger = transactions(rows, head['owner_sha256'])
    commits, rejected = ledger['commits'], ledger['rejected']
    count, proposals = len(commits), ledger['proposals']
    stop = trial.read(directory / 'finite_stop.json')
    head_stop = trial.read(directory / 'head_stop.json')
    if (result['status'] != 'STOPPED' or result['stages'] or health_audit['verified']
            or health_audit['after_issues'] or health_audit['remaining_own_group_processes']
            or health_audit['fatal_matches'] or health_audit['oom_matches'] or rejected is None
            or stop['proposals'] != proposals or stop['committed'] != count
            or stop['committed_before_failure'] or not stop['uncommitted_batch_buffers_restored']
            or not head_stop['pending'] or head_stop['committed']
            or head_stop['owner_sha256'] != head['owner_sha256']
            or not head_stop['frozen_actor_unchanged']
            or head_stop['before_buffer_sha256'] != head_stop['after_buffer_sha256']):
        raise ValueError('Unexpected scientific stop, rollback or cleanup evidence')
    console = (directory / 'native_console.log').read_text()
    if 'CandidateRejected: All fixed finite candidates rejected' not in console:
        raise ValueError('No explicit native CandidateRejected evidence')
    if trial.read(directory / 'native_exit.json')['returncode'] == 0:
        raise ValueError('Scientific rejection cannot claim successful native exit')
    routes = trial.read(directory / 'reference_route_coverage.json')['rollouts']
    optimizer = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    finite_path = directory / 'finite_contract.json'
    finite = trial.read(finite_path) if finite_path.exists() else dict(committed=0, rows=[], evaluations=0)
    cloth = trial.read(directory / 'cloth_contract.json')
    batches = trial.read(directory / 'sampled_token_contract.json')['batches']
    rollouts = (proposals + 3) // 4
    if (len(routes) != rollouts or [r['update'] for r in routes] != list(range(1, rollouts + 1))
            or any(r['transitions'] != 384 or sum(r['routes'].values()) != 384 for r in routes)
            or [r['attempt_optimizer_step'] for r in optimizer] != list(range(1, count + 1))
            or finite['committed'] != count or finite['rows'] != commits
            or finite['evaluations'] + rejected['evaluations'] != ledger['evaluations']
            or len(cloth['optimizer_steps']) != count or len(batches) != proposals
            or any(not r['decoder_gradient_enabled'] for r in batches)
            or [r['commit'] for r in head['steps']] != list(range(1, count + 1))
            or any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in cloth['optimizer_steps'])):
        raise ValueError('Derived budgets or teacher evidence inconsistent')
    for row, commit in zip(head['steps'], commits):
        if (row['gradients'] != commit['head_gradients'] or row['owner_sha256'] != head['owner_sha256']
                or row['proposal'] != row['commit'] or not row['frozen_actor_unchanged']):
            raise ValueError('Head boundary receipts inconsistent')
    saved = {}
    complete_updates = count // 4
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        if step <= complete_updates and step != 8:
            receipt = trial.read(path.with_suffix('.json'))
            owners = trial.read(path.parent / 'head_owners.json')
            if (receipt['bytes'] != path.stat().st_size or receipt['sha256'] != sha256(path)
                    or owners['checkpoint_sha256'] != receipt['sha256']
                    or owners['ownership'] != head['ownership'] or owners['owner_sha256'] != head['owner_sha256']):
                raise ValueError('Saved evidence checkpoint identity changed')
            saved[str(step)] = receipt
        elif path.exists():
            raise ValueError('Unreached checkpoint exists')
    if any((trial.OUT / stage).exists() for stage in ('cloth', 'official')):
        raise ValueError('Prohibited stopped effect evaluation exists')
    replay = trial.read(directory / 'replay_loss_contract.json')
    if replay['design_sha256'] != sha256(trial.DESIGN):
        raise ValueError('Stopped replay design changed')
    validate_receipt(replay, trial.read(trial.DESIGN), proposals, rows)
    proof = dict(status='VERIFIED_STOPPED_HEAD_RECEIPTS_NOT_EFFECT_ACCEPTANCE',
        at_utc=trial.runtime.stamp(), experiment=plan['experiment'], optimizer_proposals=proposals,
        optimizer_commits=count, complete_updates=complete_updates,
        partial_update_commits=count % 4, sampled_transitions=rollouts * 384,
        sampled_rollouts=rollouts, candidate_evaluations=ledger['candidate_checks'],
        complete_native_candidate_checks=ledger['evaluations'],
        derived_outer_actor_decoder_calls=forward_accounting(rollouts, proposals, count, ledger['evaluations']),
        call_count_basis='PINNED_SOURCE_PATHS_AND_COMPLETED_JSON_RECEIPTS_NOT_KERNEL_INSTRUMENTATION',
        rejected_live_parameter_optimizer_digest_unchanged=True,
        uncommitted_batch_buffers_restored=True, checkpoints=saved, final_step8_exists=False,
        final_effect_rollouts=0, tensors_deserialized=0, machine_fault_evidenced=False,
        policy_improvement_proven=False, no_resume_no_budget_extension_no_early_evaluation=True,
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        train_audit_sha256=sha256(trial.PLAN.with_name(trial.NAME + '_train_audit.json')),
        auditor_sha256=sha256(trial.ROOT / 'script/project_tools/humanplus_head_stop_audit.py'),
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(output, proof)
    print(json.dumps({k: v for k, v in proof.items() if k != 'checkpoints'}, indent=2))


if __name__ == '__main__':
    main()

