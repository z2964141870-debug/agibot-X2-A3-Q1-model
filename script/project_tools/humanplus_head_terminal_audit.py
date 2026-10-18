"""Audit HP13's broker-log failure from existing JSON and bytes, never saved tensors."""

import json
import re

from script.a3 import humanplus_head as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_head_protocol import digest, forward_accounting, transactions


def main():
    plan = trial.checked()
    target = trial.PLAN.with_name(trial.NAME + '_terminal_receipt_audit.json')
    if target.exists():
        raise FileExistsError('Terminal receipt audit is single-use')
    result = trial.read(trial.RESULT)
    health_path = trial.PLAN.with_name(trial.NAME + '_train_audit.json')
    health = trial.read(health_path)
    directory = trial.LOG / 'train'
    state = trial.read(directory / 'state.json')
    head = trial.read(directory / 'head_contract.json')
    rows = [json.loads(line) for line in (directory / 'finite_candidates.jsonl').read_text().splitlines()]
    ledger = transactions(rows, digest(head['ownership']))
    finite = trial.read(directory / 'finite_contract.json')
    cloth = trial.read(directory / 'cloth_contract.json')
    native = trial.read(directory / 'native_training.json')
    exit_receipt = trial.read(directory / 'native_exit.json')
    warm = trial.read(directory / 'warm_start_loaded.json')
    routes = trial.read(directory / 'reference_route_coverage.json')['rollouts']
    batches = trial.read(directory / 'sampled_token_contract.json')['batches']
    optimizer = [json.loads(line) for line in (directory / 'actual_optimizer_steps.jsonl').read_text().splitlines()]
    progress = [json.loads(line) for line in (directory / 'progress.jsonl').read_text().splitlines()]
    phases = [r['phase'] for r in progress]
    required = ['training_environment_created', 'training_complete', 'context_clear_returned', 'native_close_begin']
    broker_error = 'ValueError: Exactly one authorized native Kit log required'
    console = (directory / 'console.log').read_text()
    native_console = (directory / 'native_console.log').read_text()
    selected = re.findall(r'Logging to file: (/[^\s\x1b]+\.log)', native_console)
    expected_root = trial.ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs'
    if (result['status'] != 'STOPPED' or result['stages'] or state['status'] != 'STOPPED'
            or result['reason'] != 'RuntimeError:WORKER_EXIT:1' or broker_error not in console
            or exit_receipt['returncode'] != 0 or selected != ['/tmp/isaaclab/logs/isaaclab_2026-10-18_15-39-35.log']
            or health['verified'] or health['after_issues'] or health['remaining_own_group_processes']
            or health['fatal_matches'] or health['oom_matches']):
        raise ValueError('Unexpected terminal cause or unhealthy cleanup')
    if (head['owner_sha256'] != digest(head['ownership']) or len(head['steps']) != 32
            or ledger['rejected'] is not None or ledger['proposals'] != 32
            or finite['rows'] != ledger['commits'] or finite['proposals'] != 32 or finite['committed'] != 32
            or finite['evaluations'] != ledger['evaluations'] or ledger['evaluations'] != 64
            or len(cloth['optimizer_steps']) != 32 or len(batches) != 32
            or [r['attempt_optimizer_step'] for r in optimizer] != list(range(1, 33))
            or [r['commit'] for r in head['steps']] != list(range(1, 33))
            or any(not r['decoder_gradient_enabled'] for r in batches)
            or any(r['teacher_action_rmse'] > .05 or r['teacher_action_max_abs'] > .5 for r in cloth['optimizer_steps'])):
        raise ValueError('Recorded full-budget transaction coverage inconsistent')
    for row, commit in zip(head['steps'], ledger['commits']):
        if (row['proposal'] != row['commit'] or row['owner_sha256'] != head['owner_sha256']
                or row['gradients'] != commit['head_gradients'] or not row['frozen_actor_unchanged']):
            raise ValueError('Recorded head gradient/frozen protection inconsistent')
    if (len(routes) != 8 or [r['update'] for r in routes] != list(range(1, 9))
            or any(r['transitions'] != 384 or sum(r['routes'].values()) != 384 for r in routes)
            or native['status'] != 'TRAINING_COMPLETE' or native['global_step'] != 8
            or native['optimizer_commits'] != 32 or native['sampled_transitions'] != 3072
            or not native['cuda_synchronized'] or native['owner_sha256'] != head['owner_sha256']
            or any(phases.count(p) != 1 for p in required)
            or [phases.index(p) for p in required] != sorted(phases.index(p) for p in required)
            or warm['loaded_global_step'] != 0 or warm['optimizer_state_entries'] != 0
            or not all(warm['networks_match_source'].values())):
        raise ValueError('Recorded training/native lifecycle coverage inconsistent')
    saved = {}
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        receipt = trial.read(path.with_suffix('.json'))
        owners = trial.read(path.parent / 'head_owners.json')
        if (receipt['bytes'] != path.stat().st_size or receipt['sha256'] != sha256(path)
                or owners['checkpoint_sha256'] != receipt['sha256'] or owners['step'] != step
                or owners['ownership'] != head['ownership'] or owners['owner_sha256'] != head['owner_sha256']):
            raise ValueError('Evidence checkpoint bytes or owner sidecar changed')
        saved[str(step)] = dict(receipt, sidecar_sha256=sha256(path.parent / 'head_owners.json'))
    if (trial.OUT / 'training_verification.json').exists() or any(
            (trial.OUT / stage).exists() or (trial.LOG / stage).exists()
            for stage in ('verify-training', 'cloth', 'official', 'effect')):
        raise ValueError('Prohibited post-stop verification or effects exist')
    counts = forward_accounting(8, 32, 32, ledger['evaluations'])
    if native['derived_outer_actor_decoder_calls'] != counts:
        raise ValueError('Recorded call accounting inconsistent')
    proof = dict(status='VERIFIED_STOPPED_BROKER_LOG_RECEIPTS_NOT_MODEL_OR_EFFECT_ACCEPTANCE',
        at_utc=trial.runtime.stamp(), experiment=plan['experiment'],
        original_terminal_preserved=True, resume_or_rerun_allowed=False,
        broker_error=broker_error, native_returncode=0,
        console_log_paths=selected, required_kit_log_root=str(expected_root),
        kit_shutdown_log_verified=False, normal_close_return_receipt_present='native_close_returned' in phases,
        recorded_updates=8, optimizer_proposals=32, optimizer_commits=32,
        sampled_transitions=3072, finite_baseline_checks=32, candidate_checks=ledger['candidate_checks'],
        all_accepted_scales=[r['candidates'][-1]['scale'] for r in ledger['commits']],
        maximum_recorded_teacher_rmse=max(r['teacher_action_rmse'] for r in cloth['optimizer_steps']),
        maximum_recorded_teacher_maxabs=max(r['teacher_action_max_abs'] for r in cloth['optimizer_steps']),
        owner_sha256=head['owner_sha256'], checkpoints=saved,
        derived_outer_actor_decoder_calls=counts,
        call_count_basis='PINNED_SOURCE_PATHS_AND_EXISTING_JSON_NOT_KERNEL_INSTRUMENTATION',
        final_tensor_verification_runs=0, post_stop_tensor_deserializations=0,
        effect_rollouts=0, policy_improvement_proven=False, machine_fault_evidenced=False,
        new_health_samples=0, existing_health_audit_sha256=sha256(health_path),
        plan_sha256=sha256(trial.PLAN), result_sha256=sha256(trial.RESULT),
        auditor_sha256=sha256(trial.ROOT / 'script/project_tools/humanplus_head_terminal_audit.py'),
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(target, proof)
    print(json.dumps({k: v for k, v in proof.items() if k not in ('checkpoints', 'all_accepted_scales')}, indent=2))


if __name__ == '__main__':
    main()
