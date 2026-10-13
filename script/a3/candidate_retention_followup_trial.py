"""E50 guarded saved-batch displacement projection, no optimization or simulation."""

import argparse
import json
import os
import time

from script.a3 import fresh_batch_trial as previous
from script.a3 import retention_direction_trial as prior
from script.a3.candidate_retention_protocol import Budget, TOLERANCES
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.diagnostics.host_loadcheck import FATAL

runtime, shared = prior.runtime, prior.shared
PARENT = shared.OUT
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_candidate_retention_followup_20261014_E50'
shared.LOG = shared.ROOT / 'logs/a3_zhy_candidate_retention_followup_20261014_E50'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_candidate_retention_followup_20261014_E50.json'
shared.EXPERIMENT = 'A3_ZHY_CANDIDATE_RETENTION_FOLLOWUP_20261014_E50'
shared.MODULE = 'script.a3.candidate_retention_followup_trial'
PLAN = shared.ROOT / 'data/manifests/a3_candidate_retention_followup_plan_20261014.json'
shared.CODE += ('script/a3/candidate_retention_trial.py', 'script/a3/candidate_retention_probe.py',
                'script/a3/candidate_retention_protocol.py', 'script/a3/test_candidate_retention_protocol.py',
    'script/a3/candidate_retention_followup_trial.py', 'script/a3/candidate_retention_followup_probe.py',
    'script/a3/optimizer_metadata_contract.py', 'script/a3/test_optimizer_metadata_contract.py')


def register():
    runtime.lightweight_controller()
    if shared.OUT.exists() or shared.LOG.exists():
        raise FileExistsError('Never reset or repeat E50')
    if shared.read(PARENT / 'state.json')['status'] != 'COMPLETE':
        raise ValueError('Requires completed E48')
    parent = shared.read(PARENT / 'registration.json')
    if shared.read(prior.PARENT / 'state.json')['status'] != 'STOPPED':
        raise ValueError('E45 must remain stopped')
    previous_stop = shared.read(shared.ROOT / 'data/manifests/a3_candidate_retention_stop_20261014.json')
    frozen_state = shared.read(shared.ROOT / 'data/experiments/a3_zhy_candidate_retention_20261014_E49/state.json')
    if previous_stop['status'] != 'STOPPED_IMPLEMENTATION_JSON_COMPARISON' or frozen_state['status'] != 'STOPPED':
        raise ValueError('E49 stop must remain preserved')
    rows = prior.preflight(parent)
    plan = shared.read(PLAN)
    inputs = dict(parent['input_sha256'])
    inputs.update({r['path']: r['sha256'] for r in
                   plan['inputs']})
    inputs.update(plan['source_sha256'])
    if plan['status'] != 'IMPLEMENTED_UNREGISTERED' or not plan.get('explicit_retry_authorization') or plan['budget'] != Budget.LIMITS or plan['tolerances'] != TOLERANCES:
        raise ValueError('Implementation and fixed budget not published')
    for name, expected in inputs.items():
        if sha256(shared.ROOT / name) != expected:
            raise ValueError('Fixed source/input changed: ' + name)
    extra = (shared.ROOT / 'data/manifests/a3_candidate_retention_stop_20261014.json',
        shared.ROOT / 'data/experiments/a3_zhy_candidate_retention_20261014_E49/state.json',
        shared.ROOT / 'data/experiments/a3_zhy_candidate_retention_20261014_E49/registration.json',
        PLAN, PARENT / 'registration.json', PARENT / 'state.json', PARENT / 'result.json',
        PARENT / 'postexit_verification.json',
        shared.ROOT / 'script/vendor/sonic_for_a3/gear_sonic/utils/running_mean_std.py',
        shared.ROOT / 'script/vendor/sonic_for_a3/gear_sonic/trl/modules/base_module.py',
        shared.ROOT / 'data/env/lib/python3.11/site-packages/torch/nn/utils/clip_grad.py')
    inputs.update({str(p.relative_to(shared.ROOT)): sha256(p) for p in extra})
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    fresh = [line for line in kernel['stdout'].splitlines()
             if line not in parent['kernel_baseline'] and FATAL.search(line)]
    if fresh:
        raise RuntimeError('New critical kernel evidence; no computation')
    job = dict(experiment=shared.EXPERIMENT, registered_at=time.time(),
        project_head=command(['git', 'rev-parse', 'HEAD'])['stdout'],
        boot_id=parent['boot_id'], baseline_oom_kill=parent['baseline_oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={n: sha256(shared.ROOT / n) for n in shared.CODE}, input_sha256=inputs,
        main_variable=plan['main_variable'],
        policy_step=4, saved_optimizer_step=16, budget=Budget.LIMITS,
        optimizer_steps=0, new_rollouts=0, transitions=0, candidate_count=2, new_deployable_checkpoints=0,
        tolerances=TOLERANCES, original_candidates=1, projected_candidates=1,
        critic_scope='SAVED_BUFFERS_THEN_ONE_TEMPORARY_TRAIN_FORWARD',
        seed=0, environments=16, steps=24, first_randperm_environments=4,
        initial_action_rmse_max=1e-5, initial_ratio_deviation_max=.001,
        gradient_comparison_atol=1e-7, gradient_comparison_rtol=1e-5,
        scalar_match_atol=1e-7, scalar_match_rtol=1e-5,
        teacher_rmse_max=.05, teacher_max_abs=.5, max_grad_norm=.1,
        finite_candidate_functional_parameters_only=True,
        fused_bitwise_or_historical_step_reconstruction=False,
        max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, sample_seconds=5,
        memory_reserve_gib=4, runtime_limit_seconds=900, auto_resume=False,
        repeat_allowed=False, backup_status='LOCAL_ONLY')
    shared.OUT.mkdir(parents=True)
    shared.LOG.mkdir(parents=True)
    atomic_json(shared.OUT / 'registration.json', job)
    atomic_json(shared.OUT / 'state.json', dict(status='REGISTERED'))
    for row in rows:
        append_record(shared.LOG / 'entry_preflight.jsonl', row)
    append_record(shared.LOG / 'health.jsonl', rows[-1])
    print(json.dumps(dict(status='REGISTERED', experiment=job['experiment'], budget=job['budget'])))


def worker():
    state = shared.read(shared.OUT / 'state.json')
    if (state['status'] != 'PROBE_RUNNING' or state['monitor_pid'] != os.getppid() or
            os.environ.get('YUANQI_E36_MONITOR_PID') != str(os.getppid())):
        raise RuntimeError('Requires guarded parent monitor')
    job = shared.read(shared.OUT / 'registration.json')
    shared.unchanged(job)
    os.environ.update(PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WANDB_MODE='disabled', ACCELERATE_MIXED_PRECISION='no')
    from script.a3.candidate_retention_followup_probe import probe
    probe(job, shared.ROOT, shared.OUT, prior.META)
    shared.unchanged(job)
    atomic_json(shared.OUT / 'post_input_verification.json', dict(at=time.time(), unchanged=True,
        input_file_count=len(job['input_sha256']), source_file_count=len(job['code_sha256']),
        progress_sha256=sha256(shared.OUT / 'progress.json'), result_sha256=sha256(shared.OUT / 'result.json')))


def run():
    runtime.lightweight_controller()
    if shared.read(shared.OUT / 'state.json')['status'] != 'REGISTERED':
        raise RuntimeError('No repeat or recovery')
    for row in prior.preflight(shared.read(shared.OUT / 'registration.json')):
        append_record(shared.LOG / 'probe_entry_preflight.jsonl', row)
    code = runtime.run('probe')
    receipt = shared.read(shared.MANIFEST)
    if (shared.OUT / 'result.json').exists():
        receipt['result'] = shared.read(shared.OUT / 'result.json')
    atomic_json(shared.MANIFEST, receipt)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'register', 'run', 'worker'))
    parser.add_argument('--stage', choices=('probe',), default='probe')
    args = parser.parse_args()
    if args.action == 'plan':
        runtime.lightweight_controller()
        print(json.dumps(dict(status='PLAN_ONLY', budget=Budget.LIMITS, optimizer_steps=0)))
    else:
        code = register() if args.action == 'register' else worker() if args.action == 'worker' else run()
        raise SystemExit(code if isinstance(code, int) else 0)
