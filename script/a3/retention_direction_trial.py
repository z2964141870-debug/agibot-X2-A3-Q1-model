"""E46: bounded saved-state gradient geometry, without simulation or optimization."""

import argparse
import json
import os
from pathlib import Path
import time

from script.a3 import cloth_resource_trial as runtime
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record, command
from script.a3.retention_direction_protocol import Budget

shared = runtime.shared
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_retention_direction_20261013_E46'
shared.LOG = shared.ROOT / 'logs/a3_zhy_retention_direction_20261013_E46'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_retention_direction_20261013_E46.json'
shared.EXPERIMENT = 'A3_ZHY_RETENTION_DIRECTION_20261013_E46'
shared.MODULE = 'script.a3.retention_direction_trial'
shared.STAGES = {'probe': ('REGISTERED', 'COMPLETE')}
PLAN = shared.ROOT / 'data/manifests/a3_retention_conflict_plan_20261013.json'
META = shared.ROOT / 'logs/a3_zhy_cloth_capture_20261013_E39/training/runtime_metadata.json'
PARENT = shared.ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45'
STOP = shared.ROOT / 'data/manifests/a3_zhy_cloth_failure_stop_20261013_E45.json'
shared.CODE += ('script/a3/retention_direction_trial.py', 'script/a3/retention_direction_probe.py',
    'script/a3/retention_direction_protocol.py', 'script/a3/test_retention_direction_protocol.py',
    'script/a3/cloth_cached_tokens.py', 'script/a3/cloth_std_contract.py')


def preflight(baseline):
    rows = []
    for index in range(3):
        value = shared.guard.preflight()
        if (value['boot_id'] != baseline['boot_id'] or
                value['container_memory']['oom_kill'] != baseline['baseline_oom_kill']):
            raise RuntimeError('New boot or OOM; do not start')
        if value['container_memory']['available_estimate_bytes'] < 8 * 1024**3:
            raise RuntimeError('Original 8 GiB entry headroom unavailable')
        if any(r['rss_bytes'] > 1024**3 for r in value['process_memory']['rows']):
            raise RuntimeError('Other large process; no concurrent computation')
        # Resolve missing smaps coverage through status, without collecting argv/env.
        checks = []
        for error in value['process_memory']['errors']:
            path = Path('/proc') / str(error['pid']) / 'status'
            if not path.exists():
                checks.append(dict(pid=error['pid'], status='EXITED'))
                continue
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)
            state = fields['State'].strip()
            if state.startswith('Z'):
                checks.append(dict(pid=error['pid'], status='ZOMBIE'))
                continue
            rss = int(fields['VmRSS'].split()[0]) * 1024
            name = fields['Name'].strip()
            if rss > 1024**3 or name not in ('pod', 'ssh-agent', 'gpg-agent'):
                raise RuntimeError('Unresolved process telemetry; do not compute')
            checks.append(dict(pid=error['pid'], status=state, comm=name, rss_bytes=rss))
        value['missing_process_status_checks'] = checks
        rows.append(value)
        if index < 2:
            time.sleep(5)
    return rows


def register():
    runtime.lightweight_controller()
    if shared.OUT.exists() or shared.LOG.exists():
        raise FileExistsError('Never reset or repeat E46')
    stopped = shared.read(STOP)
    if (shared.read(PARENT / 'state.json')['status'] != 'STOPPED' or
            stopped['status'] != 'STOPPED_ORIGINAL_TEACHER_DRIFT_PROTECTION'):
        raise ValueError('Requires preserved E45 teacher stop')
    parent = shared.read(PARENT / 'registration.json')
    rows = preflight(parent)
    plan = shared.read(PLAN)
    inputs = {r['path']: r['sha256'] for r in plan['proposed_input_files'] + plan['sources']}
    for name, expected in inputs.items():
        if sha256(shared.ROOT / name) != expected:
            raise ValueError('Fixed source or input changed: ' + name)
    vendor = shared.ROOT / 'script/vendor/sonic_for_a3'
    for p in (PLAN, META, PARENT / 'state.json', PARENT / 'registration.json',
              vendor / 'gear_sonic/trl/modules/actor_critic_modules.py'):
        inputs[str(p.relative_to(shared.ROOT))] = sha256(p)
    kernel = command(['dmesg', '--color=never'])
    if kernel['returncode']:
        raise RuntimeError('Kernel telemetry unavailable')
    job = dict(experiment=shared.EXPERIMENT, registered_at=time.time(),
        project_head=command(['git','rev-parse','HEAD'])['stdout'],
        boot_id=parent['boot_id'], baseline_oom_kill=parent['baseline_oom_kill'],
        kernel_baseline=kernel['stdout'].splitlines(),
        code_sha256={n:sha256(shared.ROOT/n) for n in shared.CODE}, input_sha256=inputs,
        main_variable='OBJECTIVE_TERM_ON_FIXED_SAVED_INPUT_AND_POLICY_STATE',
        policy_steps=[0,4], budget=Budget.LIMITS, optimizer_steps=0, new_simulation_rollouts=0,
        clipped_or_adam_update_computed=False, checkpoint_write_allowed=False,
        initial_action_rmse_max=1e-5, initial_ratio_deviation_max=.001,
        gradient_sum_atol=1e-7, gradient_sum_rtol=1e-5,
        saved_teacher_rmse_atol=1e-5,
        value_term='SAVED_VALUE_CONSTANT_FOR_ACTOR_PARTIAL_GRADIENT_NOT_CRITIC_REPLAY',
        fixed_input_counterfactual_not_historical_failure=True,
        max_cpu_c=90,max_gpu_c=85,consecutive_hot_samples=2,sample_seconds=5,
        memory_reserve_gib=4,runtime_limit_seconds=900,auto_resume=False,repeat_allowed=False,
        backup_status='LOCAL_ONLY')
    shared.OUT.mkdir(parents=True)
    shared.LOG.mkdir(parents=True)
    atomic_json(shared.OUT/'registration.json',job)
    atomic_json(shared.OUT/'state.json',dict(status='REGISTERED'))
    for row in rows:
        append_record(shared.LOG/'entry_preflight.jsonl',row)
    append_record(shared.LOG/'health.jsonl',rows[-1])
    print(json.dumps(dict(status='REGISTERED',experiment=job['experiment'],budget=job['budget'])))


def worker():
    state=shared.read(shared.OUT/'state.json')
    if (state['status']!='PROBE_RUNNING' or state['monitor_pid']!=os.getppid() or
            os.environ.get('YUANQI_E36_MONITOR_PID')!=str(os.getppid())):
        raise RuntimeError('Requires registered guarded monitor')
    job=shared.read(shared.OUT/'registration.json')
    shared.unchanged(job)
    from script.a3.retention_direction_probe import probe
    probe(job,shared.ROOT,shared.OUT,META)
    shared.unchanged(job)


def run():
    runtime.lightweight_controller()
    if shared.read(shared.OUT/'state.json')['status']!='REGISTERED':
        raise RuntimeError('No replay or recovery')
    job=shared.read(shared.OUT/'registration.json')
    for row in preflight(job):
        append_record(shared.LOG/'probe_entry_preflight.jsonl',row)
    result=runtime.run('probe')
    receipt=shared.read(shared.MANIFEST)
    if (shared.OUT/'result.json').exists():
        receipt['result']=shared.read(shared.OUT/'result.json')
    atomic_json(shared.MANIFEST,receipt)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','register','run','worker'))
    parser.add_argument('--stage',choices=('probe',),default='probe')
    args=parser.parse_args()
    if args.action=='plan':
        runtime.lightweight_controller()
        print(json.dumps(dict(status='PLAN_ONLY_NOT_RUNTIME_STATE',budget=Budget.LIMITS,
                             optimizer_steps=0,new_rollouts=0)))
    else:
        result=register() if args.action=='register' else worker() if args.action=='worker' else run()
        raise SystemExit(result if isinstance(result,int) else 0)
