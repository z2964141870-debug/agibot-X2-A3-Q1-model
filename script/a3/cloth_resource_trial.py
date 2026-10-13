"""E38 keeps the clothing recipe and adds lightweight, measured supervision."""

import argparse
import os
import sys
import time

from script.a3 import cloth_constrained_trial as shared
from script.a3.stage_resources import with_processes, summarize
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import append_record

BASE_HEALTH = shared.guard.health
shared.guard.health = lambda: with_processes(BASE_HEALTH)
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_canonical_20261013_E37'
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_resource_20261013_E38'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_resource_20261013_E38'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_resource_20261013_E38/R15_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_resource_20261013_E38.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_RESOURCE_20261013_E38'
shared.TRIAL = 'R15'
shared.MODULE = 'script.a3.cloth_resource_trial'
shared.CANONICAL_STD = True
shared.CODE += ('script/a3/cloth_resource_trial.py', 'script/a3/stage_resources.py',
                'script/a3/cloth_std_contract.py', 'script/a3/test_cloth_std_contract.py',
                'script/a3/test_checkpoint_store.py', 'script/diagnostics/host_loadcheck.py',
                'script/a3/idle_reboot_watch.py')


def lightweight_controller():
    if 'torch' in sys.modules or 'gear_sonic.trl.trainer.ppo_trainer' in sys.modules:
        raise RuntimeError('Controller imported tensor/training runtime')


def register():
    lightweight_controller()
    previous = shared.read(PRIOR / 'state.json')
    if previous['status'] != 'STOPPED' or previous['reason'] != 'MEMORY_RESERVE_PROTECTION':
        raise ValueError('Expected archived E37 resource stop')
    old = shared.read(PRIOR / 'registration.json')
    baselines = []
    for i in range(3):
        value = shared.guard.preflight()
        if value['boot_id'] != old['boot_id'] or value['container_memory']['oom_kill'] != old['baseline_oom_kill']:
            raise RuntimeError('New machine fault; no computation')
        if value['container_memory']['available_estimate_bytes'] < 8 * 1024**3:
            raise RuntimeError('Insufficient extra start headroom for E38')
        if any(row['rss_bytes'] > 1024**3 for row in value['process_memory']['rows']):
            raise RuntimeError('Existing large process requires inspection before E38')
        baselines.append(value)
        if i < 2:
            time.sleep(5)
    shared.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment=old['experiment'], finalized_at=time.time(),
        main_variable='RESOURCE_AWARE_LIGHTWEIGHT_ENTRY_SAME_CONSTRAINED_RECIPE',
        resource_evidence_required=True, prepare_process_budget_bytes=3 * 1024**3,
        preflight_extra_available_bytes=8 * 1024**3, preflight_samples=3,
        controller_tensor_import_allowed=False, baseline_process_peak_e37_known=False)
    for path in (PRIOR / 'registration.json', PRIOR / 'state.json',
            shared.ROOT / 'data/manifests/a3_zhy_cloth_canonical_20261013_E37.json'):
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)
    for value in baselines:
        append_record(shared.LOG / 'resource_preflight.jsonl', value)
    print('E38/R15 registered; E37 remains stopped; no optimizer has run', flush=True)


def run(stage):
    lightweight_controller()
    result = shared.run(stage)
    lightweight_controller()
    receipt = shared.read(shared.MANIFEST)
    receipt['resources'] = summarize(shared.LOG / 'health.jsonl')
    receipt['controller_pid'] = os.getpid()
    receipt['controller_tensor_runtime_imported'] = False
    if (shared.LOG / 'stage_resources.jsonl').exists():
        receipt['stage_resources_sha256'] = sha256(shared.LOG / 'stage_resources.jsonl')
    atomic_json(shared.MANIFEST, receipt)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='prepare')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
