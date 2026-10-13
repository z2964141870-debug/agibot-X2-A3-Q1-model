"""E37 archives E36's pre-update std boundary and uses the official runtime clamp."""

import argparse
import time

from script.a3 import cloth_constrained_trial as shared
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.smpl_frozen_followup import health_summary

PRIOR = shared.OUT
PRIOR_LOG = shared.LOG
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_canonical_20261013_E37'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_canonical_20261013_E37'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_canonical_20261013_E37/R14_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_canonical_20261013_E37.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_CANONICAL_20261013_E37'
shared.TRIAL = 'R14'
shared.MODULE = 'script.a3.cloth_canonical_trial'
shared.CANONICAL_STD = True
shared.CODE += ('script/a3/cloth_canonical_trial.py', 'script/a3/cloth_std_contract.py',
                'script/a3/test_cloth_std_contract.py')


def register():
    before = shared.read(PRIOR / 'state.json')
    console = PRIOR_LOG / 'prepare/console.log'
    if before['status'] != 'STOPPED' or 'Official std needs clamping' not in console.read_text():
        raise ValueError('Unexpected E36 state; preserve and inspect before a new branch')
    if list((PRIOR / 'motionlib').glob('*.pkl')) or shared.read(
            shared.ROOT / 'data/manifests/a3_zhy_cloth_constrained_20261013_E36.json').get('training_verification'):
        raise ValueError('E36 executed beyond its expected pre-update boundary')
    health = health_summary(PRIOR_LOG)
    if health['critical_kernel_errors'] or health['telemetry_errors'] or health['oom_counts'] != [0]:
        raise ValueError('E36 machine fault; no new branch')
    shared.register()
    job = shared.read(shared.OUT / 'registration.json')
    if health['boot_ids'] != [job['boot_id']]:
        raise ValueError('Boot changed after E36; stop')
    job.update(finalized_at=time.time(), parent_experiment='A3_ZHY_CLOTH_CONSTRAINED_20261013_E36',
        main_variable='EXPLICIT_OFFICIAL_RUNTIME_STD_CANONICALIZATION_SAME_CONSTRAINED_RECIPE',
        official_deterministic_baseline_reused=True, original_serialized_std_preserved=True)
    for path in (PRIOR / 'state.json', console,
            shared.ROOT / 'data/manifests/a3_zhy_cloth_constrained_20261013_E36.json'):
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)
    print('E37/R14 registration finalized; E36 remains STOPPED with zero updates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='prepare')
    args = parser.parse_args()
    value = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else shared.run(args.stage)
    raise SystemExit(value if isinstance(value, int) else 0)
