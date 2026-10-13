"""E42/R17 tests sampled-token consistency before constrained clothing updates."""

import argparse
import io
import os
import time
import unittest

from script.a3 import cloth_resource_trial as runtime
from script.a3.checkpoint_store import atomic_json, sha256

shared = runtime.shared
PRIOR = shared.ROOT / 'data/experiments/a3_zhy_cloth_resource_20261013_E38'
PROOF = shared.ROOT / 'data/experiments/a3_zhy_cloth_intervention_20261013_E41/training_verification.json'
VERIFY = shared.verify_training
shared.OUT = shared.ROOT / 'data/experiments/a3_zhy_cloth_cached_20261013_E42'
shared.LOG = shared.ROOT / 'logs/a3_zhy_cloth_cached_20261013_E42'
shared.RUN = shared.ROOT / 'data/training/a3_zhy_cloth_cached_20261013_E42/R17_001_s0'
shared.MANIFEST = shared.ROOT / 'data/manifests/a3_zhy_cloth_cached_20261013_E42.json'
shared.EXPERIMENT = 'A3_ZHY_CLOTH_CACHED_20261013_E42'
shared.TRIAL = 'R17'
shared.MODULE = 'script.a3.cloth_cached_trial'
shared.TRAINER = 'script.a3.cloth_cached_tokens.ClothCachedTokenTrainer'
shared.CODE += ('script/a3/cloth_cached_trial.py', 'script/a3/cloth_cached_tokens.py',
                'script/a3/test_cloth_cached_tokens.py')


def register():
    proof = shared.read(PROOF)
    if proof['status'] != 'VERIFIED_TOKEN_INTERVENTION' or not proof['intervention_passed']:
        raise ValueError('Requires verified E41 token intervention')
    runtime.register()
    job = shared.read(shared.OUT / 'registration.json')
    job.update(parent_experiment=proof.get('experiment','A3_ZHY_CLOTH_INTERVENTION_20261013_E41'),
        finalized_at=time.time(), main_variable='SAMPLED_ASSEMBLED_TOKEN_REUSE_FROZEN_CODEC_SAME_CLOTHING_RECIPE',
        cached_token_controls=3, prepared_motions_reused=True, decoder_gradient_required=True)
    prepared = shared.read(PRIOR / 'prepare.json')
    for path in [PROOF, PRIOR / 'prepare.json'] + [shared.ROOT / row['path'] for row in prepared['motions']]:
        job['input_sha256'][str(path.relative_to(shared.ROOT))] = sha256(path)
    atomic_json(shared.OUT / 'registration.json', job)


def prepare(job):
    from script.a3 import test_cloth_cached_tokens as tests
    stream = io.StringIO()
    checked = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    if not checked.wasSuccessful() or checked.testsRun != 3:
        raise ValueError('Cached-token controls failed: '+stream.getvalue())
    prepared = shared.read(PRIOR / 'prepare.json')
    for row in prepared['motions']:
        path = shared.ROOT / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('Previously accepted motion changed')
        (shared.OUT / 'motionlib' / path.name).symlink_to(os.path.relpath(path, shared.OUT / 'motionlib'))
    atomic_json(shared.OUT / 'prepare.json', dict(prepared, cached_token_tests=3,
        cached_token_test_output=stream.getvalue(), original_motion_abi_reused=True))


def verify(job):
    VERIFY(job)
    import torch
    from script.a3.cloth_constrained_trainer import initial_probability
    directory = shared.LOG / 'training'
    contract = shared.read(directory / 'sampled_token_contract.json')
    if len(contract['batches']) != 8 or any(not row['decoder_gradient_enabled'] for row in contract['batches']):
        raise ValueError('Cached-token actual gradient/batch coverage failed')
    repair = torch.load(directory / 'initial_token_repair.pt', map_location='cpu', weights_only=False)
    rollout = torch.load(directory / 'initial_sampled_tokens.pt', map_location='cpu', weights_only=False)
    batch = repair['batch']
    if not torch.equal(rollout['tokens'][batch['micro_batch_inds']], batch['mb_sampled_tokens']):
        raise ValueError('First training tokens do not match actual sampling identity')
    record = initial_probability(repair['corrected'], batch)
    output = shared.read(shared.OUT / 'training_verification.json')
    output.update(sampled_token_contract=contract, initial_cached_probability=record,
        first_sampled_token_identity_exact=True, raw_first_action_max_abs=float(
            (repair['raw_reencoded_mean']-batch['mb_old_mu']).abs().max()))
    atomic_json(shared.OUT / 'training_verification.json', output)


shared.prepare = prepare
shared.verify_training = verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run', 'worker'))
    parser.add_argument('--stage', choices=tuple(shared.STAGES), default='prepare')
    args = parser.parse_args()
    result = register() if args.action == 'register' else shared.worker(args.stage) if args.action == 'worker' else runtime.run(args.stage)
    raise SystemExit(result if isinstance(result, int) else 0)
