"""CUDA objective gradients on fixed training-only saved states, with no Adam."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'q1_five_stage'), str(HERE.parent/'primebot_q1')]
from common import identity, require
from delivery import read
from paused_runtime import assert_owner


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.expected_gpu, 'Use the admitted GPU only')
    auth, original, plan = read(args.output/'authorization.json'), read(args.plan), read(args.audit_plan)
    require(time.monotonic_ns() < args.deadline_ns-30_000_000_000
        and all(identity(r['path']) == r for r in auth['protected_inputs'])
        and original['source'] == str(args.morph_source), 'Protected sources or reserved deadline changed')
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Admitted CUDA unavailable')
    sys.path[:0] = [str(args.morph_source/'src'), str(HERE.parent/'morphmimic_q1')]
    from bootstrap import install
    from inference import load_policy
    from student_fit import module_sha, save_json
    from student_objective_audit import audit, load_train
    from upper_mps_forecast import audit_cpu_tensors
    base = read(plan['base_plan']['path'])
    native, metadata = load_policy(args.checkpoint, install('q1')['robot'], original['code_commit'],
        'cuda:0', expected_plan=original)
    require(metadata.get('iteration') == 32000, 'Keep the frozen actual32k teacher')
    norms = {k:module_sha(v) for k,v in native.norms.items()}
    payloads = {}
    for name, record in [('initial', base['initial_checkpoint']), *plan['last_checkpoints'].items()]:
        require(identity(record['path']) == record, 'Registered actual checkpoint changed')
        value = torch.load(record['path'], map_location='cpu', weights_only=True)
        audit_cpu_tensors(torch, value)
        require(value['arm'] == 'causal_history' and value['human_real_future_frames'] == 0
            and value['actual_updates'] == 3000 and value['actual_adam_steps'] == [3000]*4,
            'Use only the actual initial and completed last checkpoints')
        if name == 'initial':
            require(value['schema'] == 'q1_f0_proprio_residual_v1' and value['selected_update'] == 3000,
                'The initial is the previously trained selected3000 policy')
        else:
            require(value['schema'] == 'q1_f0_retained_sequence_v1' and value['checkpoint_update'] == 3000
                and value['variant'] == name.removesuffix('_last'), 'Do not use selected0 as last3000')
            require(value['current_statistics'] == payloads['initial']['current_statistics']
                and all(torch.equal(v, payloads['initial']['base_encoder'][k]) for k,v in value['base_encoder'].items()),
                'Common human encoder and normalization must stay exactly frozen')
        require(value['frozen_teacher_policy_sha256'] == module_sha(native.policy)
            and value['decoder_sha256'] == module_sha(native.policy.decoder), 'Teacher or decoder provenance mismatch')
        payloads[name] = value
    origins = [load_train(Path(o['root'])/'train_data.npz') for o in base['origins']]
    pure = load_train(base['pure_transitions']['train']['path'])
    result = audit(native, payloads, origins, pure, read(base['candidate_plan']['path']), base['jobs'],
        read(plan['sequence_summary']['path']), plan['probe'], args.deadline_ns)
    require(result['training_records'] == dict(old=6725, pure=406)
        and result['training_windows'] == dict(old=6581, pure=387)
        and norms == {k:module_sha(v) for k,v in native.norms.items()}
        and all(not v.training for v in native.norms.values())
        and all(identity(r['path']) == r for r in auth['protected_inputs']),
        'Training coverage, frozen normalizers or protected input terminal gate changed')
    torch.cuda.synchronize()
    save_json(args.output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'checkpoint', 'morph-source', 'output', 'audit-plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-gpu', 'gpu-owner-ticks'):
        parser.add_argument('--'+name, required=True)
    for name in ('gpu-owner-pid', 'deadline-ns'):
        parser.add_argument('--'+name, type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps(dict(status=result['status'], counts=result['counts'], optimizer_updates=0)))
