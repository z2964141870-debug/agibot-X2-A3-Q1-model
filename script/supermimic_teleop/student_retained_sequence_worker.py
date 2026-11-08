"""Only owned saved train/validation features reach the new sequence fitter."""

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
    auth, original, plan = read(args.output/'authorization.json'), read(args.plan), read(args.sequence_plan)
    require(time.monotonic_ns() < args.deadline_ns-60_000_000_000
        and all(identity(r['path']) == r for r in auth['protected_inputs'])
        and original['source'] == str(args.morph_source), 'Published sources or reserved deadline changed')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Admitted CUDA unavailable')
    sys.path[:0] = [str(args.morph_source/'src'), str(HERE.parent/'morphmimic_q1')]
    from bootstrap import install
    from inference import load_policy
    from student_fit import module_sha
    from student_retained_sequence_fit import fit_pair
    from student_proprio_models import zero_adapter
    from upper_mps_forecast import audit_cpu_tensors
    base = read(plan['base_plan']['path'])
    native, metadata = load_policy(args.checkpoint, install('q1')['robot'], original['code_commit'],
        'cuda:0', expected_plan=original)
    initial = base['initial_checkpoint']
    payload = torch.load(initial['path'], map_location='cpu', weights_only=True)
    audit_cpu_tensors(torch, payload)
    require(metadata.get('iteration') == 32000 and payload.get('schema') == 'q1_f0_proprio_residual_v1'
        and payload.get('arm') == 'causal_history' and payload.get('selected_update') == 3000
        and payload.get('actual_updates') == 3000 and payload.get('actual_adam_steps') == [3000]*4
        and payload['provenance']['teacher'] == identity(args.checkpoint)
        and payload.get('human_real_future_frames') == 0 and not zero_adapter(payload['adapter']),
        'Require the exact actual nonzero selected3000 student')
    norms = {k:module_sha(v) for k,v in native.norms.items()}
    require(all(not v.training for v in native.norms.values()), 'Frozen normalizers must stay in eval mode')

    def inputs(path):
        with np.load(path, allow_pickle=False) as data:
            names = ('f0', 'history', 'teacher_action', 'clip_id', 'frame')
            values = {k:data[k] for k in names}
            if 'terminal_after_action' in data.files:
                values['terminal_after_action'] = data['terminal_after_action']
            return values

    origins = [{s:inputs(Path(o['root'])/(s+'_data.npz')) for s in ('train', 'validation')}
        for o in base['origins']]
    pure = {s:inputs(base['pure_transitions'][s]['path']) for s in ('train', 'validation')}
    result = fit_pair(native, origins, pure, read(base['candidate_plan']['path']), base['jobs'],
        read(plan['sequence_summary']['path']), payload, plan['fit'], args.output, args.deadline_ns,
        dict(teacher=identity(args.checkpoint), initial_student=initial, sequence_plan=identity(args.sequence_plan),
            sequence_summary=plan['sequence_summary'], original_normalizers=norms))
    require(result['old_records'] == dict(train=6725, validation=2358)
        and result['pure_records'] == dict(train=406, validation=219)
        and result['sequence_windows'] == dict(old=dict(train=6581, validation=2304), pure=dict(train=387, validation=209))
        and norms == {k:module_sha(v) for k,v in native.norms.items()}
        and all(identity(r['path']) == r for r in auth['protected_inputs']),
        'Actual sequence coverage, normalizers or protected inputs changed')
    torch.cuda.synchronize()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('plan', 'checkpoint', 'morph-source', 'output', 'sequence-plan'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('expected-gpu', 'gpu-owner-ticks'):
        parser.add_argument('--'+name, required=True)
    for name in ('gpu-owner-pid', 'deadline-ns'):
        parser.add_argument('--'+name, type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps(dict(status=result['status'], selected=[dict(arm=r['variant'], update=r['selected_update'],
        unchanged=r['selected_adapter_unchanged']) for r in result['arms']], actual_optimizer_updates=result['actual_optimizer_updates'])))
