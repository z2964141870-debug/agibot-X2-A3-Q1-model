"""Freeze only successful HP21 training inputs for independent HP22 completion."""

import json
from pathlib import Path
import subprocess

from script.a3 import humanplus_specialist_completion as trial
from script.a3.checkpoint_store import atomic_json, sha256


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=sha256(path))


def main():
    trial.runtime.require_root()
    if trial.DESIGN.exists() or any(trial.ROOT.glob('data/manifests/*HP22*')):
        raise FileExistsError('Independent D06/HP22 only')
    trial.require_source()
    source = trial.read(trial.SOURCE_PLAN)
    # Retain only the actual source inputs; no stopped artifact recursion.
    inputs = dict(source['input_identity'])
    paths = [trial.SOURCE_PLAN, trial.SOURCE_RESULT,
        trial.SOURCE_LOG / 'verify-training/console.log',
        trial.SOURCE_LOG / 'train/specialist_contract.json',
        trial.SOURCE_LOG / 'train/actual_optimizer_steps.jsonl',
        trial.SOURCE_LOG / 'train/initial_sampled_tokens.pt',
        trial.SOURCE_LOG / 'train/initial_probability.pt', trial.RUN / 'config.yaml',
        trial.SOURCE_LOG / 'isaac-baseline/isaac_metrics.json',
        trial.SOURCE_LOG / 'isaac-baseline/full_isaac_trace.json',
        trial.SOURCE_LOG / 'isaac-baseline/bridge_metadata.json',
        trial.SOURCE_LOG / 'isaac-baseline/replay_progress.json']
    for stage in ('isaac-baseline', 'train', 'verify-training'):
        paths.append(trial.SOURCE_PLAN.with_name(trial.SOURCE_NAME + '_' + stage + '_audit.json'))
    for step in (0, 256):
        p = trial.checkpoint(step)
        sidecar = trial.read(p.with_suffix('.json'))
        if identity(p) != {k: sidecar[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Completed training checkpoint identity changed')
        paths.extend([p, p.with_suffix('.json'), p.parent / 'specialist_owners.json'])
    for p in paths:
        inputs[str(p.relative_to(trial.ROOT))] = identity(p)
    for name, expected in inputs.items():
        if identity(trial.ROOT / name) != expected:
            raise ValueError('Source input changed: ' + name)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
        'script/a3', 'script/project_tools'], text=True).splitlines()
    code = {}
    for name in sorted(set(names)):
        if name.endswith('.py'):
            p = trial.ROOT / name
            compile(p.read_text(), name, 'exec')
            code[name] = identity(p)
    record = dict(source, code_identity=code, input_identity=inputs,
        status='IMPLEMENTED_CONTROLLED_READY_FOR_REGISTRATION',
        source_training='HP21_COMPLETE_256_UPDATES_NO_RETRAINING',
        source_trial_remains='STOPPED_PRELOAD_IMPORT_ERROR',
        completion_stages=list(trial.STAGES), new_training_updates=0, new_interactions=0,
        repaired_verification='PURE_SAVED_PROBABILITY_NO_TRAINER_IMPORT',
        controls=dict(mac_passed=3, server_passed=3, no_model_or_tensor_controls=True))
    atomic_json(trial.DESIGN, record)
    print(json.dumps(dict(code_files=len(code), input_files=len(inputs),
        environment_files=len(record['environment_identity']), new_training_updates=0)))


if __name__ == '__main__':
    main()
