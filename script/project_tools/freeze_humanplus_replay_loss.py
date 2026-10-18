"""Freeze independent D02 execution identities after standard-library controls."""

import json
from pathlib import Path
import subprocess

from script.a3 import humanplus_replay_loss as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_replay_protocol import CALL_BUDGET, validate_selection


def main():
    trial.runtime.require_root()
    target = trial.ROOT / 'data/manifests/a3_humanplus_replay_loss_readiness_v2_20261018.json'
    if target.exists():
        raise FileExistsError('Implementation freeze is single-use')
    source_path = trial.ROOT / 'data/manifests/a3_humanplus_head_logged_20261018_HP14_plan.json'
    original = trial.read(source_path)
    design = trial.read(trial.DESIGN)
    validate_selection(design)
    inputs = dict(original['input_sha256'])
    for name, identity in design['read_only_sources'].items():
        path = trial.ROOT / name
        if path.stat().st_size != identity['bytes'] or sha256(path) != identity['sha256']:
            raise ValueError('Reviewed D02 source changed: ' + name)
        inputs[name] = identity['sha256']
    for path in (trial.DESIGN, source_path, trial.OFFICIAL.with_name('config.yaml')):
        inputs[str(path.relative_to(trial.ROOT))] = sha256(path)
    identity = {}
    for name, expected in inputs.items():
        path = trial.ROOT / name
        if sha256(path) != expected:
            raise ValueError('Inherited input changed: ' + name)
        identity[name] = dict(bytes=path.stat().st_size, sha256=expected)
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                     'script/a3', 'script/project_tools'], text=True).splitlines()
    code = {name: sha256(trial.ROOT / name) for name in sorted(set(files)) if name.endswith('.py')}
    for name in code:
        compile((trial.ROOT / name).read_text(), name, 'exec')
    environment = original['environment_source_sha256']
    for name, expected in environment.items():
        if sha256(Path(name)) != expected:
            raise ValueError('Shared environment source changed')
    result = subprocess.run(['/usr/bin/python3', '-B', '-m', 'unittest',
        'script.a3.test_humanplus_replay_loss'], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    record = dict(status='IMPLEMENTED_STDLIB_CONTROLLED_NOT_REGISTERED_NOT_SCIENTIFICALLY_RUN',
        at_utc=trial.runtime.stamp(), parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        text=True).strip(), host='humanplus@100.101.30.68', root=str(trial.ROOT),
        science_variable='TEACHER_REGULARIZER_SAMPLE_COVERAGE_ONLY',
        code_sha256=code, input_identity=identity, environment_source_sha256=environment,
        call_budget=CALL_BUDGET, controls=dict(server_returncode=result.returncode,
        server_output=result.stdout + result.stderr, mac_passed=10,
        scope='STDLIB_BOUNDARIES_NOT_NUMERICAL_OR_SIMULATION_VALIDATION'),
        activity=dict(new_tensor_checkpoint_loads=0, new_policy_forwards=0,
                      new_simulation_steps=0, new_optimizer_steps=0),
        label_reconstruction='ONLY_INSIDE_REGISTERED_REAL_TRAINING_MAXABS_1E-5',
        legacy_teacher_rmse_max=.05, legacy_teacher_maxabs=.5,
        github_status='LOCAL_COMMITTED_UNPUSHED', backup_status='LOCAL_ONLY')
    atomic_json(target, record)
    print(json.dumps(dict(status=record['status'], code_files=len(code), input_files=len(identity),
                         environment_files=len(environment), calls=CALL_BUDGET, controls=10)))


if __name__ == '__main__':
    main()
