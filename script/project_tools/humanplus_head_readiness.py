"""Freeze D01 implementation/input provenance without importing tensor libraries."""

import ast
import json
from pathlib import Path
import subprocess

from script.a3 import humanplus_head as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_training_entry import adapted_tree
from script.a3.humanplus_head_protocol import CALL_BUDGET

SOURCES = (
    'script/a3/humanplus_head.py', 'script/a3/humanplus_head_protocol.py',
    'script/a3/humanplus_head_trainer.py', 'script/a3/humanplus_head_entry.py',
    'script/a3/humanplus_head_verification.py', 'script/a3/test_humanplus_head.py',
    'script/project_tools/humanplus_head_audit.py',
    'script/project_tools/humanplus_head_stop_audit.py',
    'script/project_tools/humanplus_head_readiness.py',
)


def main():
    trial.runtime.require_root()
    output = trial.ROOT / 'data/manifests/a3_humanplus_head_readiness_20261018.json'
    if output.exists():
        raise FileExistsError('Implementation provenance freeze is single-use')
    if any(p.exists() for p in (trial.PLAN, trial.RESULT, trial.OUT, trial.LOG, trial.RUN.parent)):
        raise ValueError('Readiness must precede new registration or scientific execution')
    trial.evaluation_base.clean_vendor()
    old = trial.read(trial.ROOT / 'data/manifests/a3_humanplus_finite_20261018_HP12_plan.json')
    design_path = trial.ROOT / 'data/manifests/a3_humanplus_head_design_20261018_D01.json'
    design = trial.read(design_path)
    reviewed = dict(design['read_source_sha256'], **design['static_implementation_review']['source_sha256'])
    for name, expected in reviewed.items():
        path = Path(name) if Path(name).is_absolute() else trial.ROOT / name
        if sha256(path) != expected:
            raise ValueError('Reviewed inheritance or environment source changed: ' + name)
    inputs = {}
    for name, expected in old['input_sha256'].items():
        path = trial.ROOT / name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(path) != expected:
            raise ValueError('Project-relative prior input identity changed: ' + name)
        inputs[name] = dict(bytes=path.stat().st_size, sha256=expected)
    if inputs[str(trial.OFFICIAL.relative_to(trial.ROOT))] != dict(bytes=402050931,
            sha256='9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d'):
        raise ValueError('Original035 identity required')
    environment = dict(old['environment_source_sha256'])
    environment.update({n: d for n, d in reviewed.items() if Path(n).is_absolute()})
    for name, expected in environment.items():
        if sha256(Path(name)) != expected:
            raise ValueError('Pinned shared environment source changed: ' + name)
    paths = subprocess.check_output(['git', 'ls-files', 'script/a3', 'script/project_tools'], text=True).splitlines()
    paths = sorted({n for n in paths + list(SOURCES) if n.endswith('.py')})
    code = {n: sha256(trial.ROOT / n) for n in paths}
    for name in SOURCES:
        compile((trial.ROOT / name).read_text(), name, 'exec')
    vendor_entry = trial.VENDOR / 'gear_sonic/train_agent_trl.py'
    compile(adapted_tree(vendor_entry.read_text(), str(vendor_entry)), str(vendor_entry), 'exec')
    arguments = trial.launch_arguments()
    def options(values):
        return dict(value.split('=', 1) for value in values)
    before, after = options(old['launch_arguments']), options(arguments)
    expected_changes = {'experiment_dir', '++manager_env.commands.motion.motion_lib_cfg.motion_file',
                        'trainer._target_', 'callbacks.model_save._target_'}
    if before.keys() != after.keys() or {n for n in before if before[n] != after[n]} != expected_changes:
        raise ValueError('Launch changed beyond storage and declared head trainer/callback')
    result = dict(status='IMPLEMENTED_STDLIB_CONTROLLED_NOT_REGISTERED_NOT_SCIENTIFICALLY_RUN',
        at_utc=trial.runtime.stamp(), host='humanplus@100.101.30.68', root=str(trial.ROOT),
        parent_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        prospective_experiment=trial.NAME, prospective_run=str(trial.RUN.relative_to(trial.ROOT)),
        code_sha256=code, input_identity=inputs, environment_source_sha256=environment,
        design_sha256=sha256(design_path), original_outer_actor_decoder_call_budget=CALL_BUDGET,
        launcher_changed_fields=sorted(expected_changes), vendor_ast_adapter_compiled=True,
        compile_only_sources=list(SOURCES), new_resource_startup_evidence=False,
        activity=dict(tensor_deserializations=0, model_loads=0, simulation_steps=0, optimizer_steps=0),
        actual_head_shape_gradient_or_training_verified=False,
        github_status='LOCAL_COMMITTED_UNPUSHED', backup_status='LOCAL_ONLY')
    atomic_json(output, result)
    print(json.dumps(dict(status=result['status'], code_files=len(code), input_files=len(inputs),
        environment_sources=len(environment), manifest_sha256=sha256(output))))


if __name__ == '__main__':
    main()
