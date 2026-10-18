"""Preserve the HP05 terminal receipts without rerunning any runtime control."""

from datetime import datetime, timezone
import json
from pathlib import Path

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_health import snapshot, reasons


def main():
    root = Path('/home/humanplus/YUANQI_A3')
    if Path.cwd().resolve() != root or Path.home() != root.parent:
        raise ValueError('Authorized humanplus workspace required')
    name = 'a3_humanplus_runtime_20261018_HP05'
    manifest = root / 'data/manifests'
    output = manifest / (name + '_audit.json')
    if output.exists():
        raise FileExistsError('Terminal audit already preserved')
    plan_path = manifest / (name + '_plan.json')
    plan = json.loads(plan_path.read_text())
    result_path = manifest / (name + '_result.json')
    result = json.loads(result_path.read_text())
    if result['status'] != 'STOPPED':
        raise ValueError('Expected the actual terminal HP05 state')
    samples, stages = [], []
    for name_stage in plan['stages']:
        directory = root / 'logs' / name / name_stage
        state = json.loads((directory / 'state.json').read_text())
        stage_samples = [json.loads(line) for line in (directory / 'health.jsonl').read_text().splitlines()]
        samples.extend(stage_samples)
        row = dict(stage=name_stage, state=state, samples=len(stage_samples),
                   files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                          for path in directory.iterdir() if path.is_file()})
        if (directory / 'result.json').exists():
            row['receipt'] = json.loads((directory / 'result.json').read_text())
        stages.append(row)
    console = (root / 'logs' / name / 'physx/console.log').read_text()
    signature = 'Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF when reading a line'
    if signature not in console or stages[-1]['state']['status'] != 'STOPPED':
        raise ValueError('Actual license bootstrap failure signature changed')
    baseline = json.loads((root / 'data/experiments' / name / 'baseline.json').read_text())
    after = snapshot(baseline)
    audit = dict(experiment=plan['experiment'], at_utc=datetime.now(timezone.utc).isoformat(),
        status='STOPPED_ENVIRONMENT_BOOTSTRAP_EULA_EOF', no_retry_performed=True,
        failure_is_confirmed_hardware_fault=False,
        cause='New launcher omitted the existing noninteractive Isaac Sim bootstrap setting; EOF before PhysX context',
        correction_for_review='Existing fresh_batch_trial.py worker sets OMNI_KIT_ACCEPT_EULA=YES; keep HP05 terminal and do not rerun its successful tensor/MuJoCo stages',
        plan_sha256=sha256(plan_path), result_sha256=sha256(result_path),
        code_unchanged=all(sha256(root / name_code) == digest for name_code, digest in plan['code_sha256'].items()),
        stages=stages, samples=len(samples),
        boot_ids=sorted({row['boot_id'] for row in samples}),
        min_available_bytes=min(row['memory']['available_bytes'] for row in samples),
        max_cpu_c=max(row['cpu_celsius'] for row in samples),
        max_gpu_c=max(row['gpu_celsius'] for row in samples),
        fatal_matches=sum(len(row['journal']['fatal']) for row in samples),
        new_oom_matches=sum(len(row['new_oom_events']) for row in samples),
        max_collection_seconds=max(row['collection_seconds'] for row in samples),
        after_health_reasons=reasons(after, startup=False), after_boot=after['boot_id'],
        policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0,
        physx_steps=0, completed_mujoco_steps=10, completed_transaction_tests=4,
        next='Save STOPPED and pause heartbeat; explicit rearm before correcting only the incomplete PhysX validation',
        backup_status='LOCAL_ONLY')
    atomic_json(output, audit)
    print(json.dumps({key: audit[key] for key in ('status', 'code_unchanged', 'samples',
          'max_cpu_c', 'max_gpu_c', 'min_available_bytes', 'after_health_reasons')}))


if __name__ == '__main__':
    main()
