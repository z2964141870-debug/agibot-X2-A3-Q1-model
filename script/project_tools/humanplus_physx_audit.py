"""Archive HP06 terminal evidence without importing or relaunching PhysX."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from script.a3 import humanplus_physx_validation as runtime
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256


def main():
    runtime.prior.require_root()
    output = runtime.ROOT / 'data/manifests' / (runtime.NAME + '_audit.json')
    if output.exists():
        raise FileExistsError('HP06 terminal evidence already preserved')
    result = json.loads(runtime.RESULT.read_text())
    if result['status'] not in ('STOPPED', 'COMPLETE_PHYSX_CONTROL'):
        raise ValueError('HP06 must have a terminal state')
    plan = json.loads(runtime.PLAN.read_text())
    directory = runtime.LOG / 'physx'
    state = json.loads((directory / 'state.json').read_text())
    rows = [json.loads(line) for line in (directory / 'health.jsonl').read_text().splitlines()]
    related = []
    sources = (
        runtime.prior.ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20261018_104955.log',
        Path('/tmp/isaaclab/logs/isaaclab_2026-10-18_10-49-59.log'))
    for source in sources:
        if source.is_file():
            target = runtime.LOG / source.name
            if source.stat().st_size > 2 * 1024**2 or target.exists():
                raise ValueError('Related log copy is not small and new')
            shutil.copy2(source, target)
            if sha256(target) != sha256(source):
                raise ValueError('Related log copy mismatch')
            related.append(dict(source=str(source), saved=str(target.relative_to(runtime.ROOT)),
                                bytes=target.stat().st_size, sha256=sha256(target)))
    baseline = json.loads((runtime.OUT / 'baseline.json').read_text())
    try:
        after = guard.collect(baseline)
        atomic_json(runtime.LOG / 'after_health.json', after)
        after_issues = guard.issues(after, startup=False, monitor_pid=os.getpid())
        survivors = [row for row in after['processes'] if row['pgid'] == state.get('worker_pid')]
    except Exception as exc:
        after_issues = [type(exc).__name__ + ':' + str(exc)]
        survivors = None
    receipt_path = directory / 'result.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    audit = dict(experiment=plan['experiment'], at_utc=datetime.now(timezone.utc).isoformat(),
        status=result['status'], reason=result.get('reason'), state=state,
        authorization=plan['authorization'], no_retry=True,
        code_unchanged=all(sha256(runtime.ROOT / name) == digest for name, digest in plan['code_sha256'].items()),
        environment_sources_unchanged=all(sha256(Path(name)) == digest
            for name, digest in plan['environment_source_sha256'].items()),
        samples=len(rows), max_cpu_c=max(row['cpu_celsius'] for row in rows),
        max_gpu_c=max(row['gpu_celsius'] for row in rows),
        min_available_bytes=min(row['memory']['available_bytes'] for row in rows),
        fatal_matches=sum(len(row['journal']['fatal']) for row in rows),
        oom_matches=sum(len(row['new_oom_events']) for row in rows),
        boot_ids=sorted({row['boot_id'] for row in rows}),
        max_sample_gap_seconds=max(b['monotonic'] - a['monotonic'] for a, b in zip(rows, rows[1:])),
        after_issues=after_issues, remaining_own_group_processes=survivors,
        worker_receipt=receipt, ten_control_steps_verified=receipt is not None and receipt.get('steps') == 10,
        actual_control_steps_unknown=receipt is None,
        bootstrap_eula_eof_present='Do you accept the EULA?' in (directory / 'console.log').read_text(),
        preexisting_crash_warning_not_attributed_to_current_run=True,
        native_runtime_default_logs_written_inside_environment=True,
        package_source_edits=False, related_logs=related,
        plan_sha256=sha256(runtime.PLAN), result_sha256=sha256(runtime.RESULT),
        files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
               for path in directory.iterdir() if path.is_file()},
        policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0,
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(output, audit)
    print(json.dumps({key: audit[key] for key in ('status', 'reason', 'samples', 'max_cpu_c',
          'max_gpu_c', 'ten_control_steps_verified', 'after_issues', 'remaining_own_group_processes')}))


if __name__ == '__main__':
    main()
