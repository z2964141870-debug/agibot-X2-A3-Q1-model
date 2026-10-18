"""Preserve HP07 terminal receipts and native logs without launching simulation."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil

from script.a3 import humanplus_physx_exit_validation as runtime
from script.a3 import humanplus_supervisor as guard
from script.a3.checkpoint_store import atomic_json, sha256


def main(runtime=runtime):
    runtime.previous.prior.require_root()
    output = runtime.PLAN.with_name(runtime.NAME + '_audit.json')
    if output.exists():
        raise FileExistsError('HP07 audit already preserved')
    plan = runtime.checked()
    result = json.loads(runtime.RESULT.read_text())
    if result['status'] not in ('STOPPED', 'COMPLETE_PHYSX_CONTROL'):
        raise ValueError('Terminal state required')
    directory = runtime.LOG / 'physx'
    state = json.loads((directory / 'state.json').read_text())
    rows = [json.loads(line) for line in (directory / 'health.jsonl').read_text().splitlines()]
    progress_path = directory / 'progress.jsonl'
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
    console = (directory / 'console.log').read_text()
    related = []
    allowed = (runtime.previous.prior.ENV / 'lib/python3.11/site-packages/isaacsim/kit/logs',
               Path('/tmp/isaaclab/logs'))
    for name in dict.fromkeys(re.findall(r'Logging to file: (/[^\s\x1b]+\.log)', console)):
        source = Path(name)
        if not any(source.resolve().is_relative_to(parent.resolve()) for parent in allowed):
            raise ValueError('Unexpected native log location')
        target = runtime.LOG / source.name
        if target.exists() or source.stat().st_size > 2 * 1024**2:
            raise ValueError('Native log must be small and new')
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise ValueError('Native log identity mismatch')
        related.append(dict(source=name, saved=str(target.relative_to(runtime.ROOT)),
                            bytes=target.stat().st_size, sha256=sha256(target)))
    after = guard.collect(json.loads((runtime.OUT / 'baseline.json').read_text()))
    atomic_json(runtime.LOG / 'after_health.json', after)
    survivors = [row for row in after['processes'] if row['pgid'] == state.get('worker_pid')]
    issues = guard.issues(after, startup=False, monitor_pid=os.getpid())
    receipts = {name: json.loads((directory / (name + '.json')).read_text())
                if (directory / (name + '.json')).exists() else None for name in ('control', 'result')}
    completed_steps = [row['steps'] for row in progress if row['phase'] == 'step_returned']
    phases = [row['phase'] for row in progress]
    verified = (result['status'] == 'COMPLETE_PHYSX_CONTROL'
                and completed_steps == list(range(1, 11))
                and receipts['control'] is not None and receipts['control']['steps'] == 10
                and receipts['result'] is not None
                and ((receipts['result'].get('shutdown_returned') and 'app_close_returned' in phases)
                     or (receipts['result'].get('native_shutdown_verified')
                         and receipts['result'].get('native_returncode') == 0
                         and 'native_exit_verified' in phases))
                and state.get('returncode') == 0
                and not survivors and not issues)
    audit = dict(experiment=plan['experiment'], at_utc=datetime.now(timezone.utc).isoformat(),
        status=result['status'], reason=result.get('reason'), state=state,
        code_environment_inputs_unchanged=True, no_retry=True,
        samples=len(rows), max_cpu_c=max(row['cpu_celsius'] for row in rows),
        max_gpu_c=max(row['gpu_celsius'] for row in rows),
        min_available_bytes=min(row['memory']['available_bytes'] for row in rows),
        fatal_matches=sum(len(row['journal']['fatal']) for row in rows),
        oom_matches=sum(len(row['new_oom_events']) for row in rows),
        boot_ids=sorted({row['boot_id'] for row in rows}),
        max_sample_gap_seconds=max((b['monotonic'] - a['monotonic']
            for a, b in zip(rows, rows[1:])), default=0),
        after_issues=issues, remaining_own_group_processes=survivors,
        progress=progress, receipts=receipts, verified_control_and_exit=bool(verified),
        last_progress=progress[-1] if progress else None,
        python_timeout_stack_present='Timeout (0:00:30)!' in console,
        related_logs=related, plan_sha256=sha256(runtime.PLAN), result_sha256=sha256(runtime.RESULT),
        files={path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
               for path in directory.iterdir() if path.is_file()},
        policy_model_loads=0, policy_rollouts=0, policy_optimizer_calls=0,
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(output, audit)
    print(json.dumps({key: audit[key] for key in ('status', 'reason', 'samples',
        'max_cpu_c', 'max_gpu_c', 'verified_control_and_exit', 'last_progress',
        'after_issues', 'remaining_own_group_processes')}))


if __name__ == '__main__':
    main()
