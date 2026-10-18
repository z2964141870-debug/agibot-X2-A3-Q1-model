"""Single-use, metadata-only RC01 review under the user's repair/resume request."""

import json
import os
from pathlib import Path
import time

from script.a3 import humanplus_health as health, humanplus_supervisor as guard
from script.a3 import humanplus_recovered_health as recovery
from script.a3.checkpoint_store import atomic_json, sha256


def main():
    root = recovery.ROOT
    if Path.cwd().resolve() != root or Path.home() != root.parent or recovery.RECORD.exists():
        raise ValueError('Single-use RC01 at authorized root only')
    baseline = json.loads(recovery.OLD.read_text())
    after = json.loads(recovery.AFTER.read_text())
    events = after['journal']['fatal']
    if len(events) != 1 or events[0]['message'] != recovery.EXPECTED_MESSAGE:
        raise ValueError('Only the reviewed HP20 Xid43 may be resolved')
    samples = []
    for ordinal in range(3):
        if ordinal:
            time.sleep(max(0., next_at - time.monotonic()))
        next_at = time.monotonic() + 5.
        live = guard.collect(baseline)
        extra = guard.query('gpu_recovery_action')
        if extra != [['None']]:
            raise ValueError('Driver requests recovery action: ' + repr(extra))
        if (live['journal']['fatal'] != events or live['new_oom_events']
                or live['boot_id'] != after['boot_id']):
            raise ValueError('New event or changed boot during RC01')
        issues = guard.issues(live, startup=True, monitor_pid=os.getpid())
        if issues != ['CRITICAL_KERNEL_EVENT']:
            raise ValueError('Additional unresolved issues: ' + repr(issues))
        if any(p['pgid'] == 2970867 or p['pid'] == 2970870 for p in live['processes']):
            raise ValueError('Faulting process/group still exists')
        # Full-prefix verification uses the same original continuity parser.
        health.snapshot(after)
        samples.append(dict(ordinal=ordinal, driver_recovery_action='None',
                            original_guard_issues=issues, evidence=live))
    directory = root / 'logs/a3_humanplus_recovery_20261018_RC01'
    directory.mkdir(exist_ok=False)
    receipt = directory / 'metadata_samples.json'
    atomic_json(receipt, dict(samples=samples, no_cuda_context_or_models=True))
    paths = [recovery.OLD, recovery.AFTER, receipt,
        root / 'data/manifests/a3_humanplus_specialist_20261018_HP20_result.json',
        root / 'data/manifests/a3_humanplus_specialist_20261018_HP20_isaac-baseline_audit.json',
        root / 'data/manifests/a3_humanplus_specialist_bridge_20261018_B01.json']
    identities = {str(p.relative_to(root)): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in paths}
    record = dict(status='EXPLICIT_APPLICATION_RESTART_AUTHORIZED',
        authorization='修，修好了继续优化', boot_id=after['boot_id'], resolved_events=events,
        post_fault_journal=after['journal'], original_baseline_path=str(recovery.OLD.relative_to(root)),
        evidence_identity=identities, driver_recovery_action='None',
        three_metadata_samples_five_seconds_apart=True,
        gpu_execution_recovery_verified=False, old_trial_remains_stopped=True,
        hardware_health_guarantee=False, unique_root_cause_proven=False,
        documentation='https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html',
        new_event_policy='STOP_ON_ANY_ADDITIONAL_EVENT_OR_PREFIX_LOSS',
        backup_status='LOCAL_ONLY', github_synced=False)
    atomic_json(recovery.RECORD, record)
    print(json.dumps(dict(status=record['status'], resolved_events=len(events),
        metadata_samples=3, record_sha256=sha256(recovery.RECORD), no_model_or_simulation=True)))


if __name__ == '__main__':
    main()
