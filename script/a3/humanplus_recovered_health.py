"""Explicit RC01 historical-fault handling; original whole-boot checks remain."""

import json
from pathlib import Path
import subprocess

from script.a3 import humanplus_health as original
from script.a3.checkpoint_store import sha256

GIB = original.GIB
TelemetryError = original.TelemetryError
ROOT = Path('/home/humanplus/YUANQI_A3')
RECORD = ROOT / 'data/manifests/a3_humanplus_recovery_20261018_RC01.json'
OLD = ROOT / 'data/experiments/a3_humanplus_specialist_20261018_HP20/baseline.json'
AFTER = ROOT / 'logs/a3_humanplus_specialist_20261018_HP20/isaac-baseline_after_health.json'
EXPECTED_MESSAGE = 'NVRM: Xid (PCI:0000:02:00): 43, pid=2970870, name=python, channel 0x00000003'


def validate_resolution(record, evidence):
    if (record['status'] != 'EXPLICIT_APPLICATION_RESTART_AUTHORIZED'
            or record['authorization'] != '修，修好了继续优化'
            or evidence['boot_id'] != record['boot_id']
            or len(record['resolved_events']) != 1
            or record['resolved_events'][0]['message'] != EXPECTED_MESSAGE):
        raise TelemetryError('Invalid explicit RC01 resolution')
    if evidence['journal']['fatal'] != record['resolved_events']:
        raise TelemetryError('New, missing, or changed kernel event after RC01')
    if evidence['new_oom_events']:
        raise TelemetryError('OOM after historical fault')


def require_record():
    record = json.loads(RECORD.read_text())
    for name, identity in record['evidence_identity'].items():
        path = ROOT / name
        if path.stat().st_size != identity['bytes'] or sha256(path) != identity['sha256']:
            raise TelemetryError('Historical recovery evidence changed: ' + name)
    return record


def snapshot(baseline=None, root=ROOT):
    if root.resolve() != ROOT or Path.home() != ROOT.parent or baseline is None:
        raise TelemetryError('Original humanplus baseline required')
    record = require_record()
    if baseline != json.loads(OLD.read_text()):
        raise TelemetryError('Original pre-fault baseline must remain unchanged')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != baseline['boot_id'] or boot != record['boot_id']:
        raise TelemetryError('Boot changed')
    memory = original.memory_evidence()
    response = subprocess.run(['journalctl', '-k', '-b', '-o', 'json', '--all', '--no-pager'],
                              capture_output=True, text=True, timeout=20)
    if response.returncode:
        raise TelemetryError('Current-boot journal unavailable')
    # Verify both the original pre-fault prefix and the retained post-fault one.
    journal = original.journal_evidence(response.stdout, boot, baseline['journal'])
    original.journal_evidence(response.stdout, boot, record['post_fault_journal'])
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
        raise TelemetryError('Boot changed while sampling')
    result = dict(boot_id=boot, memory=memory, journal=journal,
        new_oom_events=original.memory_changes(baseline['memory'], memory), runtime_validated=False)
    validate_resolution(record, result)
    result['recovery_record_sha256'] = sha256(RECORD)
    result['historical_resolved_kernel_events'] = record['resolved_events']
    result['new_kernel_events'] = []
    return result


def reasons(evidence, *, startup):
    record = require_record()
    validate_resolution(record, evidence)
    if (evidence.get('recovery_record_sha256') != sha256(RECORD)
            or evidence.get('historical_resolved_kernel_events') != record['resolved_events']):
        raise TelemetryError('Unbound historical fault resolution')
    result = original.reasons(evidence, startup=startup)
    # This explicitly resolved historical event remains visible in every row.
    # New events have already failed exact cursor/message and prefix checks.
    result.remove('CRITICAL_KERNEL_EVENT')
    return result
