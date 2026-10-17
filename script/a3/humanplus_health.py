"""Read-only cgroup-v2 and kernel evidence for the humanplus A3 workspace.

This module does not register, launch, resume, or stop scientific workloads.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


GIB = 1024 ** 3
OOM_KEYS = ('oom', 'oom_kill', 'oom_group_kill')
KERNEL_FATAL = re.compile(
    r'Hardware Error|Machine check events logged|Kernel panic|NVRM: Xid|'
    r'GPU has fallen off|Out of memory: Killed process|BUG:|watchdog:.*lockup|'
    r'oom-kill|oom-killer|Memory cgroup out of memory', re.I)


class TelemetryError(RuntimeError):
    pass


def integer(value):
    if not value.isdecimal():
        raise TelemetryError('Invalid nonnegative integer')
    return int(value)


def counters(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if key in result:
            raise TelemetryError('Duplicate memory event key')
        result[key] = integer(value)
    if not set(OOM_KEYS) <= result.keys():
        raise TelemetryError('Missing OOM event counters')
    return result


def mount_path(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), value)


def cgroup_location(membership, mountinfo):
    rows = membership.splitlines()
    if len(rows) != 1 or not rows[0].startswith('0::/'):
        raise TelemetryError('Unified cgroup-v2 membership required')
    group = PurePosixPath(rows[0][3:])
    if '..' in group.parts or '.' in rows[0][3:].split('/'):
        raise TelemetryError('Invalid cgroup membership path')
    mounts = []
    for line in mountinfo.splitlines():
        left, right = line.split(' - ', 1)
        if right.split()[0] == 'cgroup2':
            fields = left.split()
            mounts.append((PurePosixPath(mount_path(fields[3])),
                           Path(mount_path(fields[4]))))
    if len(mounts) != 1:
        raise TelemetryError('Exactly one cgroup-v2 mount required')
    mount_root, mount = mounts[0]
    # Hidden ancestors could impose an unobservable limit or OOM counter.
    if mount_root != PurePosixPath('/') or not mount.is_absolute():
        raise TelemetryError('Full visible cgroup hierarchy required')
    return mount, mount.joinpath(*group.parts[1:]), str(group)


def memory_evidence(proc=Path('/proc')):
    memory = {}
    for line in (proc / 'meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable'):
            fields = value.split()
            if len(fields) != 2 or fields[1] != 'kB':
                raise TelemetryError('Unknown meminfo unit')
            memory[name] = integer(fields[0]) * 1024
    if not 0 <= memory['MemAvailable'] <= memory['MemTotal']:
        raise TelemetryError('Invalid physical memory accounting')
    mount, current, membership = cgroup_location(
        (proc / 'self/cgroup').read_text(), (proc / 'self/mountinfo').read_text())
    available = memory['MemAvailable']
    ancestors = []
    while True:
        filenames = ('memory.current', 'memory.max', 'memory.high',
                     'memory.events', 'memory.events.local')
        if current == mount and not any((current / name).exists() for name in filenames):
            # The real v2 hierarchy root has no memory controller limit files.
            ancestors.append({'path': str(current), 'current_bytes': None,
                'memory.max': 'root_unconstrained', 'memory.high': 'root_unconstrained',
                'memory.events': {}, 'memory.events.local': {},
                'root_accounting': 'physical MemAvailable and full-boot kernel journal'})
            break
        row = {'path': str(current), 'current_bytes': integer(
            (current / 'memory.current').read_text().strip())}
        for filename in ('memory.max', 'memory.high'):
            value = (current / filename).read_text().strip()
            row[filename] = value if value == 'max' else integer(value)
            if value != 'max':
                available = min(available, max(0, int(value) - row['current_bytes']))
        for filename in ('memory.events', 'memory.events.local'):
            row[filename] = counters((current / filename).read_text())
        ancestors.append(row)
        if current == mount:
            break
        current = current.parent
    return {'physical_bytes': memory, 'membership': membership,
            'ancestors': ancestors, 'available_bytes': available,
            'accounting': 'min(MemAvailable, each finite max/high minus current); no cache credit'}


def memory_changes(before, after):
    if before['membership'] != after['membership']:
        raise TelemetryError('Cgroup membership changed')
    old = {row['path']: row for row in before['ancestors']}
    new = {row['path']: row for row in after['ancestors']}
    if old.keys() != new.keys():
        raise TelemetryError('Visible cgroup ancestors changed')
    events = []
    for path, row in new.items():
        previous = old[path]
        if any(row[key] != previous[key] for key in ('memory.max', 'memory.high')):
            raise TelemetryError('Cgroup limits changed')
        for filename in ('memory.events', 'memory.events.local'):
            if row[filename].keys() != previous[filename].keys():
                raise TelemetryError('Cgroup counter schema changed')
            for key, count in row[filename].items():
                delta = count - previous[filename][key]
                if delta < 0:
                    raise TelemetryError('Cgroup counters decreased')
                if key in OOM_KEYS and delta:
                    events.append({'path': path, 'source': filename,
                                   'event': key, 'increase': delta})
    return events


def journal_evidence(text, boot_id, baseline=None):
    rows = [json.loads(line) for line in text.splitlines()]
    if not rows:
        raise TelemetryError('Empty current-boot kernel journal')
    digest = hashlib.sha256()
    cursors = set()
    previous_time = -1
    prefix = None
    fatal = []
    for index, row in enumerate(rows, 1):
        cursor = row['__CURSOR']
        stamp = integer(row['__MONOTONIC_TIMESTAMP'])
        message = row['MESSAGE']
        if (not isinstance(cursor, str) or not cursor or cursor in cursors
                or row['_BOOT_ID'] != boot_id.replace('-', '')
                or stamp < previous_time or not isinstance(message, str)):
            raise TelemetryError('Inconsistent kernel journal sequence')
        cursors.add(cursor)
        previous_time = stamp
        entry = [cursor, row['_BOOT_ID'], stamp, message]
        digest.update(json.dumps(entry, ensure_ascii=True, separators=(',', ':')).encode() + b'\n')
        if baseline and index == baseline['records']:
            prefix = digest.hexdigest()
        if KERNEL_FATAL.search(message):
            fatal.append({'cursor': cursor, 'message': message})
    if baseline and (baseline['boot_id'] != boot_id or prefix != baseline['sha256']):
        raise TelemetryError('Kernel journal baseline lost or changed')
    return {'boot_id': boot_id, 'records': len(rows), 'sha256': digest.hexdigest(),
            'first_cursor': rows[0]['__CURSOR'], 'last_cursor': rows[-1]['__CURSOR'],
            'fatal': fatal, 'scope': 'sampled visible journal; not proof against unlogged events'}


def snapshot(baseline=None, root=Path('/home/humanplus/YUANQI_A3')):
    if root.resolve() != Path('/home/humanplus/YUANQI_A3') or Path.home() != Path('/home/humanplus'):
        raise TelemetryError('Only the authorized humanplus workspace is supported')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if baseline and baseline['boot_id'] != boot:
        raise TelemetryError('Boot changed')
    memory = memory_evidence()
    result = subprocess.run(['journalctl', '-k', '-b', '-o', 'json', '--all', '--no-pager'],
                            capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        raise TelemetryError('Current-boot kernel journal unreadable')
    journal = journal_evidence(result.stdout, boot, baseline['journal'] if baseline else None)
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
        raise TelemetryError('Boot changed during snapshot')
    changes = memory_changes(baseline['memory'], memory) if baseline else []
    return {'boot_id': boot, 'memory': memory, 'journal': journal,
            'new_oom_events': changes, 'runtime_validated': False}


def reasons(evidence, *, startup):
    result = []
    if evidence['journal']['fatal']:
        result.append('CRITICAL_KERNEL_EVENT')
    if evidence['new_oom_events']:
        result.append('CGROUP_OOM_EVENT')
    if evidence['memory']['available_bytes'] < (8 if startup else 4) * GIB:
        result.append('STARTUP_MEMORY_BELOW_8GIB' if startup else 'MEMORY_RESERVE_BELOW_4GIB')
    return result
