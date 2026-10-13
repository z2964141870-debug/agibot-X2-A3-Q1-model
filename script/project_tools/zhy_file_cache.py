"""Bounded idle-time page-cache maintenance; never deletes or writes artifacts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3 import zhy_trial as guard
from script.a3.checkpoint_store import atomic_json
from script.a3.stage_resources import processes
from script.diagnostics.host_loadcheck import FATAL

ROOT = Path('/opt/data/private/hechengzhu/ZHY')
LOG = ROOT / 'logs/zhy_file_cache'
GIB = 1024**3
CACHE_HIGH = 4 * GIB
MAX_RELEASE = 2 * GIB
MAX_FILES = 32
INTERVAL = 1800
TERMINAL = {'COMPLETE', 'STOPPED', 'REJECTED', 'FAILED'}
SUFFIXES = {'.pt', '.pth', '.npy', '.npz', '.pkl', '.onnx'}


def memory():
    result = guard.container_memory()
    stat = dict(line.split() for line in (guard.CGROUP / 'memory.stat').read_text().splitlines())
    result['cache_bytes'] = int(stat['total_cache'])
    result['kernel_bytes'] = int((guard.CGROUP / 'memory.kmem.usage_in_bytes').read_text())
    result['failcnt'] = int((guard.CGROUP / 'memory.failcnt').read_text())
    return result


def key(path):
    s = path.stat()
    return s.st_dev, s.st_ino


def metadata(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def inside(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Path escapes ZHY')
    return resolved


def protected_paths(value):
    result = []
    if isinstance(value, dict):
        if isinstance(value.get('path'), str):
            result.append(inside(ROOT / value['path']))
        if isinstance(value.get('input_sha256'), dict):
            result.extend(inside(ROOT / p) for p in value['input_sha256'])
        for item in value.values():
            result.extend(protected_paths(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(protected_paths(item))
    return result


def require_idle(baseline):
    health = guard.health()
    if (health['boot_id'] != baseline['boot_id'] or
            health['container_memory']['oom_kill'] != baseline['baseline_oom_kill']):
        raise RuntimeError('Boot/OOM changed; preserve evidence and stop')
    if health['temperatures_c'].get('x86_pkg_temp', 100) >= 90:
        raise RuntimeError('CPU telemetry unavailable or hot')
    if not health['gpus'] or any(g.get('temperature.gpu', 100) is None or
            g['temperature.gpu'] >= 85 or g.get('utilization.gpu') != 0 for g in health['gpus']):
        raise RuntimeError('GPU busy/hot or telemetry unavailable')
    gpu = guard.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if gpu['returncode'] or gpu['stdout']:
        raise RuntimeError('GPU compute present or unreadable')
    procs = processes()
    for row in procs['rows']:
        if row['pid'] != os.getpid() and (row['rss_bytes'] > GIB or
                row['comm'].lower().startswith(('python', 'kit', 'isaac', 'stress'))):
            raise RuntimeError('Other compute/large process; defer cache maintenance')
    for err in procs['errors']:
        p = Path('/proc') / str(err['pid']) / 'status'
        try:
            fields = dict(x.split(':', 1) for x in p.read_text().splitlines() if ':' in x)
        except FileNotFoundError:
            continue
        if fields['State'].strip().startswith('Z'):
            continue
        if fields['Name'].strip() not in ('pod', 'ssh-agent', 'gpg-agent') or int(fields['VmRSS'].split()[0]) > 16384:
            raise RuntimeError('Unresolved process visibility')
    kernel = guard.command(['dmesg', '--color=never'])
    if kernel['returncode'] or any(FATAL.search(line) and line not in baseline['kernel_baseline']
                                  for line in kernel['stdout'].splitlines()):
        raise RuntimeError('Kernel evidence unavailable or new critical event')
    return health['boot_id']


def residency(paths):
    rows = []
    for start in range(0, len(paths), 100):
        result = subprocess.check_output(['fincore', '-J', '-b', '-o', 'FILE,SIZE,RES,PAGES',
                                          *map(str, paths[start:start+100])], text=True, timeout=20)
        rows.extend(json.loads(result)['fincore'])
    return rows


def maintain(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Only authorized ZHY working directory')
    baseline = json.loads(inside(ROOT / args.baseline).read_text())
    boot = require_idle(baseline)
    plan = json.loads(inside(ROOT / args.protect_plan).read_text())
    protect = set(key(p) for p in protected_paths(plan) if p.is_file())
    roots = []
    for directory in (ROOT / 'data/experiments').iterdir():
        state = directory / 'state.json'
        if not state.is_file() or directory.is_symlink():
            continue
        status = json.loads(state.read_text()).get('status', '')
        if status not in TERMINAL:
            registration = directory / 'registration.json'
            if registration.is_file():
                protect.update(key(p) for p in protected_paths(json.loads(registration.read_text())) if p.is_file())
            continue
        roots.extend((directory, ROOT / 'data/training' / directory.name, ROOT / 'logs' / directory.name))
    visibility_gaps = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if 'Z (zombie)' in (proc / 'status').read_text():
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    protect.add(key(fd))
                except FileNotFoundError:
                    continue
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith(str(ROOT)):
                    p = Path(fields[5])
                    if p.is_file():
                        protect.add(key(p))
        except FileNotFoundError:
            continue
        except PermissionError:
            name = (proc / 'comm').read_text().strip()
            if name not in ('pod', 'ssh-agent', 'gpg-agent'):
                raise
            visibility_gaps.append({'pid': int(proc.name), 'comm': name})
    before = memory()
    record = dict(at=time.time(), boot_id=boot, before=before, targets=[],
                  fd_maps_visibility_gaps=visibility_gaps, mode='TRIM' if args.trim else 'INSPECT',
                  policy=dict(cache_high_bytes=CACHE_HIGH, max_release_bytes=MAX_RELEASE,
                              max_files=MAX_FILES, minimum_interval_seconds=INTERVAL),
                  file_deletions=0, global_drop_caches=False, accounting_changed=False)
    latest = LOG / 'last_trim.json'
    if before['cache_bytes'] <= CACHE_HIGH:
        return dict(record, status='BELOW_CACHE_HIGH_WATERMARK')
    if args.trim and latest.exists() and time.time() - json.loads(latest.read_text())['at'] < INTERVAL:
        return dict(record, status='COOLDOWN')
    paths, seen, visited = [], set(), 0
    for base in roots:
        if not base.is_dir() or base.is_symlink():
            continue
        for directory, subdirs, files in os.walk(base, followlinks=False):
            subdirs[:] = [d for d in subdirs if not (Path(directory)/d).is_symlink()]
            for name in files:
                visited += 1
                if visited > 10000:
                    raise RuntimeError('Metadata scan budget exceeded')
                p = Path(directory) / name
                if p.is_symlink() or p.suffix not in SUFFIXES or p.stat().st_size < 1024**2:
                    continue
                inside(p)
                inode = key(p)
                if inode not in seen and inode not in protect:
                    seen.add(inode)
                    paths.append(p)
    candidates = sorted(residency(paths), key=lambda row: (-int(row['res']), row['file']))
    total = 0
    for row in candidates:
        resident = int(row['res'])
        if resident < 1024**2 or resident + total > MAX_RELEASE:
            continue
        record['targets'].append(row)
        total += resident
        if len(record['targets']) == MAX_FILES:
            break
    record.update(status='READY' if total else 'NO_ELIGIBLE_RESIDENT_PROJECT_FILES',
                  candidate_resident_bytes=total, inspected_artifacts=len(paths))
    if not args.trim or not total:
        return record
    require_idle(baseline)
    atomic_json(LOG / ('intent_' + str(time.time_ns()) + '.json'), record)
    for row in record['targets']:
        p = Path(row['file'])
        before_meta = metadata(p)
        if key(p) in protect or p.is_symlink():
            raise RuntimeError('Target became protected')
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if tuple(os.fstat(fd)[i] for i in (2, 1)) != before_meta[:2]:
                raise RuntimeError('Target inode changed')
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        if metadata(p) != before_meta:
            raise RuntimeError('Artifact metadata changed')
        row['identity_size_mtime_ctime_before_after'] = before_meta
    time.sleep(5)
    record.update(status='TRIMMED', after=memory(), residency_after=residency([Path(r['file']) for r in record['targets']]))
    record['target_resident_decrease_bytes'] = total - sum(int(r['res']) for r in record['residency_after'])
    atomic_json(latest, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('--protect-plan', required=True)
    parser.add_argument('--baseline', required=True)
    options = parser.parse_args()
    LOG.mkdir(exist_ok=True)
    with (LOG / 'maintenance.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = maintain(options)
        atomic_json(LOG / ('receipt_' + str(time.time_ns()) + '.json'), result)
        print(json.dumps(result, indent=2))
