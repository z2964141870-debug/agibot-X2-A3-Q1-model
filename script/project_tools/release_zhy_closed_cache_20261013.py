"""One-time release of two inactive checkpoint page caches, preserving files."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/opt/data/private/hechengzhu/ZHY')
OUT = ROOT / 'logs/zhy_memory_20261013/cache_release.json'
TARGETS = (
    'data/training/a3_zhy_cloth_curve_20261013_E44/R18_001_s0/milestone_0/model_step_000000.pt',
    'data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/milestone_2/model_step_000002.pt',
)


def snapshot():
    cg = Path('/sys/fs/cgroup/memory')
    result = {'at': time.time(), 'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    for name in ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt',
                 'memory.oom_control', 'memory.kmem.usage_in_bytes', 'memory.stat'):
        result[name] = (cg / name).read_text()
    return result


def resident(paths):
    return json.loads(subprocess.check_output(
        ['fincore', '-J', '-b', '-o', 'FILE,SIZE,RES,PAGES', *map(str, paths)], text=True))


def identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def main():
    if Path.cwd().resolve() != ROOT or OUT.exists():
        raise RuntimeError('Exact ZHY root required; never repeat this release receipt')
    for exp in ('a3_zhy_cloth_curve_20261013_E44', 'a3_zhy_cloth_failure_20261013_E45'):
        if json.loads((ROOT / 'data/experiments' / exp / 'state.json').read_text())['status'] != 'STOPPED':
            raise RuntimeError('Source experiment must remain stopped')
    e47 = ROOT / 'data/experiments/a3_zhy_adam_direction_20261013_E47'
    if json.loads((e47 / 'state.json').read_text())['status'] != 'REGISTERED':
        raise RuntimeError('E47 must not have started')
    job = json.loads((e47 / 'registration.json').read_text())
    protected = {tuple(identity(ROOT / p)[:2]) for p in job['input_sha256']}
    paths = [ROOT / p for p in TARGETS]
    ids = {tuple(identity(p)[:2]) for p in paths}
    if len(ids) != len(paths) or ids & protected:
        raise RuntimeError('Aliased target or target needed by registered E47')
    coverage, unreadable_base_services = [], []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            status = (proc / 'status').read_text()
            state = next(s for s in status.splitlines() if s.startswith('State:'))
            if 'Z (zombie)' in state:
                continue
            for fd in (proc / 'fd').iterdir():
                try:
                    if tuple(identity(fd)[:2]) in ids:
                        raise RuntimeError('Target is currently open: ' + proc.name)
                except FileNotFoundError:
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith('/'):
                    p = Path(fields[5])
                    if p.exists() and tuple(identity(p)[:2]) in ids:
                        raise RuntimeError('Target is currently mapped: ' + proc.name)
            coverage.append(proc.name)
        except FileNotFoundError:
            continue
        except PermissionError:
            fields = dict(s.split(':', 1) for s in status.splitlines() if ':' in s)
            name = fields['Name'].strip()
            rss = int(fields.get('VmRSS', '0 kB').split()[0]) * 1024
            if name not in ('pod', 'ssh-agent', 'gpg-agent') or rss > 16 * 1024**2:
                raise
            # Record incomplete FD/maps visibility. DONTNEED is a cache hint on a
            # read-only descriptor, including if another process has an open FD;
            # no file deletion, mutation, global cache operation, or service stop.
            unreadable_base_services.append({'pid': proc.name, 'comm': name, 'rss': rss})
    record = {'scope': 'TWO_CLOSED_PROJECT_CHECKPOINT_PAGE_CACHES_ONLY',
              'authorization': 'User 2026-10-13 requested appropriate memory release',
              'before': snapshot(), 'residency_before': resident(paths),
              'checked_live_pids': coverage, 'unreadable_base_services': unreadable_base_services,
              'targets': [], 'status': 'PREPARED'}
    if record['before']['boot'] != job['boot_id']:
        raise RuntimeError('Boot changed')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2))
    for path in paths:
        before = identity(path)
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(1024 * 1024):
                digest.update(block)
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        after = identity(path)
        if before != after:
            raise RuntimeError('File identity/size/timestamps changed')
        record['targets'].append({'path': str(path.relative_to(ROOT)), 'sha256_before': digest.hexdigest(),
                                 'identity_before_after': before, 'opened_read_only': True})
        OUT.write_text(json.dumps(record, indent=2))
    time.sleep(5)
    record.update(status='COMPLETE', after=snapshot(), residency_after=resident(paths))
    OUT.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
