"""Move the three authorized project trees into private/ZHY without copying data."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import time


PRIVATE = Path('/opt/data/private')
Q1 = PRIVATE / 'YuanQi_Q1'
OPS = PRIVATE / 'yuanqi_ops'
OLD = PRIVATE / 'hechengzhu/ZHY'
DEST = PRIVATE / 'ZHY'
SOURCE_ROOTS = [Q1, OPS, OLD]
SPEC_REL = 'data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/spec.json'
FINAL_REL = 'data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/run/final.pt'
MAP = {Q1: DEST, OPS: DEST / 'script/yuanqi_ops', OLD: DEST / 'script/legacy_a3'}
for source, namespace in [(OPS, 'yuanqi_ops'), (OLD, 'legacy_a3')]:
    for name in ['data', 'logs', 'reports']:
        MAP[source / name] = DEST / name / namespace
MAP[OPS / 'envs'] = DEST / 'data/environments/yuanqi_ops'
MAP[OPS / 'script/amass_bridge/data'] = DEST / 'data/datasets/amass_bridge'


def mapped(path, mapping=MAP):
    path = Path(path)
    for source in sorted(mapping, key=lambda p: len(p.parts), reverse=True):
        if path == source or source in path.parents:
            return mapping[source] / path.relative_to(source)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    st = Path(path).stat()
    return {'path': str(path), 'bytes': st.st_size, 'sha256': sha256(path)}


def record(path):
    st = path.lstat()
    row = {'path': str(path), 'device': st.st_dev, 'inode': st.st_ino,
           'mode': st.st_mode, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
           'allocated': st.st_blocks * 512}
    if path.is_symlink():
        row.update(type='link', target=os.readlink(path), exists=path.exists(),
                   resolved=str(path.resolve()))
    elif stat.S_ISREG(st.st_mode):
        row['type'] = 'file'
    else:
        row['type'] = 'directory'
    return row


def du(path):
    import subprocess
    return int(subprocess.check_output(['du', '-x', '-B1', '-s', str(path)], text=True).split()[0])


def inventory(output):
    if DEST.exists() or DEST.is_symlink():
        raise ValueError('Destination already exists; do not merge implicitly')
    if any(not p.is_dir() or p.is_symlink() for p in SOURCE_ROOTS):
        raise ValueError('Expected all three original real directories')
    devices = {p.stat().st_dev for p in SOURCE_ROOTS + [DEST.parent]}
    if len(devices) != 1:
        raise ValueError('Only same-filesystem renames are authorized by this tool')
    output.mkdir(parents=True, exist_ok=False)
    fs = os.statvfs(PRIVATE)
    summary = {'status': 'INVENTORIED', 'time': time.time(),
               'mapping': {str(k): str(v) for k, v in MAP.items()},
               'before_du': {str(p): du(p) for p in SOURCE_ROOTS},
               'available': fs.f_bavail * fs.f_frsize, 'counts': {'file': 0, 'link': 0, 'directory': 0}}
    manifest = output / 'before.jsonl.gz'
    excluded = Q1 / 'data/storage'
    with gzip.open(manifest, 'xt') as stream:
        for root in SOURCE_ROOTS:
            for base, dirs, files in os.walk(root, followlinks=False):
                base = Path(base)
                if base == excluded:
                    dirs[:] = []
                    continue
                for name in list(dirs):
                    p = base / name
                    if p == excluded:
                        dirs.remove(name)
                        continue
                    row = record(p)
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    summary['counts'][row['type']] += 1
                    if row['type'] == 'link':
                        dirs.remove(name)
                for name in files:
                    row = record(base / name)
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                    summary['counts'][row['type']] += 1
    spec_path = Q1 / SPEC_REL
    spec = json.loads(spec_path.read_text())
    summary['frozen'] = spec['frozen_files']
    summary['critical'] = [file_identity(spec_path), file_identity(Q1 / FINAL_REL)]
    run_root = spec_path.parent
    for rel in ['storage_guard.json', 'controller_status.json', 'run/status.json']:
        summary['critical'].append(file_identity(run_root / rel))
    for row in summary['frozen']:
        if file_identity(row['path']) != row:
            raise ValueError('Frozen content changed before migration: ' + row['path'])
    summary['manifest_sha256'] = sha256(manifest)
    (output / 'inventory.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['frozen', 'critical']}), flush=True)


def move_and_alias(source, target, journal):
    if not source.is_dir() or source.is_symlink() or target.exists() or target.is_symlink():
        raise ValueError('Move source/destination differs: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    inode = source.stat().st_ino
    os.rename(source, target)
    journal.write(json.dumps({'operation': 'RENAME', 'source': str(source), 'target': str(target), 'inode': inode}) + '\n')
    journal.flush()
    os.fsync(journal.fileno())
    source.symlink_to(target, target_is_directory=True)
    journal.write(json.dumps({'operation': 'COMPATIBILITY_LINK', 'source': str(source), 'target': str(target)}) + '\n')
    journal.flush()
    os.fsync(journal.fileno())
    if source.resolve() != target.resolve() or target.stat().st_ino != inode:
        raise ValueError('Rename/link verification failed')


def apply(output):
    summary = json.loads((output / 'inventory.json').read_text())
    manifest = output / 'before.jsonl.gz'
    if summary['status'] != 'INVENTORIED' or sha256(manifest) != summary['manifest_sha256']:
        raise ValueError('Inventory integrity differs')
    if DEST.exists() or DEST.is_symlink():
        raise ValueError('Destination already exists')
    # The journal remains reachable through the compatibility link after the first move.
    with (output / 'moves.jsonl').open('x') as journal:
        move_and_alias(Q1, DEST, journal)
        for source in [OPS, OLD]:
            move_and_alias(source, MAP[source], journal)
            for name in ['data', 'logs', 'reports'] + (['envs'] if source == OPS else []):
                p = MAP[source] / name
                move_and_alias(p, MAP[source / name], journal)
        move_and_alias(MAP[OPS] / 'script/amass_bridge/data',
                       MAP[OPS / 'script/amass_bridge/data'], journal)
        repairs = 0
        with gzip.open(manifest, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                if row['type'] != 'link' or not row['exists']:
                    continue
                p, expected = mapped(row['path']), mapped(row['resolved'])
                if not p.is_symlink():
                    raise ValueError('An inventoried link was replaced: ' + str(p))
                if p.resolve() != expected.resolve():
                    before = os.readlink(p)
                    p.unlink()
                    p.symlink_to(expected, target_is_directory=expected.is_dir())
                    repairs += 1
                    journal.write(json.dumps({'operation': 'RELOCATE_LINK_TARGET', 'path': str(p),
                                              'before': before, 'after': str(expected)}) + '\n')
                if not p.exists():
                    raise ValueError('Previously valid link became broken: ' + str(p))
        journal.flush()
        os.fsync(journal.fileno())
    result = verify(output)
    result['repaired_links'] = repairs
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def verify(output):
    summary = json.loads((output / 'inventory.json').read_text())
    manifest = output / 'before.jsonl.gz'
    if sha256(manifest) != summary['manifest_sha256']:
        raise ValueError('Inventory content changed')
    failures, mutable_logs = [], []
    counts = {'files': 0, 'links': 0, 'directories': 0}
    with gzip.open(manifest, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            p = mapped(row['path'])
            if not p.exists() and not p.is_symlink():
                failures.append({'path': str(p), 'error': 'MISSING'})
                continue
            st = p.lstat()
            if row['type'] == 'file':
                counts['files'] += 1
                if (st.st_dev, st.st_ino, st.st_mode) != (row['device'], row['inode'], row['mode']):
                    failures.append({'path': str(p), 'error': 'INODE_OR_MODE_CHANGED'})
                if (st.st_size, st.st_mtime_ns) != (row['bytes'], row['mtime_ns']):
                    if any('logs' in Path(row['path']).relative_to(root).parts for root in SOURCE_ROOTS
                           if root in Path(row['path']).parents):
                        mutable_logs.append(str(p))
                    else:
                        failures.append({'path': str(p), 'error': 'SIZE_OR_MTIME_CHANGED'})
            elif row['type'] == 'link':
                counts['links'] += 1
                if not p.is_symlink() or (row['exists'] and p.resolve() != mapped(row['resolved']).resolve()):
                    failures.append({'path': str(p), 'error': 'LINK_TARGET_CHANGED'})
            else:
                counts['directories'] += 1
                if not p.is_dir() or p.is_symlink():
                    failures.append({'path': str(p), 'error': 'DIRECTORY_TYPE_CHANGED'})
    for row in summary['frozen'] + summary['critical']:
        current = file_identity(mapped(row['path']))
        if current['bytes'] != row['bytes'] or current['sha256'] != row['sha256']:
            failures.append({'path': current['path'], 'error': 'FROZEN_OR_CRITICAL_CONTENT_CHANGED'})
        if Path(row['path']).resolve() != mapped(row['path']).resolve():
            failures.append({'path': row['path'], 'error': 'LEGACY_ALIAS_DIFFERS'})
    aliases = {str(p): str(p.resolve()) for p in SOURCE_ROOTS}
    for p in SOURCE_ROOTS:
        if not p.is_symlink() or p.resolve() != MAP[p].resolve():
            failures.append({'path': str(p), 'error': 'ROOT_COMPATIBILITY_LINK_INVALID'})
    result = {'status': 'PASSED' if not failures else 'FAILED', 'verified': counts,
              'frozen_files': len(summary['frozen']), 'critical_files': len(summary['critical']),
              'failures': failures, 'mutable_logs': mutable_logs, 'aliases': aliases,
              'manifest_sha256': summary['manifest_sha256'], 'canonical_root': str(DEST),
              'after_du': du(DEST), 'copy_operations': 0,
              'note': 'Regular files retain inode/size/mtime; frozen and critical files also retain SHA256. Old content-based identities remain; path-bound legacy launchers require explicit relocation handling.'}
    if failures:
        (output / 'verification_failure.json').write_text(json.dumps(result, indent=2) + '\n')
        raise ValueError('Migration verification failed: ' + json.dumps(failures[:10]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'apply', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent not in [Q1 / 'data/storage', DEST / 'data/storage']:
        raise ValueError('Audit files must be in the designated storage directory')
    if args.action == 'verify':
        print(json.dumps(verify(args.output)))
    else:
        (inventory if args.action == 'inventory' else apply)(args.output)
