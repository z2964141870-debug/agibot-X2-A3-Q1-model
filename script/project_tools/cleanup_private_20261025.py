"""One-time, manifest-first retirement of explicitly authorized A3 artifacts."""

import argparse
import collections
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
WEIGHT_ROOTS = [OPS / 'data/training', OPS / 'data/supermimic_for_a3',
                OPS / 'data/experiments', OLD / 'data/training', OLD / 'data/experiments']
ENV_ROOTS = [OLD / 'data/environments' / name for name in (
    'isaac-runtime-raw', 'isaac-runtime_incomplete_01',
    'isaac-runtime_incomplete_02', 'a3-sonic')]
ALLOWED = WEIGHT_ROOTS + ENV_ROOTS + [OPS / 'script/tools']
KEEP = {
    OPS / 'data/training/p2_main/m11d/checkpoints/milestone_32/model_step_000032.pt',
    OPS / 'data/training/p12_fb/a0/checkpoints/milestone_64/model_step_000064.pt',
    OPS / 'data/training/a3_zhy_rootchannel_20261019_E53/arm_i_tier32/milestone_32/model_step_000032.pt',
    OPS / 'data/training/a3_zhy_projection_finetune_20261018_E52/arm_c/milestone_8/model_step_000008.pt',
    *[OPS / 'data/supermimic_for_a3' / name / 'final.pt' for name in (
        'R02_20261023', 'R03_20261023', 'R04_20261023', 'R05A_H32_20261023')],
}
SPEC = Q1 / 'data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/spec.json'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def within(path, roots):
    return any(path == root or root in path.parents for root in roots)


def validate_path(path, allowed, protected, owned_hardlinks=frozenset()):
    path = Path(os.path.abspath(path))
    if not within(path, allowed) or path in protected:
        raise ValueError('Outside authorized candidates or protected: ' + str(path))
    # Never traverse a substituted parent directory or follow an environment link.
    if path.parent.resolve() != path.parent:
        raise ValueError('Symlink in parent path: ' + str(path))
    st = path.lstat()
    if (st.st_dev, st.st_ino) in protected.values():
        raise ValueError('Protected inode: ' + str(path))
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
        raise ValueError('Not a regular file or link: ' + str(path))
    if st.st_nlink != 1 and (st.st_dev, st.st_ino) not in owned_hardlinks:
        raise ValueError('Shared inode requires separate audit: ' + str(path))
    return st


def file_record(path, reason, protected, allowed=ALLOWED, owned_hardlinks=frozenset()):
    st = validate_path(path, allowed, protected, owned_hardlinks)
    link = os.readlink(path) if stat.S_ISLNK(st.st_mode) else None
    digest = hashlib.sha256(os.fsencode(link)).hexdigest() if link is not None else sha256(path)
    row = {'path': str(path), 'reason': reason, 'bytes': st.st_size,
           'allocated': st.st_blocks * 512, 'device': st.st_dev, 'inode': st.st_ino,
           'nlink': st.st_nlink, 'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns,
           'sha256': digest, 'link': link}
    after = path.lstat()
    if any(getattr(after, key) != getattr(st, key) for key in (
            'st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_mode')):
        raise ValueError('File changed while hashing: ' + str(path))
    return row


def check_record(row, allowed, protected, owned_hardlinks=frozenset(), own_changes=None):
    path = Path(row['path'])
    st = validate_path(path, allowed, protected, owned_hardlinks)
    actual = [st.st_size, st.st_blocks * 512, st.st_dev, st.st_ino,
              st.st_nlink, st.st_mtime_ns, st.st_ctime_ns]
    expected = [row[k] for k in ('bytes', 'allocated', 'device', 'inode', 'nlink', 'mtime_ns', 'ctime_ns')]
    changed = (own_changes or {}).get((st.st_dev, st.st_ino))
    if changed:
        expected[4], expected[6] = changed
    if actual != expected:
        raise ValueError('Identity changed since plan: ' + str(path))
    return path


def owned_env_hardlinks(roots=ENV_ROOTS):
    groups = collections.defaultdict(list)
    for root in roots:
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                p = Path(base) / name
                st = p.lstat()
                if (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)) and st.st_nlink > 1:
                    groups[(st.st_dev, st.st_ino)].append((p, st.st_nlink))
    return {key for key, rows in groups.items() if len(rows) == rows[0][1]
            and all(nlink == len(rows) for _, nlink in rows)}


def unlink_record(row, allowed, protected, owned_hardlinks, own_changes):
    p = check_record(row, allowed, protected, owned_hardlinks, own_changes)
    if row['link'] is not None and os.readlink(p) != row['link']:
        raise ValueError('Candidate link changed: ' + str(p))
    flags = (os.O_PATH if row['link'] is not None else os.O_RDONLY) | os.O_NOFOLLOW
    descriptor = os.open(p, flags) if row['nlink'] > 1 else None
    try:
        p.unlink()
        if descriptor is not None:
            st = os.fstat(descriptor)
            own_changes[(st.st_dev, st.st_ino)] = (st.st_nlink, st.st_ctime_ns)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def process_references():
    refs, errors = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for token in (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0'):
                value = token.split('=', 1)[-1]
                if value.startswith('/') and '\n' not in value:
                    refs.append(value)
        except PermissionError:
            errors.append([int(proc.name), 'cmdline'])
        except (FileNotFoundError, ProcessLookupError):
            pass
        for field in ('cwd', 'exe'):
            try:
                refs.append(os.readlink(proc / field).removesuffix(' (deleted)'))
            except PermissionError:
                errors.append([int(proc.name), field])
            except (FileNotFoundError, ProcessLookupError):
                pass
        try:
            refs.extend(os.readlink(fd).removesuffix(' (deleted)') for fd in (proc / 'fd').iterdir())
        except PermissionError:
            errors.append([int(proc.name), 'fd'])
        except (FileNotFoundError, ProcessLookupError):
            pass
        try:
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith('/'):
                    refs.append(fields[5].removesuffix(' (deleted)'))
        except PermissionError:
            errors.append([int(proc.name), 'maps'])
        except (FileNotFoundError, ProcessLookupError):
            pass
    return set(refs), errors


def protected_files():
    records = json.loads(SPEC.read_text())['frozen_files']
    paths = {Path(row['path']) for row in records} | {SPEC} | {p for p in KEEP if p.exists()}
    result = {p: (p.stat().st_dev, p.stat().st_ino) for p in paths}
    return result


def candidates():
    for root in WEIGHT_ROOTS:
        for p in root.rglob('*.pt'):
            if not p.is_symlink() and p not in KEEP and p.stat().st_size > 16 * 1024 * 1024:
                yield p, 'RETIRED_A3_CHECKPOINT_KEEP_BASELINES_AND_SELECTED_FINALS'
    for p in OLD.joinpath('data/training').rglob('*.incomplete'):
        if not p.is_symlink():
            yield p, 'ABANDONED_OLD_A3_PARTIAL_CHECKPOINT'
    for p in OLD.joinpath('data/experiments').rglob('*.timeseries.json'):
        metrics = p.with_name(p.name.replace('.timeseries.json', '.metrics.json'))
        if metrics.is_file():
            yield p, 'RETIRED_A3_DENSE_TRACE_METRICS_RETAINED'
    for root in ENV_ROOTS:
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in list(dirs):
                p = Path(base) / name
                if p.is_symlink():
                    dirs.remove(name)
                    yield p, 'UNUSED_LEGACY_ENVIRONMENT'
            for name in files:
                yield Path(base) / name, 'UNUSED_LEGACY_ENVIRONMENT'
    for duplicate, keeper in [('FGP-main.zip', 'FGP-main.zip'),
                              ('CLOTHO-mocap-main.zip', 'CLOTHO-mocap-main_974bd8d9.zip')]:
        p, k = OPS / 'script/tools' / duplicate, OPS / 'data/fgp' / keeper
        if sha256(p) != sha256(k):
            raise ValueError('Archive duplicate content differs: ' + str(p))
        yield p, 'EXACT_ARCHIVE_DUPLICATE_OF:' + str(k)


def plan(output, reuse_plan=None):
    output.mkdir(parents=True, exist_ok=False)
    protected = protected_files()
    owned_hardlinks = owned_env_hardlinks()
    cache = {}
    if reuse_plan:
        with gzip.open(reuse_plan, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                check_record(row, ALLOWED, protected, owned_hardlinks)
                cache[row['path']] = row
    refs, errors = process_references()
    if any(within(Path(p), ENV_ROOTS) for p in refs):
        raise ValueError('An environment candidate is still in use')
    summary = {'status': 'PLANNING', 'time': time.time(), 'groups': {},
               'process_permission_errors': errors, 'keep': [], 'skipped': []}
    for p in sorted(KEEP):
        if p.exists():
            summary['keep'].append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    counts = collections.defaultdict(lambda: {'files': 0, 'bytes': 0, 'allocated': 0})
    seen, seen_inodes = set(), set()
    with gzip.open(output / 'deletion_plan.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for p, reason in candidates():
            if p in seen:
                raise ValueError('Duplicate plan path: ' + str(p))
            seen.add(p)
            if str(p) in refs or p in protected:
                summary['skipped'].append(str(p))
                continue
            row = cache.get(str(p))
            if row is None:
                row = file_record(p, reason, protected, owned_hardlinks=owned_hardlinks)
            elif row['reason'] != reason:
                raise ValueError('Cached reason differs')
            inode = (row['device'], row['inode'])
            row['reclaimable_allocated'] = 0 if inode in seen_inodes else row['allocated']
            seen_inodes.add(inode)
            if reason.startswith('EXACT_ARCHIVE_DUPLICATE_OF:'):
                row['keeper'] = {'path': reason.split(':', 1)[1], 'sha256': row['sha256']}
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            group = 'legacy_environment' if within(p, ENV_ROOTS) else str(next(r for r in ALLOWED if within(p, [r])))
            for key, value in [('files', 1), ('bytes', row['bytes']), ('allocated', row['reclaimable_allocated'])]:
                counts[group][key] += value
            if len(seen) % 50000 == 0:
                print(json.dumps({'planned_files': len(seen), 'groups': dict(counts)}), flush=True)
    summary.update(status='PLANNED', groups=dict(counts), plan_sha256=sha256(output / 'deletion_plan.jsonl.gz'),
                   reused_hash_records=len(cache), owned_hardlink_groups=len(owned_hardlinks),
                   reused_plan_sha256=sha256(reuse_plan) if reuse_plan else None)
    (output / 'plan_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def apply(output):
    summary = json.loads((output / 'plan_summary.json').read_text())
    plan_path = output / 'deletion_plan.jsonl.gz'
    if summary['status'] != 'PLANNED' or sha256(plan_path) != summary['plan_sha256']:
        raise ValueError('Plan integrity or status differs')
    protected = protected_files()
    owned_hardlinks = owned_env_hardlinks()
    refs, errors = process_references()
    before = os.statvfs(PRIVATE)
    totals = collections.Counter()
    # Preflight every planned identity before performing the first unlink.
    with gzip.open(plan_path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            p = check_record(row, ALLOWED, protected, owned_hardlinks)
            if str(p) in refs:
                raise ValueError('Candidate opened by process: ' + str(p))
            if 'keeper' in row and sha256(row['keeper']['path']) != row['keeper']['sha256']:
                raise ValueError('Duplicate keeper changed')
    own_changes = {}
    with gzip.open(plan_path, 'rt') as source, gzip.open(output / 'deletions.jsonl.gz', 'xt') as journal:
        for line in source:
            row = json.loads(line)
            # The plan already hashes content; inode, size, mtime and ctime must
            # still match immediately before unlink, avoiding a second 180GB read.
            unlink_record(row, ALLOWED, protected, owned_hardlinks, own_changes)
            row['result'] = 'DELETED'
            journal.write(json.dumps(row, sort_keys=True) + '\n')
            totals.update(files=1, bytes=row['bytes'], allocated=row['reclaimable_allocated'])
            if totals['files'] % 10000 == 0:
                journal.flush()
                print(json.dumps(dict(totals)), flush=True)
    for root in ENV_ROOTS:
        for base, dirs, files in os.walk(root, topdown=False):
            Path(base).rmdir()
    after = os.statvfs(PRIVATE)
    result = {'status': 'COMPLETED', 'totals': dict(totals),
              'before_available': before.f_bavail * before.f_frsize,
              'after_available': after.f_bavail * after.f_frsize,
              'shared_fs_available_delta': (after.f_bavail - before.f_bavail) * after.f_frsize,
              'process_permission_errors': errors,
              'plan_sha256': summary['plan_sha256'],
              'journal_sha256': sha256(output / 'deletions.jsonl.gz'),
              'note': 'Allocated bytes count each inode once; hardlinks are removed only when every name is owned by the retired environments. Shared filesystem delta includes external activity.'}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'apply'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-plan', type=Path)
    args = parser.parse_args()
    if args.output.parent != Q1 / 'data/storage':
        raise ValueError('The audit output must be inside the designated Q1 storage directory')
    if args.action == 'plan':
        plan(args.output, args.reuse_plan)
    else:
        apply(args.output)
