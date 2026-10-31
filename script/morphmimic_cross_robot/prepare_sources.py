"""Select the exact G1 subset's SOMA sources without creating a GPU context."""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time


ROOT = Path('/opt/data/private/ZHY')
BOOT = '4d859008-4c83-4a6d-814d-cbb9dfe885c9'
MIB = 1024 ** 2
GIB = 1024 ** 3


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def identity(path):
    value = path.stat()
    return dict(device=value.st_dev, inode=value.st_ino, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            result.update(block)
    return result.hexdigest()


def source_plan(clips, metadata):
    names = [PurePosixPath(clip).stem for clip in clips]
    if not names or len(set(names)) != len(names):
        raise ValueError('The selected subset must have distinct, nonempty clip stems')
    requested = set(names)
    selected = {}
    for row in metadata:
        name = row['filename']
        if name not in requested:
            continue
        if name in selected:
            raise ValueError('Ambiguous metadata filename: ' + name)
        raw = row['move_soma_uniform_path']
        path = PurePosixPath(raw)
        if (path.is_absolute() or '..' in path.parts or len(path.parts) != 4
                or path.parts[:2] != ('soma_uniform', 'bvh')
                or path.suffix != '.bvh' or path.stem != name):
            raise ValueError('Unexpected SOMA archive path for ' + name)
        selected[name] = dict(clip=clips[names.index(name)], archive_member=str(path),
                              output_path=str(PurePosixPath(*path.parts[2:])))
    missing = sorted(requested - selected.keys())
    if missing:
        raise ValueError('Missing metadata rows: ' + ', '.join(missing[:10]))
    return [selected[name] for name in names]


class StreamingInput:
    """Discard only the archive pages already consumed by this sequential reader."""

    def __init__(self, path, check, progress):
        self.stream = path.open('rb', buffering=0)
        self.check = check
        self.progress = progress
        self.bytes_read = 0
        self.advice_bytes = 0
        self.compressed_prefix_sha256 = hashlib.sha256()
        self.last_check = float('-inf')

    def read(self, size):
        now = time.monotonic()
        if now - self.last_check > 2:
            self.check()
            self.progress(self)
            self.last_check = now
        data = self.stream.read(size)
        self.bytes_read += len(data)
        self.compressed_prefix_sha256.update(data)
        complete = (self.bytes_read // (8 * MIB)) * (8 * MIB)
        if complete > self.advice_bytes and hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(self.stream.fileno(), self.advice_bytes,
                            complete - self.advice_bytes, os.POSIX_FADV_DONTNEED)
            self.advice_bytes = complete
        return data

    def close(self):
        self.stream.close()


def extract(archive, plan, destination, check=lambda: None, progress=lambda _: None):
    check()
    destination.mkdir(parents=True, exist_ok=False)
    pending = {row['archive_member']: row for row in plan}
    if len(pending) != len(plan):
        raise ValueError('Repeated archive member in plan')
    outputs = []
    reader = StreamingInput(archive, check, progress)
    try:
        with tarfile.open(fileobj=reader, mode='r|gz') as stream:
            for member in stream:
                row = pending.get(member.name)
                if row is None:
                    continue
                if not member.isfile():
                    raise ValueError('Selected archive member is not a regular file')
                relative = PurePosixPath(row['output_path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe destination path')
                path = destination / str(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied = 0
                source = stream.extractfile(member)
                with path.open('xb') as target:
                    while True:
                        check()
                        block = source.read(MIB)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        copied += len(block)
                if copied != member.size:
                    raise ValueError('Selected member was truncated')
                outputs.append(dict(**row, bytes=copied, sha256=digest.hexdigest()))
                pending.pop(member.name)
                if not pending:
                    break
        if pending:
            raise ValueError('Archive lacks selected members: ' + ', '.join(sorted(pending)[:5]))
        return dict(outputs=outputs, compressed_bytes_read=reader.bytes_read,
                    compressed_prefix_sha256=reader.compressed_prefix_sha256.hexdigest(),
                    prefix_is_full_archive=reader.bytes_read == archive.stat().st_size,
                    clean_archive_page_advice_bytes=reader.advice_bytes,
                    advice_is_actual_freed_bytes=False)
    finally:
        reader.close()


def check_resources(output):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise RuntimeError('Server boot identity changed')
    usage = int(Path('/sys/fs/cgroup/memory/memory.usage_in_bytes').read_text())
    if usage > 28 * GIB:
        raise RuntimeError('Stop this CPU extraction to protect the existing training memory budget')
    if shutil.disk_usage(output).free < 100 * GIB:
        raise RuntimeError('Original 100 GiB free disk reserve would be violated')


def run(args):
    import yaml

    record = dict(status='PREPARING', mode='CPU_SOURCE_EXTRACTION',
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_commit=args.source_commit, source_sha256=sha256(Path(__file__)),
                  pid=os.getpid(), gpu_context_created=False,
                  new_optimizer_updates=0, new_simulation_steps=0,
                  existing_training_must_continue=True, automatic_retry=False)
    save(args.output / 'status.json', record)
    try:
        check_resources(args.output)
        before = {str(p): identity(p) for p in (args.archive, args.subset, args.metadata)}
        subset = yaml.safe_load(args.subset.read_text())
        with args.metadata.open(newline='') as stream:
            plan = source_plan(subset['clips'], csv.DictReader(stream))
        plan_record = dict(subset=str(args.subset), subset_sha256=sha256(args.subset),
                           metadata=str(args.metadata), metadata_sha256=sha256(args.metadata),
                           archive=str(args.archive), archive_identity=before[str(args.archive)],
                           clips=len(plan), sources=plan)
        save(args.output / 'plan.json', plan_record)
        record.update(status='EXTRACTING', planned_clips=len(plan))
        save(args.output / 'status.json', record)

        def progress(reader):
            save(args.output / 'progress.json', dict(
                status='EXTRACTING', updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                compressed_bytes_read=reader.bytes_read,
                clean_archive_page_advice_bytes=reader.advice_bytes,
                advice_is_actual_freed_bytes=False))

        result = extract(args.archive, plan, args.output / 'bvh',
                         check=lambda: check_resources(args.output), progress=progress)
        after = {str(p): identity(p) for p in (args.archive, args.subset, args.metadata)}
        if before != after:
            raise RuntimeError('Input file identity changed during source extraction')
        result.update(status='COMPLETED', input_identity_before=before,
                      input_identity_after=after, plan_sha256=sha256(args.output / 'plan.json'),
                      clips=len(result['outputs']), total_output_bytes=sum(r['bytes'] for r in result['outputs']),
                      new_simulation_steps=0, new_optimizer_updates=0, retargeting_started=False)
        save(args.output / 'result.json', result)
        record.update(status='COMPLETED', clips=result['clips'], total_output_bytes=result['total_output_bytes'])
    except BaseException as error:
        record.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(args.output / 'status.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'subset', 'metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if len(args.source_commit) != 40 or any(c not in '0123456789abcdef' for c in args.source_commit):
        parser.error('Source commit must be the independently published full Git SHA')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / 'data/supermimic_for_q1'):
        parser.error('Output must be a new directory in the sole Q1 data root')
    for name in ('archive', 'subset', 'metadata'):
        path = getattr(args, name).resolve(strict=True)
        if not path.is_relative_to(ROOT / 'data') or not path.is_file():
            parser.error('Inputs must be existing regular project data files')
        setattr(args, name, path)
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [arg for arg in sys.argv[1:] if arg != '--detach']
        log = ROOT / 'logs' / (args.output.name + '.log')
        with log.open('xb') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      *forwarded, '--worker'], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / 'launch.json', dict(pid=child.pid, mode='CPU_SOURCE_EXTRACTION',
                                             new_simulation_steps=0, new_optimizer_updates=0))
        print(json.dumps(dict(pid=child.pid, output=str(args.output))))
    else:
        if hasattr(os, 'nice'):
            os.nice(10)
        run(args)


if __name__ == '__main__':
    main()
