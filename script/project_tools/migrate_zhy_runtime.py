#!/usr/bin/env python3
"""Resume bounded file-only runtime transfer through an authenticated socket."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from script.project_tools import migrate_zhy as migration

BASE = Path('/home/yu/miniconda3/envs/x2-sonic-isaaclab')
RAW = migration.DEST + '/data/environments/isaac-runtime-raw'
LOG = migration.SOURCE / 'logs/zhy_runtime_20261012/raw_copy'


def scan():
    rows = []
    roots = [(BASE, '')]
    for root, prefix in roots:
      for directory, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            rel = prefix + path.relative_to(root).as_posix()
            st = path.lstat()
            row = dict(path='data/environments/isaac-runtime-raw/' + rel,
                       source_rel=path.relative_to(root).as_posix(), root=str(root), mtime_ns=st.st_mtime_ns,
                       mode=st.st_mode & 0o777)
            if path.is_symlink():
                target = os.readlink(path)
                if os.path.isabs(target):
                    if not target.startswith(str(BASE) + '/'):
                        if target.startswith('/home/yu/.local/share/ov/data/exts/v2/') and path.is_dir():
                            roots.append((path.resolve(), rel + '/'))
                            row['kind'] = 'directory'
                            row['materialized_external_source'] = target
                            rows.append(row)
                            continue
                        raise RuntimeError('External runtime symlink: ' + str(path))
                    target = os.path.relpath(RAW + target[len(str(BASE)):],
                                             str(Path(RAW) / rel).rsplit('/', 1)[0])
                row.update(kind='symlink', target=target)
            elif path.is_dir():
                row['kind'] = 'directory'
            elif path.is_file():
                row.update(kind='file', size=st.st_size, sha256=migration.digest(path))
                if path.stat().st_mtime_ns != st.st_mtime_ns:
                    raise RuntimeError('Source changed: ' + str(path))
            else:
                raise RuntimeError('Unsupported entry: ' + str(path))
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', required=True)
    args = p.parse_args()
    migration.SSH[2] = args.socket
    LOG.mkdir(parents=True, exist_ok=True)
    manifest = LOG / 'source_manifest.jsonl'
    state_path = LOG / 'state.json'
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    else:
        print('Hashing original runtime without imports or CUDA.', flush=True)
        rows, state = scan(), {}
        with manifest.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    state.setdefault('completed_batches', [])
    def save(status):
        state.update(status=status, manifest_sha256=migration.digest(manifest),
                     files=sum(r['kind'] == 'file' for r in rows),
                     bytes=sum(r.get('size', 0) for r in rows), backup_status='LOCAL_ONLY')
        state_path.write_text(json.dumps(state, indent=2) + '\n')
    save('TRANSFERRING')
    batches, batch, size, current_root = [], [], 0, None
    for row in rows:
        if row['kind'] == 'symlink' or 'materialized_external_source' in row:
            continue
        if batch and (row['root'] != current_root or size + row.get('size', 0) > 512 * 1048576):
            batches.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row.get('size', 0)
        current_root = row['root']
    if batch:
        batches.append(batch)
    try:
        for number, batch in enumerate(batches):
            if number in state['completed_batches']:
                continue
            root = Path(batch[0]['root'])
            relative_prefix = batch[0]['path'][:-len(batch[0]['source_rel'])]
            destination = migration.DEST + '/' + relative_prefix.rstrip('/')
            result = migration.transfer(batch, root, destination, LOG, 'runtime-' + str(number))
            migration.verify(batch, LOG)
            state['completed_batches'].append(number)
            save('TRANSFERRING')
            if result['bytes'] > 64 * 1048576 and result['mib_per_second'] < 10:
                save('PAUSED_SLOW_TRANSFER')
                return 3
        links = [r for r in rows if r['kind'] == 'symlink']
        code = '''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
for row in json.loads(sys.stdin.read()):
 p=root/row['path']; p.parent.mkdir(parents=True,exist_ok=True)
 if p.is_symlink() and os.readlink(p)==row['target']: continue
 if os.path.lexists(p): raise RuntimeError('Conflicting link: '+str(p))
 p.symlink_to(row['target'])
'''
        migration.remote('python3 -c ' + shlex.quote(code) + ' ' + shlex.quote(migration.DEST),
                         input=json.dumps(links).encode(), stdout=subprocess.PIPE)
        state['verification'] = migration.verify(rows, LOG)
        migration.remote('chown -R root:root ' + shlex.quote(RAW), stdin=subprocess.DEVNULL)
        save('VERIFIED')
        print(json.dumps(state), flush=True)
        return 0
    except Exception as exc:
        state['error'] = str(exc)
        save('STOPPED_ERROR')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
