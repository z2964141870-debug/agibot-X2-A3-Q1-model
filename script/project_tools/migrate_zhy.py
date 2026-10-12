#!/usr/bin/env python3
"""Copy an A3 snapshot over an existing SSH control socket, without compute."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time


SOURCE = Path('/media/yu/FAFF-E9771/YUANQI_A3')
DEST = '/opt/data/private/hechengzhu/ZHY'
VENDOR = Path('/media/yu/FAFF-E9771/YUANQI/script/vendor/sonic_for_a3')
HISTORY = Path('/media/yu/FAFF-E9771/YUANQI/data/experiments/a3_fullchain_20261010')
SSH = ['ssh', '-S', '/tmp/zhy-migration-20261012.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', '-p', '25115', 'root@59.77.15.15']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def scan(root, prefix='', skip=()):
    entries = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__'})
        for name in sorted(dirs + files):
            path = Path(base) / name
            rel = path.relative_to(root).as_posix()
            if any(rel == s or rel.startswith(s + '/') for s in skip):
                if name in dirs:
                    dirs.remove(name)
                continue
            if name.endswith(('.pyc', '.pyo')):
                continue
            st = path.lstat()
            row = {'path': prefix + rel, 'source_rel': rel, 'root': str(root),
                   'mode': st.st_mode & 0o777, 'mtime_ns': st.st_mtime_ns}
            if path.is_symlink():
                target = os.readlink(path)
                row.update(kind='symlink', source_target=target)
                if target.startswith('/'):
                    mappings = [(str(SOURCE), DEST),
                                (str(VENDOR), DEST + '/script/vendor/sonic_for_a3'),
                                (str(HISTORY), DEST + '/data/references/a3_fullchain_20261010_history')]
                    mapped = next((b + target[len(a):] for a, b in mappings
                                   if target == a or target.startswith(a + '/')), None)
                    if mapped is None:
                        raise RuntimeError('Unmapped external symlink: ' + str(path))
                    row['target'] = os.path.relpath(mapped, str(Path(DEST) / row['path']).rsplit('/', 1)[0])
                else:
                    row['target'] = target
            elif path.is_dir():
                row['kind'] = 'directory'
            elif path.is_file():
                row.update(kind='file', size=st.st_size, sha256=digest(path))
                after = path.stat()
                if after.st_size != st.st_size or after.st_mtime_ns != st.st_mtime_ns:
                    raise RuntimeError('Source changed during hash: ' + str(path))
            else:
                raise RuntimeError('Unsupported file type: ' + str(path))
            entries.append(row)
    return entries


def transfer(rows, root, dest, logdir, label):
    for row in rows:
        st = (root / row['source_rel']).lstat()
        if st.st_mtime_ns != row['mtime_ns'] or (row['kind'] == 'file' and st.st_size != row['size']):
            raise RuntimeError('Source changed before transfer: ' + row['path'])
    remote('mkdir -p ' + shlex.quote(dest))
    with tempfile.NamedTemporaryFile(dir=logdir, suffix='.nul') as names:
        names.write(b''.join(os.fsencode(r['source_rel']) + b'\0' for r in rows))
        names.flush()
        start = time.monotonic()
        with (logdir / 'transfer.stderr').open('ab') as err:
            sender = subprocess.Popen(['tar', '-C', str(root), '-cf', '-', '--no-recursion',
                                       '--null', '-T', names.name], stdout=subprocess.PIPE, stderr=err)
            receiver = subprocess.Popen(SSH + ['tar -xpf - -C ' + shlex.quote(dest)],
                                        stdin=sender.stdout, stdout=subprocess.DEVNULL, stderr=err)
            sender.stdout.close()
            try:
                recv_rc = receiver.wait(timeout=180)
                send_rc = sender.wait(timeout=15)
            except subprocess.TimeoutExpired:
                receiver.terminate()
                sender.terminate()
                for proc in (receiver, sender):
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise RuntimeError('Slow transfer timeout at ' + label)
        if recv_rc or send_rc:
            raise RuntimeError(f'Tar/SSH failed at {label}: {send_rc}/{recv_rc}')
        seconds = time.monotonic() - start
    size = sum(r.get('size', 0) for r in rows)
    result = {'label': label, 'files': len(rows), 'bytes': size, 'seconds': round(seconds, 3),
              'mib_per_second': round(size / max(seconds, .001) / 1048576, 3)}
    with (logdir / 'batches.jsonl').open('a') as f:
        f.write(json.dumps(result) + '\n')
    print(json.dumps(result), flush=True)
    return result


def verify(entries, logdir):
    payload = [{k: v for k, v in r.items() if k not in {'root', 'source_rel'}} for r in entries]
    script = '''import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows=json.loads(sys.stdin.read()); bad=[]; count=0; total=0
for r in rows:
 p=root/r['path']
 if r['kind']=='symlink':
  if not p.is_symlink() or os.readlink(p)!=r['target']: bad.append(r['path'])
  elif not p.exists(): bad.append(r['path']+' (broken)')
 elif r['kind']=='directory':
  if not p.is_dir(): bad.append(r['path'])
 else:
  if not p.is_file() or p.stat().st_size!=r['size']: bad.append(r['path']); continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4194304),b''): h.update(b)
  if h.hexdigest()!=r['sha256']: bad.append(r['path'])
  count+=1; total+=r['size']
print(json.dumps({'verified_files':count,'verified_bytes':total,'failures':bad}))
sys.exit(bool(bad))
'''
    result = remote('nice -n 15 python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(DEST),
                    input=json.dumps(payload).encode(), stdout=subprocess.PIPE)
    receipt = json.loads(result.stdout)
    (logdir / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', type=Path, required=True)
    args = parser.parse_args()
    logdir = args.logdir.resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    state = {'status': 'INVENTORY', 'source': str(SOURCE), 'destination': DEST,
             'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'excluded': ['.git', '__pycache__', '*.pyc', 'data/environments/a3-sonic'],
             'environment_status': 'NOT_MIGRATED_NONPORTABLE', 'backup_status': 'LOCAL_ONLY'}
    def save():
        (logdir / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        print('Hashing source snapshot; no model imports or CUDA.', flush=True)
        entries = scan(SOURCE, skip=('data/environments/a3-sonic', 'script/vendor/sonic_for_a3',
                                    'data/references/a3_fullchain_20261010_history',
                                    logdir.relative_to(SOURCE).as_posix()))
        entries += scan(VENDOR, 'script/vendor/sonic_for_a3/')
        entries += scan(HISTORY, 'data/references/a3_fullchain_20261010_history/')
        with (logdir / 'source_manifest.jsonl').open('w') as f:
            for row in entries:
                f.write(json.dumps(row) + '\n')
        state.update(status='TRANSFERRING', entries=len(entries),
                     bytes=sum(r.get('size', 0) for r in entries))
        save()
        model = next(r for r in entries if r['path'] == 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt')
        test = transfer([model], SOURCE, DEST, logdir, 'official-model-speed-probe')
        verify([model], logdir)
        state['speed_probe'] = test
        if test['mib_per_second'] < 10:
            state['status'] = 'PAUSED_SLOW_TRANSFER'
            save()
            print('Paused per large-file policy: speed below 10 MiB/s.', flush=True)
            return 3
        state['model_sha256_verified'] = True
        save()
        for root, dest in [(SOURCE, DEST), (VENDOR, DEST + '/script/vendor/sonic_for_a3'),
                           (HISTORY, DEST + '/data/references/a3_fullchain_20261010_history')]:
            rows = [r for r in entries if r['root'] == str(root) and r['kind'] != 'symlink' and r is not model]
            batch = []; size = 0; number = 0
            for row in rows:
                if batch and size + row.get('size', 0) > 512 * 1048576:
                    transfer(batch, root, dest, logdir, root.name + '-' + str(number))
                    number += 1; batch = []; size = 0
                batch.append(row); size += row.get('size', 0)
            if batch:
                transfer(batch, root, dest, logdir, root.name + '-' + str(number))
        links = [r for r in entries if r['kind'] == 'symlink']
        link_code = '''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
for r in json.loads(sys.stdin.read()):
 p=root/r['path']; p.parent.mkdir(parents=True,exist_ok=True)
 if os.path.lexists(p):
  if p.is_symlink() and os.readlink(p)==r['target']: continue
  raise RuntimeError('Target exists: '+str(p))
 p.symlink_to(r['target'])
'''
        remote('python3 -c ' + shlex.quote(link_code) + ' ' + shlex.quote(DEST),
               input=json.dumps(links).encode())
        state['status'] = 'VERIFYING'; save()
        state['verification'] = verify(entries, logdir)
        state.update(status='VERIFIED', ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                     manifest_sha256=digest(logdir / 'source_manifest.jsonl'))
        save()
        print(json.dumps(state), flush=True)
        return 0
    except Exception as exc:
        state.update(status='STOPPED_ERROR', error=str(exc))
        save()
        print(json.dumps(state), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
