"""Transfer the eleven fixed A3 inputs under the user's renewed authorization."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import time


ROOT = Path('/home/humanplus/YUANQI_A3')
SOURCE = Path('/opt/data/private/hechengzhu/ZHY')
HANDOFF = ROOT / 'data/manifests/a3_humanplus_large_transfer_handoff_20261018.json'
HANDOFF_SHA = '6931368214761a6ec6aa19c76bb82440ec78583750c961e10f5bfde0ba10bb6a'
PLAN = ROOT / 'data/manifests/a3_humanplus_large_transfer_plan_20261018_HP04.json'
RESULT = ROOT / 'data/manifests/a3_humanplus_large_transfer_result_20261018_HP04.json'
STAGING = ROOT / 'data/migrations/a3_humanplus_large_inputs_20261018_HP04'
LOGS = ROOT / 'logs/a3_humanplus_20261018_HP04'
SSH_OPTIONS = ['-o', 'ControlPath=/tmp/yuanqi-zhy-source.sock',
               '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
               '-o', 'ServerAliveCountMax=10']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write('\n')


def checked_path(text, base):
    path = Path(text)
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', text):
        raise ValueError('Unexpected transfer path')
    if '..' in path.parts or path == base or base not in path.parents:
        raise ValueError('Transfer outside the fixed workspace')
    return path


def no_symlinks(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('Symlink in destination path')


def verify(path, item):
    no_symlinks(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('Input is not a regular file')
    if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
        raise ValueError('Input size or SHA mismatch: ' + path.name)


def load_files():
    if sha256(HANDOFF) != HANDOFF_SHA:
        raise ValueError('Handoff manifest changed')
    files = json.loads(HANDOFF.read_text())['files']
    if len(files) != 11 or sum(item['bytes'] for item in files) != 613119744:
        raise ValueError('Unexpected transfer scope')
    for item in files:
        checked_path(item['source_path'], SOURCE)
        checked_path(item['destination_path'], ROOT)
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid input SHA')
    if len({item['destination_path'] for item in files}) != len(files):
        raise ValueError('Duplicate destination')
    return files


def register():
    files = load_files()
    no_symlinks(STAGING)
    no_symlinks(LOGS)
    STAGING.mkdir()
    LOGS.mkdir()
    write_json(PLAN, dict(stage='HP04_REAUTHORIZED_LARGE_TRANSFER',
        status='REGISTERED', at_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User explicitly requested direct transfer even if slow on 2026-10-18',
        handoff_sha256=HANDOFF_SHA, script_sha256=sha256(Path(__file__)),
        files=files, bytes=613119744, transport='OpenSSH SFTP, one file at a time',
        slow_transfer_cutoff_seconds=None, per_file_timeout_seconds=21600,
        automatic_transfer_retries=0, old_hp02_archive_reused=False,
        overwrite_existing=False, models_loaded=0, backup_status='LOCAL_ONLY'))
    print('REGISTERED', flush=True)


def execute():
    files = load_files()
    plan = json.loads(PLAN.read_text())
    if plan['script_sha256'] != sha256(Path(__file__)) or plan['files'] != files:
        raise ValueError('Registered code or inputs changed')
    # Authentication checks do not consume the single transfer execution.
    auth = subprocess.run(['ssh', *SSH_OPTIONS, '-p', '25115',
                           'root@59.77.15.15', 'true'], capture_output=True, timeout=30)
    if auth.returncode:
        print('SOURCE_AUTH_UNKNOWN; no transfer started', flush=True)
        return 2
    if shutil.disk_usage(ROOT).free < 2 * 1024 ** 3:
        raise ValueError('Less than 2 GiB free for transfer staging')
    with (LOGS / 'started.json').open('x') as handle:
        json.dump({'at_utc': datetime.now(timezone.utc).isoformat(),
                   'pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}, handle)
    started = time.monotonic()
    result = dict(stage=plan['stage'], status='RUNNING', files=[],
                  models_loaded=0, optimizer_steps=0, backup_status='LOCAL_ONLY')
    try:
        # Check the exact source bytes before starting network payload transfer.
        source_code = ('import hashlib,json,os,stat; rows=' + repr(files) + '; out=[]\n'
            'for row in rows:\n'
            ' p=row["source_path"]; s=os.lstat(p); assert stat.S_ISREG(s.st_mode)\n'
            ' h=hashlib.sha256()\n'
            ' with open(p,"rb") as f:\n'
            '  for block in iter(lambda:f.read(1048576),b""): h.update(block)\n'
            ' assert s.st_size==row["bytes"] and h.hexdigest()==row["sha256"]\n'
            ' out.append({"path":p,"bytes":s.st_size,"sha256":h.hexdigest()})\n'
            'print("A3_SOURCE_RECEIPT:"+json.dumps(out))')
        receipt = subprocess.run(['ssh', *SSH_OPTIONS, '-p', '25115',
            'root@59.77.15.15', 'python3 -c ' + shlex.quote(source_code)],
            capture_output=True, text=True, timeout=300)
        (LOGS / 'source_check.stdout').write_text(receipt.stdout)
        (LOGS / 'source_check.stderr').write_text(receipt.stderr)
        lines = [line for line in receipt.stdout.splitlines() if line.startswith('A3_SOURCE_RECEIPT:')]
        if receipt.returncode or len(lines) != 1:
            raise RuntimeError('Source size/SHA verification failed')
        result['source_verified'] = json.loads(lines[0].split(':', 1)[1])
        for index, item in enumerate(files):
            target = checked_path(item['destination_path'], ROOT)
            no_symlinks(target)
            if target.exists():
                verify(target, item)
                disposition = 'EXISTING_VERIFIED'
            else:
                partial = STAGING / (str(index).zfill(2) + '.part')
                if partial.exists() or partial.is_symlink():
                    raise FileExistsError('Staging file already exists')
                print(json.dumps({'event': 'TRANSFERRING', 'file': item['filename'],
                                  'bytes': item['bytes']}), flush=True)
                with (LOGS / (str(index).zfill(2) + '.sftp.log')).open('x') as log:
                    transfer = subprocess.run(['sftp', *SSH_OPTIONS, '-P', '25115',
                        '-b', '-', 'root@59.77.15.15'], text=True,
                        input='get "' + item['source_path'] + '" "' + str(partial) + '"\n',
                        stdout=log, stderr=subprocess.STDOUT, timeout=21600)
                if transfer.returncode:
                    raise RuntimeError('SFTP failed: ' + item['filename'])
                verify(partial, item)
                no_symlinks(target.parent)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(partial, target)
                verify(target, item)
                disposition = 'TRANSFERRED_VERIFIED'
            result['files'].append(dict(path=str(target), bytes=item['bytes'],
                sha256=item['sha256'], verified=True, disposition=disposition))
            write_json(LOGS / (str(index).zfill(2) + '.verified.json'), result['files'][-1])
            print(json.dumps({'event': disposition, 'file': item['filename']}), flush=True)
        result['status'] = 'COMPLETE_VERIFIED'
    except Exception as exc:
        result['status'] = 'STOPPED_NO_AUTO_RETRY'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    result['elapsed_seconds'] = time.monotonic() - started
    result['at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(RESULT, result)
    print(json.dumps({'status': result['status'], 'verified': len(result['files']),
                      'result': str(RESULT)}), flush=True)
    return 0 if result['status'] == 'COMPLETE_VERIFIED' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'execute'])
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT or Path.home() != Path('/home/humanplus'):
        raise SystemExit('Run only in the authorized humanplus workspace')
    raise SystemExit(register() if args.action == 'register' else execute())
