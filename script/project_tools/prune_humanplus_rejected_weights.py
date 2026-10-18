"""Single-use, explicit pruning of closed HP12/13 and rejected HP14/18 weights."""

import argparse
import json
import os
from pathlib import Path
import stat

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_runtime import ROOT, require_root, stamp

NAMES = {
    'HP12': 'a3_humanplus_finite_20261018_HP12',
    'HP13': 'a3_humanplus_head_20261018_HP13',
    'HP14': 'a3_humanplus_head_logged_20261018_HP14',
    'HP17': 'a3_humanplus_head_completion_20261018_HP17',
    'HP18': 'a3_humanplus_replay_loss_20261018_HP18',
}
RUNS = {'HP12': 'R21_001_s0', 'HP13': 'R22_001_s0', 'HP14': 'R23_001_s0', 'HP18': 'R24_001_s0'}
BASE = ROOT / 'data/manifests/a3_humanplus_weight_prune_20261018_S01'
PLAN, RESULT = BASE.with_name(BASE.name + '_plan.json'), BASE.with_name(BASE.name + '_result.json')


def read(path):
    return json.loads(path.read_text())


def eligibility():
    evidence = {}
    for key, name in NAMES.items():
        path = ROOT / 'data/manifests' / (name + '_result.json')
        value = read(path)
        if key in ('HP12', 'HP13', 'HP14'):
            if value['status'] != 'STOPPED':
                raise ValueError('Expected permanently closed stopped trial: ' + key)
        else:
            audit = path.with_name(name + '_effect_audit.json')
            proof = read(audit)
            if (value['status'] != 'COMPLETE_MATCHED_EFFECT' or not proof['verified']
                    or proof['matched_development_gate_passed'] is not False
                    or proof['result_sha256'] != sha256(path)
                    or proof['after_issues'] or proof['remaining_own_group_processes']
                    or value['stages'][-1]['receipt']['matched_development_gate_passed'] is not False):
                raise ValueError('Complete rejected effect and clean independent audit required: ' + key)
            evidence[str(audit.relative_to(ROOT))] = sha256(audit)
        evidence[str(path.relative_to(ROOT))] = sha256(path)
    return evidence


def expected_paths():
    paths = []
    for key, run in RUNS.items():
        for step in ((0, 2, 4) if key == 'HP12' else (0, 2, 4, 8)):
            folder = ROOT / 'data/training' / NAMES[key] / run / ('milestone_' + str(step))
            paths += [folder / f'model_step_{step:06d}.pt', folder / 'last.pt']
    for name in ('a3_humanplus_head_effect_20261018_HP16', NAMES['HP18']):
        paths.append(ROOT / 'data/experiments' / name / 'checkpoint8/model_step_000008.pt')
    return paths


def active_input_check(names):
    for path in (ROOT / 'data/manifests').glob('*HP*_plan.json'):
        value = read(path)
        result_path = path.with_name(path.name.replace('_plan.json', '_result.json'))
        if result_path.exists() and read(result_path).get('status') in ('STOPPED', 'COMPLETE',
                'COMPLETE_MATCHED_EFFECT', 'COMPLETE_MATCHED_BASELINE', 'COMPLETE_PHYSX_CONTROL'):
            continue
        overlap = set(value.get('input_sha256', {})) & set(names)
        if overlap:
            raise ValueError('Active or pending plan owns selected inputs: ' + str(sorted(overlap)))


def system_auth_helper(status, proc=Path('/proc')):
    fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
    if fields.get('Name', '').strip() != '(sd-pam)':
        return False
    parent = proc / fields['PPid'].strip() / 'status'
    values = dict(line.split(':', 1) for line in parent.read_text().splitlines() if ':' in line)
    return values.get('Name', '').strip() == 'systemd' and values.get('PPid', '').strip() == '1'


def open_references(inodes):
    """Inspect task holders; leave the system login helper's protected state alone."""
    matches, checked, excluded = [], 0, []
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            status = (process / 'status').read_text()
            if '\nState:\tZ' in status:
                continue
            name = next(line.split(':', 1)[1].strip() for line in status.splitlines() if line.startswith('Name:'))
            if system_auth_helper(status) or name in ('sshd', 'ssh-agent', 'gpg-agent', 'gnome-keyring-d'):
                excluded.append(dict(pid=int(process.name),
                    name=name, reason='SYSTEM_LOGIN_OR_CREDENTIAL_DAEMON_NOT_COMPUTE_DO_NOT_INSPECT_AUTH_STATE'))
                continue
            checked += 1
            for descriptor in (process / 'fd').iterdir():
                try:
                    info = descriptor.stat()
                except FileNotFoundError:
                    continue
                if (info.st_dev, info.st_ino) in inodes:
                    matches.append(dict(pid=int(process.name), kind='fd', path=str(descriptor)))
            for line in (process / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                major, minor = (int(x, 16) for x in fields[3].split(':'))
                if (os.makedev(major, minor), int(fields[4])) in inodes:
                    matches.append(dict(pid=int(process.name), kind='maps', mapping=line))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            raise ValueError('Same-UID process FD/maps cannot be checked: ' + process.name)
    if matches:
        raise ValueError('Selected weights are open or mapped: ' + str(matches))
    return dict(same_uid_processes_checked=checked, selected_open_or_mapped=matches,
                system_login_helpers_not_inspected=excluded,
                other_uid_process_memory_not_inspected=True)


def enumerate_files():
    rows, counts = [], {}
    for path in expected_paths():
        name = str(path.relative_to(ROOT))
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            resolved = path.resolve(strict=True)
            if path.name != 'last.pt' or resolved.parent != path.parent or resolved not in expected_paths():
                raise ValueError('Unexpected alias target: ' + name)
            rows.append(dict(path=name, kind='symlink', target=target, bytes=info.st_size))
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or path.resolve() != path:
            raise ValueError('Expected owned regular checkpoint: ' + name)
        record = read(path.with_suffix('.json'))
        digest = sha256(path)
        if record['sha256'] != digest or record['bytes'] != info.st_size:
            raise ValueError('Checkpoint sidecar size/SHA changed: ' + name)
        row = dict(path=name, kind='regular', bytes=info.st_size, sha256=digest,
                   device=info.st_dev, inode=info.st_ino, nlink=info.st_nlink,
                   allocated_bytes=info.st_blocks * 512)
        rows.append(row)
        counts.setdefault((info.st_dev, info.st_ino), []).append(row)
    for same_inode in counts.values():
        if any(row['nlink'] != len(same_inode) for row in same_inode):
            raise ValueError('Unlisted hardlink exists; do not claim space reclaimed')
    return rows, counts


def plan():
    require_root()
    if PLAN.exists() or RESULT.exists():
        raise FileExistsError('Prune plan is single-use')
    evidence = eligibility()
    rows, inodes = enumerate_files()
    active_input_check([row['path'] for row in rows])
    refs = open_references(set(inodes))
    result = dict(status='PLANNED_EXPLICIT_REJECTED_WEIGHTS_ONLY', at_utc=stamp(),
        user_authorization='2026-10-18: 过大或确定完全废弃的训练权重可以删除',
        scope='HP12_HP13_CLOSED_NO_FUTURE_USE_HP14_HP18_MATCHED_CANDIDATES_REJECTED',
        files=rows, eligibility_sha256=evidence, process_check=refs,
        regular_paths=sum(r['kind'] == 'regular' for r in rows),
        aliases=sum(r['kind'] == 'symlink' for r in rows), unique_weight_inodes=len(inodes),
        unique_logical_bytes=sum(v[0]['bytes'] for v in inodes.values()),
        expected_allocated_release_bytes=sum(v[0]['allocated_bytes'] for v in inodes.values()),
        protected='OFFICIAL035_REQUIRED_TEACHER_INPUTS_DATA_LOGS_CONFIGS_METRICS_AND_JSON_SIDECARS',
        no_checkpoint_tensor_deserialization=True, no_cloud_backup_claim=True,
        historical_receipts_remain_time_of_measurement_evidence=True)
    atomic_json(PLAN, result)
    print(json.dumps({k: result[k] for k in ('status', 'regular_paths', 'aliases',
        'unique_weight_inodes', 'unique_logical_bytes', 'expected_allocated_release_bytes')}))


def execute():
    require_root()
    if RESULT.exists():
        raise FileExistsError('Prune execution must not repeat')
    original = read(PLAN)
    if original['eligibility_sha256'] != eligibility():
        raise ValueError('Terminal evidence changed after planning')
    rows, inodes = enumerate_files()
    if rows != original['files']:
        raise ValueError('Planned paths, identities or hardlinks changed')
    active_input_check([row['path'] for row in rows])
    refs = open_references(set(inodes))
    before = os.statvfs(ROOT)
    record = dict(status='DELETING', started_at_utc=stamp(), plan_sha256=sha256(PLAN),
                  process_check=refs, unlinked=[], no_cloud_backup_claim=True)
    atomic_json(RESULT, record)
    for row in sorted(rows, key=lambda r: r['kind'] != 'symlink'):
        path = ROOT / row['path']
        info = path.lstat()
        if row['kind'] == 'regular':
            if (info.st_dev, info.st_ino, info.st_size) != (row['device'], row['inode'], row['bytes']):
                raise ValueError('File replaced during scoped deletion')
        elif not path.is_symlink() or os.readlink(path) != row['target']:
            raise ValueError('Alias changed during scoped deletion')
        path.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        record['unlinked'].append(row['path'])
        atomic_json(RESULT, record)
    if any(os.path.lexists(ROOT / row['path']) for row in rows):
        raise ValueError('Selected weight paths remain')
    after = os.statvfs(ROOT)
    record.update(status='COMPLETE_REJECTED_WEIGHTS_DELETED', completed_at_utc=stamp(),
        regular_paths=original['regular_paths'], aliases=original['aliases'],
        unique_weight_inodes=original['unique_weight_inodes'],
        unlinked_allocated_bytes=original['expected_allocated_release_bytes'],
        unique_logical_bytes=original['unique_logical_bytes'],
        free_bytes_before=before.f_bavail * before.f_frsize,
        free_bytes_after=after.f_bavail * after.f_frsize,
        filesystem_available_delta_bytes=(after.f_bavail * after.f_frsize - before.f_bavail * before.f_frsize),
        free_space_delta_may_include_parallel_filesystem_activity=True,
        official035_sha256=sha256(ROOT / 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'),
        retained_records_are_historical_not_current_weight_availability=True)
    atomic_json(RESULT, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('unlinked', 'process_check')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'execute'))
    {'plan': plan, 'execute': execute}[parser.parse_args().action]()
