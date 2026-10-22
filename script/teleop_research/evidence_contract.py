"""Small fail-closed contracts for teleoperation evidence; no torch or simulator imports."""
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

REQUIRED_FILES = ('acc.pt', 'rot.pt', 'device2bone.pt', 'pose.pt', 'joint.pt')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def archive_inventory(path):
    """Inspect central directory without extracting or unpickling tensors.

    A complete entry bundle is a pairing candidate, not proof of alignment or GT.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        groups = {}
        unsafe = []
        checkpoints = []
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            p = Path(name)
            if p.is_absolute() or '..' in p.parts:
                unsafe.append(name)
            if p.name in REQUIRED_FILES:
                groups.setdefault(p.parent.as_posix(), set()).add(p.name)
            if p.suffix.lower() in ('.pt', '.pth', '.onnx', '.ckpt') and 'checkpoints' in p.parts:
                checkpoints.append({'entry': name, 'bytes': info.file_size, 'crc32': info.CRC})
        candidates = [s for s, names in groups.items() if set(REQUIRED_FILES) <= names]
        return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
                'sha256': sha256(path), 'entries': len(z.infolist()),
                'uncompressed_bytes': sum(i.file_size for i in z.infolist()),
                'bundle_candidates': sorted(candidates), 'checkpoints': checkpoints,
                'unsafe_entries': unsafe, 'alignment_verified': False,
                'ground_truth_verified': False, 'fgp_training_overlap': 'unknown',
                'crc_all_members_verified': False}

def full_task_success(metrics, expected_steps):
    """Any fall, missing result, or incomplete rollout is NOT a success.

    Return None for invalid/incomplete evidence rather than converting it to a fall.
    """
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ValueError('positive expected_steps required')
    if not isinstance(metrics, dict) or type(metrics.get('fall')) is not bool:
        return None
    steps = metrics.get('num_policy_steps')
    if type(steps) is not int or steps != expected_steps:
        return None
    return not metrics['fall']

def validate_pairs(rows):
    """Explicit same-session/time/target/controller pairing; never pair by file count."""
    seen = set()
    for r in rows:
        fields = ('session_id', 'subject_id', 'split', 'fgp_input_sha256',
                  'target_sha256', 'policy_sha256', 'start_frame', 'end_frame',
                  'time_alignment_id', 'target_provenance', 'seed', 'method')
        if any(k not in r or r[k] is None or r[k] == '' for k in fields):
            raise ValueError('missing pairing provenance')
        if r['target_provenance'] not in ('independent_gt', 'paired_label_audited'):
            raise ValueError('FGP output cannot serve as independent target')
        if not isinstance(r['start_frame'], int) or not isinstance(r['end_frame'], int) or r['end_frame'] <= r['start_frame']:
            raise ValueError('invalid time window')
        for k in ('fgp_input_sha256', 'target_sha256', 'policy_sha256'):
            if len(r[k]) != 64 or any(c not in '0123456789abcdef' for c in r[k]):
                raise ValueError('invalid SHA256')
        key = (r['session_id'], r['start_frame'], r['end_frame'], r['seed'], r['policy_sha256'], r['method'])
        if key in seen:
            raise ValueError('duplicate evaluation key')
        seen.add(key)
    subject_splits = {}
    for r in rows:
        subject_splits.setdefault(r['subject_id'], set()).add(r['split'])
    if any(len(v) > 1 for v in subject_splits.values()):
        raise ValueError('subject leakage across splits')
    return {'rows': len(rows), 'sessions': len({r['session_id'] for r in rows}),
            'subjects': len(subject_splits), 'splits': dict(Counter(r['split'] for r in rows))}


def validate_comparison(rows, methods, varying='method'):
    """Check recorded task/config matching; not a verification of provenance truth."""
    if varying not in ('method', 'policy'):
        raise ValueError('declare method or policy comparison')
    if len(set(methods)) < 2 or len(methods) != len(set(methods)):
        raise ValueError('at least two unique arms required')
    validate_pairs(rows)
    fixed = ('target_sha256', 'fgp_input_sha256', 'subject_id', 'split',
             'time_alignment_id', 'target_provenance', 'environment_id',
             'retarget_config_sha256', 'reset_config_sha256',
             'control_dt_s', 'expected_steps', 'candidate_budget')
    if varying == 'method':
        fixed += ('policy_sha256',)
    tasks = {}
    for r in rows:
        if any(k not in r or r[k] is None or r[k] == '' for k in fixed):
            raise ValueError('missing comparison configuration')
        if type(r['expected_steps']) is not int or r['expected_steps'] <= 0:
            raise ValueError('invalid rollout length')
        if type(r['candidate_budget']) is not int or r['candidate_budget'] <= 0:
            raise ValueError('invalid candidate budget')
        if not isinstance(r['control_dt_s'], (int, float)) or not math.isfinite(r['control_dt_s']) or r['control_dt_s'] <= 0:
            raise ValueError('invalid control timing')
        key = (r['session_id'], r['start_frame'], r['end_frame'], r['seed'])
        arms = tasks.setdefault(key, {})
        if r['method'] not in methods or r['method'] in arms:
            raise ValueError('unknown or duplicate comparison arm')
        arms[r['method']] = r
    if not tasks:
        raise ValueError('empty comparison')
    for arms in tasks.values():
        if set(arms) != set(methods):
            raise ValueError('incomplete task comparison')
        first = next(iter(arms.values()))
        if any(any(r[k] != first[k] for k in fixed) for r in arms.values()):
            raise ValueError('non-matched task/configuration')
    return {'paired_tasks': len(tasks),
            'independent_session_count': len({k[0] for k in tasks}),
            'methods': list(methods), 'varying': varying}

def wilson(k, n, z=1.959963984540054):
    if not 0 <= k <= n or n <= 0:
        raise ValueError('valid binomial counts required')
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return max(0, center-radius), min(1, center+radius)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump([archive_inventory(p) for p in args.archives], f, ensure_ascii=False, indent=2)
    print('Saved local archive manifest:', output)
