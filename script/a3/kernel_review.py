"""Admit only immutable, individually reviewed historical external memcg OOMs."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from script.diagnostics.host_loadcheck import FATAL


def digest(value):
    return hashlib.sha256(value).hexdigest()


def baseline_digest(lines):
    return digest(json.dumps(lines, separators=(',', ':')).encode())


def memory_cgroup(text):
    paths = [line.split(':', 2)[2] for line in text.splitlines()
             if len(line.split(':', 2)) == 3 and
             'memory' in line.split(':', 2)[1].split(',')]
    if len(paths) != 1:
        raise ValueError('Exactly one readable v1 memory cgroup required')
    return normalized_cgroup(paths[0])


def normalized_cgroup(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value or value == '/':
        raise ValueError('Ambiguous memory cgroup')
    return path


def external_oom_line(context, own):
    fatal = [i for i, line in enumerate(context) if FATAL.search(line)]
    if len(fatal) != 1 or fatal[0] == 0:
        raise ValueError('Exactly one attributed OOM per reviewed context required')
    index = fatal[0]
    line = context[index]
    killed = re.match(r'^\[\s*(\d+\.\d+)\] Memory cgroup out of memory: '
                      r'Killed process (\d+) \(', line)
    origin = re.match(r'^\[\s*(\d+\.\d+)\] oom-kill:(.*)$', context[index - 1])
    if not killed or not origin or not 0 <= float(killed[1]) - float(origin[1]) <= 1:
        raise ValueError('Adjacent bounded memcg OOM attribution required')
    pairs = [part.split('=', 1) for part in origin[2].split(',')]
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError('Ambiguous OOM fields')
    fields = dict(pairs)
    if fields.get('constraint') != 'CONSTRAINT_MEMCG' or fields.get('pid') != killed[2]:
        raise ValueError('OOM scope or PID mismatch')
    scope = normalized_cgroup(fields.get('oom_memcg', ''))
    task = normalized_cgroup(fields.get('task_memcg', ''))
    if scope != task or scope == own or scope in own.parents or own in scope.parents:
        raise ValueError('OOM belongs to this cgroup, a related cgroup, or an ambiguous task')
    return line


def read_verified(root, reference):
    root = Path(root).resolve()
    path = (root / reference['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Review file escapes project')
    raw = path.read_bytes()
    if digest(raw) != reference['sha256']:
        raise ValueError('Historical kernel review identity changed')
    return json.loads(raw)


def check_kernel(root, kernel, baseline, reference=None, *, boot_id, cgroup_text):
    if kernel.get('returncode') != 0:
        raise RuntimeError('Kernel telemetry unreadable')
    old = set(baseline['kernel_baseline'])
    fresh = [line for line in kernel['stdout'].splitlines() if FATAL.search(line) and line not in old]
    receipt = dict(new_fatal_count=len(fresh), approved_exact_lines_count=0,
                   future_events_exempted=False, kernel_baseline_changed=False)
    if not fresh:
        return receipt
    if reference is None:
        raise RuntimeError('New critical kernel evidence requires individual review')
    review = read_verified(root, reference)
    evidence = read_verified(root, review['evidence'])
    own = memory_cgroup(cgroup_text)
    if (review.get('schema') != 1 or review.get('policy') != 'EXACT_HISTORICAL_EXTERNAL_MEMCG_OOM_ONLY' or
            boot_id != baseline['boot_id'] or review['boot_id'] != boot_id or evidence['boot_id'] != boot_id or
            review['baseline_kernel_sha256'] != baseline_digest(baseline['kernel_baseline']) or
            evidence['own_memory_cgroup'] != str(own)):
        raise ValueError('Kernel review boot, baseline, or container identity mismatch')
    allowed = [external_oom_line(context, own) for context in evidence['contexts']]
    hashes = [digest(line.encode()) for line in allowed]
    if len(set(hashes)) != len(hashes) or hashes != review['approved_fatal_line_sha256']:
        raise ValueError('Reviewed historical OOM set changed')
    if any(line not in allowed for line in fresh):
        raise RuntimeError('New or unreviewed critical kernel evidence; stop without retry')
    receipt.update(approved_exact_lines_count=len(fresh),
                   review_manifest_sha256=reference['sha256'])
    return receipt
