"""Source-owned four-label windows from the completed saved-sequence audit."""

import numpy as np

from student_onpolicy_pair_data import prepare


def sequence_data(origins, pure, candidate, jobs, summary):
    old, matched, current, sources = prepare(origins, pure, candidate, jobs)
    if summary.get('status') != 'SAVED_SEQUENCE_AUDIT_PASSED' or summary.get('sequence_width') != 4:
        raise ValueError('Require the completed four-label sequence audit')
    windows = {'old': {}, 'pure': {}}
    for split in ('train', 'validation'):
        joined, pure_rows, offset = [], [], 0
        for origin, raw in enumerate([o[split] for o in origins]+[pure[split]]):
            frame, clip = raw['frame'], raw['clip_id']
            if (not np.issubdtype(frame.dtype, np.integer) or len(frame) != len(clip)
                    or len(set(zip(map(int, clip), map(int, frame)))) != len(frame)):
                raise ValueError('Keep unique original source/frame identities')
            lookup = {(int(c), int(f)): i for i, (c, f) in enumerate(zip(clip, frame))}
            rows = [r for r in summary['records'] if r['origin'] == origin and r['split'] == split]
            if sum(r['records'] for r in rows) != len(frame):
                raise ValueError('Sequence summary no longer covers its source arrays')
            destination, mapped = (joined, old[split]) if origin < 3 else (pure_rows, current[split])
            for row in rows:
                for start in row['four_label_starts']:
                    local = np.asarray([lookup[(row['clip_id'], start+2*i)] for i in range(4)], dtype=np.int64)
                    if (not np.array_equal(frame[local], start+np.arange(4)*2)
                            or row['failure_frame'] is not None and int(frame[local[-1]])+1 >= row['failure_frame']
                            or 'terminal_after_action' in raw and np.any(raw['terminal_after_action'][local])):
                        raise ValueError('Sequence crosses actual time or a terminal action')
                    indices = local+(offset if origin < 3 else 0)
                    ids = mapped['clip_id'][indices]
                    if len(set(map(int, ids))) != 1 or sources[int(ids[0])]['relative'] != row['relative']:
                        raise ValueError('Sequence crossed a source mapping or split')
                    destination.append(indices)
            if origin < 3:
                offset += len(frame)
        windows['old'][split] = np.asarray(joined, dtype=np.int64).reshape(-1, 4)
        windows['pure'][split] = np.asarray(pure_rows, dtype=np.int64).reshape(-1, 4)
        if not len(joined) or not len(pure_rows):
            raise ValueError('A registered sequence distribution is empty')
    return old, matched, current, sources, windows


def sample_windows(data, windows, sources, rng, count, offset):
    cells = {}
    for index, row in enumerate(windows):
        clip = int(data['clip_id'][row[0]])
        cells.setdefault(sources[clip]['group_id'], {}).setdefault(clip, []).append(index)
    chosen, groups = [], sorted(cells)
    for slot in range(offset, offset+count):
        group = cells[groups[slot % len(groups)]]
        visit = slot//len(groups)
        clip = sorted(group)[visit % len(group)]
        chosen.append(int(rng.choice(group[clip])))
    rng.shuffle(chosen)
    return windows[np.asarray(chosen, dtype=np.int64)]
