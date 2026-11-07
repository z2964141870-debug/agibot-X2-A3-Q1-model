"""Match source clips while changing only the behavior-state data distribution."""

import numpy as np

from student_causal_fit import causal_records
from student_coverage_data import balanced_indices, cells, combine


def prepare(origins, pure, candidate, jobs):
    old, sources = combine(origins, candidate)
    ids = {row["relative"]: index for index, row in sources.items()}
    matched, current = {}, {}
    if len({j["relative"] for j in jobs}) != len(jobs) or any(
            j["relative"] not in ids or j["split"] not in ("train", "validation")
            or any(j[k] != sources[ids[j["relative"]]][k] for k in ("split", "group_id")) for j in jobs):
        raise ValueError("Pure-student jobs must preserve registered source groups and splits")
    if set(pure) != {"train", "validation"}:
        raise ValueError("Only owned train/validation arrays are accepted")
    for split in ("train", "validation"):
        data = causal_records(pure[split])
        mapping = {i: ids[j["relative"]] for i, j in enumerate(jobs) if j["split"] == split}
        if set(map(int, data["clip_id"])) != set(mapping):
            raise ValueError("Pure-student source identities changed")
        current[split] = {name: value.copy() for name, value in data.items()}
        current[split]["clip_id"] = np.asarray([mapping[int(i)] for i in data["clip_id"]], dtype=np.int64)
        current[split]["origin_id"] = np.full(len(data["clip_id"]), 3, dtype=np.int64)
        selected = np.isin(old[split]["clip_id"], list(mapping.values())) & (old[split]["origin_id"] < 2)
        matched[split] = {name: value[selected] for name, value in old[split].items()}
        if (set(matched[split]["clip_id"]) != set(current[split]["clip_id"])
                or any(sources[int(i)]["split"] != split for i in current[split]["clip_id"])):
            raise ValueError("The control half must cover exactly the same source clips")
    if set(current["train"]["clip_id"]) & set(current["validation"]["clip_id"]):
        raise ValueError("Owned source identities cross train/validation")
    return old, matched, current, sources


def schedule(old, matched, pure, sources, rng, count, offset):
    if count < 2 or count % 2 or offset < 0:
        raise ValueError("Require an even batch and a nonnegative shared cursor")
    half = count // 2
    anchors = balanced_indices(cells(old, sources), rng, half, offset)
    a, b = cells(matched, sources), cells(pure, sources)
    if set(a) != set(b) or any(set(a[g]) != set(b[g]) for g in a):
        raise ValueError("Both treatment halves must have identical groups and source clips")
    before, after = [], []
    names = sorted(a)
    for slot in range(offset, offset + half):
        group = names[slot % len(names)]
        visit = slot // len(names)
        clips = sorted(a[group])
        clip = clips[visit % len(clips)]
        old_origins, new_origins = sorted(a[group][clip]), sorted(b[group][clip])
        old_origin = old_origins[(visit // len(clips)) % len(old_origins)]
        new_origin = new_origins[(visit // len(clips)) % len(new_origins)]
        before.append(int(rng.choice(a[group][clip][old_origin])))
        after.append(int(rng.choice(b[group][clip][new_origin])))
    order = rng.permutation(half)
    before, after = np.asarray(before, dtype=np.int64)[order], np.asarray(after, dtype=np.int64)[order]
    if not np.array_equal(matched["clip_id"][before], pure["clip_id"][after]):
        raise ValueError("Paired slots changed their source identity")
    return anchors, before, after


