"""Join owned coverage data without changing original row or split identities."""

import numpy as np

from student_causal_fit import INPUT_FIELDS, causal_records
from student_input_audit import expected_flags


def catalog(plan):
    rows = plan["selected_sources"]
    if (len({r["relative"] for r in rows}) != len(rows)
            or any(r["split"] not in ("train", "validation") for r in rows)):
        raise ValueError("Coverage must contain unique owned train/validation sources")
    owners = {}
    for row in rows:
        prior = owners.setdefault(row["group_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("Coverage group crosses source splits")
    return {i:row for i, row in enumerate(rows)}


def combine(origins, plan):
    if len(origins) != 3 or any(set(v) != {"train", "validation"} for v in origins):
        raise ValueError("Use the two preserved origins and only new coverage data")
    sources = catalog(plan)
    global_ids = {r["relative"]:i for i, r in sources.items()}
    jobs = (plan["original_jobs"], plan["original_jobs"], plan["new_collection_jobs"])
    result = {}
    for split in ("train", "validation"):
        parts = []
        for origin, (values, owned) in enumerate(zip(origins, jobs)):
            data = causal_records(values[split])
            mapping = {i:global_ids[j["relative"]] for i, j in enumerate(owned) if j["split"] == split}
            if set(map(int, data["clip_id"])) != set(mapping):
                raise ValueError("Lost, added or reassigned an owned collection source")
            part = {k:v.copy() for k, v in data.items()}
            part["clip_id"] = np.asarray([mapping[int(i)] for i in data["clip_id"]], dtype=np.int64)
            part["origin_id"] = np.full(len(data["clip_id"]), origin, dtype=np.int64)
            parts.append(part)
        result[split] = {k:np.concatenate([p[k] for p in parts]) for k in (*INPUT_FIELDS, "origin_id")}
        expected = {i for i, r in sources.items() if r["split"] == split}
        if set(map(int, result[split]["clip_id"])) != expected:
            raise ValueError("Actual coverage differs from the registered selected sources")
    return result, sources


def cells(data, sources):
    groups = {}
    for clip in sorted(set(map(int, data["clip_id"]))):
        group = sources[clip]["group_id"]
        groups.setdefault(group, {})[clip] = {
            int(origin):np.flatnonzero((data["clip_id"] == clip) & (data["origin_id"] == origin))
            for origin in sorted(set(map(int, data["origin_id"][data["clip_id"] == clip]))) }
    return groups


def balanced_indices(groups, rng, count, offset):
    if not groups or count < 1 or offset < 0:
        raise ValueError("Need nonempty owned cells and a nonnegative sampling cursor")
    names, chosen = sorted(groups), []
    for slot in range(offset, offset+count):
        group = groups[names[slot % len(names)]]
        visit = slot//len(names)
        clips = sorted(group)
        origins = group[clips[visit % len(clips)]]
        keys = sorted(origins)
        origin = keys[(visit//len(clips)) % len(keys)]
        chosen.append(int(rng.choice(origins[origin])))
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def validate_records(data, split, jobs, traces):
    count = len(data["frame"])
    shapes = dict(f0=(count, 10, 226), history=(count, 930), teacher_action=(count, 29),
        actual_state=(count, 79), frame=(count,), clip_id=(count,))
    if not count or set(data) != set(shapes) or any(
            data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError("Invalid actual coverage dataset arrays")
    if any(not np.issubdtype(data[k].dtype, np.integer) for k in ("frame", "clip_id")):
        raise ValueError("Keep integer frame and source identities")
    causal_records(data)
    keys, first = set(), {}
    for i, (clip, frame) in enumerate(zip(data["clip_id"], data["frame"])):
        clip, frame = int(clip), int(frame)
        if not 0 <= clip < len(jobs) or (clip, frame) in keys:
            raise ValueError("Duplicate or unknown coverage row")
        keys.add((clip, frame))
        job, trace = jobs[clip], traces[clip]
        if (job["split"] != split or frame < 0 or frame % 2 or frame >= job["maximum_steps"]
                or frame >= len(trace["canonical_action"]) or not trace["input_ready"][frame]):
            raise ValueError("Coverage row crosses split, visibility or terminal boundary")
        if (not np.array_equal(data["actual_state"][i], trace["actual"][frame])
                or not np.array_equal(data["teacher_action"][i], trace["canonical_action"][frame])):
            raise ValueError("Coverage state or teacher label differs from the physical trace")
        flags = expected_flags(frame, job["maximum_steps"]+10, job["packet_event"])
        if not np.allclose(data["f0"][i, :, 223:], flags, atol=1e-8, rtol=1e-6):
            raise ValueError("Coverage history violates the packet/dependency clock")
    expected = {i for i, j in enumerate(jobs) if j["split"] == split}
    if set(map(int, data["clip_id"])) != expected:
        raise ValueError("One new owned source has no available training samples")
    for clip in expected:
        ready = np.flatnonzero(traces[clip]["input_ready"])
        eligible = ready[ready % 2 == 0]
        if not len(eligible) or (clip, int(eligible[0])) not in keys:
            raise ValueError("First causally available even-frame startup row was lost")
        first[str(clip)] = int(eligible[0])
    return dict(records=count, source_clips=len(expected), actual_state_and_teacher_match=True,
        dependency_clocks_match=True, first_available_even_frames=first,
        first_200ms_records=int(np.count_nonzero(data["frame"] < 10)), test_sources_read=False)
