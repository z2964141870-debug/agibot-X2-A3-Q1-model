"""Validate pure-student behavior and separately privileged recovery labels."""

import copy

import numpy as np

from student_input_audit import expected_flags


def collection_jobs(candidate):
    jobs = []
    for split, count in (("train", 6), ("validation", 3)):
        selected = [j for j in candidate["original_jobs"] if j["split"] == split][:count]
        if len(selected) != count:
            raise ValueError("Require the registered original six/three source groups")
        for original in selected:
            job = copy.deepcopy(original)
            job["maximum_steps"] = min(192, job["maximum_steps"])
            jobs.append(job)
    if len({j["group_id"] for j in jobs}) != 9 or len({j["clip"] for j in jobs}) != 9:
        raise ValueError("Keep source groups disjoint and do not select by outcomes")
    if any(j["clip"].endswith(("A022", "A023", "A024")) for j in jobs):
        raise ValueError("Do not collect on the exploratory comparison clips")
    return jobs


def behavior(student, teacher):
    if (student.shape != (1, 29) or teacher.shape != student.shape
            or not np.isfinite(student).all() or not np.isfinite(teacher).all()):
        raise ValueError("Require finite canonical student and shadow teacher actions")
    return student.copy()


def validate_records(data, split, jobs, traces):
    count = len(data["frame"])
    shapes = dict(f0=(count, 10, 226), history=(count, 930), teacher_action=(count, 29),
        student_action=(count, 29), actual_state=(count, 79), frame=(count,), clip_id=(count,),
        teacher_dependency_ns=(count,))
    if (not count or count > 850 or set(data) != set(shapes)
            or any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items())
            or any(not np.issubdtype(data[k].dtype, np.integer)
                   for k in ("frame", "clip_id", "teacher_dependency_ns"))):
        raise ValueError("Invalid bounded on-policy label arrays")
    keys, next_states = set(), []
    for index, (clip, frame) in enumerate(zip(data["clip_id"], data["frame"])):
        clip, frame = int(clip), int(frame)
        if not 0 <= clip < len(jobs) or (clip, frame) in keys:
            raise ValueError("Unknown or duplicate source/step")
        keys.add((clip, frame))
        job, trace = jobs[clip], traces[clip]
        if (job["split"] != split or not 0 <= frame < job["maximum_steps"] or frame % 2
                or frame >= len(trace["canonical_action"]) or not trace["input_ready"][frame]):
            raise ValueError("A label crosses split, availability or physical terminal boundary")
        if (not np.array_equal(data["actual_state"][index], trace["actual"][frame])
                or not np.array_equal(data["student_action"][index], trace["canonical_action"][frame])
                or not np.array_equal(data["teacher_action"][index], trace["teacher_action"][frame])):
            raise ValueError("Saved states, pure student actions or shadow labels differ")
        if data["teacher_dependency_ns"][index] <= frame*20_000_000:
            raise ValueError("Teacher-label privilege must remain explicit")
        flags = expected_flags(frame, int(trace["reference_frames"]), job["packet_event"])
        if not np.allclose(data["f0"][index, :, 223:], flags, atol=1e-8, rtol=1e-6):
            raise ValueError("Student feature visibility differs from the original clock")
        next_states.append(trace["actual"][frame+1])
    expected = {i for i, j in enumerate(jobs) if j["split"] == split}
    if set(map(int, data["clip_id"])) != expected:
        raise ValueError("A selected source produced no valid labels")
    for clip in expected:
        eligible = np.flatnonzero(traces[clip]["input_ready"])
        eligible = eligible[eligible % 2 == 0]
        if not len(eligible) or (clip, int(eligible[0])) not in keys:
            raise ValueError("Keep the first available even-frame sample")
    return np.asarray(next_states), dict(records=count, source_clips=len(expected),
        state_action_label_trace_match=True, executed_teacher_fraction=0.,
        student_visibility_verified=True, first_available_samples_retained=True,
        test_sources_read=False)
