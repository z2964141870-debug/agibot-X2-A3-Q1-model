"""Contracts and prefix-only comparisons for saved f0 failure states."""

import numpy as np

CLIPS = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023")
VARIANTS = ("coverage_f0", "original_f0", "teacher_h4", "teacher_h10")
MAX_FRAMES = 128
SELECTED_SHA = "108e649d1466d755a66da84a61a5c6f6a552b788f945deefcbf515ff327874b3"


def cleanup_confirmed(guards, attempted_phases):
    return len(guards) == attempted_phases and all(
        g.get("cleanup", {}).get("worker_reaped") is True
        and type(g["cleanup"].get("worker_return_code")) is int for g in guards)


def prefix(actual, actions, case):
    attempted = case["attempted_steps"]
    if (case.get("clip") not in CLIPS or case.get("event") != "nominal"
            or case.get("complete") is not False or type(attempted) is not int
            or attempted != case["valid_steps"] + 1
            or case.get("failure_frame") != attempted
            or actual.shape != (attempted + 1, 79) or actions.shape != (attempted, 29)
            or not np.isfinite(actual).all() or not np.isfinite(actions).all()):
        raise ValueError("Require the actual failed trace, including its last attempted call")
    count = min(MAX_FRAMES, attempted)
    return actual[:count].copy(), actions[:count].copy()


def parity(action, expected):
    a, b = np.asarray(action, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if (a.shape != (29,) or b.shape != (29,) or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.allclose(a, b, atol=2e-5, rtol=2e-5)):
        raise ValueError("Instrumented selected f0 differs from its saved same-state action")
    return float(np.max(np.abs(a-b)))


def compare(actions, present, upper, case):
    arrays = {k: np.asarray(actions[k], dtype=np.float64) for k in VARIANTS}
    count = min(MAX_FRAMES, case["attempted_steps"])
    present = np.asarray(present, dtype=bool)
    if (present.shape != (29,) or present.sum() != 22 or len(set(upper)) != len(upper)
            or any(not 0 <= j < 29 or not present[j] for j in upper)
            or any(a.shape != (count, 29) or not np.isfinite(a).all() for a in arrays.values())
            or any(np.count_nonzero(a[:, ~present]) for a in arrays.values())):
        raise ValueError("Require four aligned, finite canonical action prefixes and actual Q1 presence")
    subsets = {"present": np.flatnonzero(present), "upper": np.asarray(upper),
               "other_present": np.asarray([j for j in np.flatnonzero(present) if j not in upper])}
    intervals = {"all_prefix": (0, count), "first_25": (0, min(25, count)),
                 "last_25": (max(0, count-25), count)}
    rows = []
    for student in VARIANTS[:2]:
        for teacher in VARIANTS[2:]:
            residual = arrays[student] - arrays[teacher]
            for window, (start, end) in intervals.items():
                r = residual[start:end]
                rows.append(dict(student=student, privileged_reference=teacher, window=window,
                    first_frame=start, end_frame_exclusive=end, samples=end-start,
                    rmse_by_group={k:float(np.sqrt(np.mean(r[:, ids]**2))) for k,ids in subsets.items()},
                    per_joint_rmse=np.sqrt(np.mean(r*r, axis=0)).tolist(),
                    maximum_absolute_residual=float(np.abs(r).max())))
    jumps = {k:dict(maximum_l2=float(np.linalg.norm(np.diff(a[:, present], axis=0), axis=1).max()),
                    maximum_absolute_joint=float(np.abs(np.diff(a[:, present], axis=0)).max()))
             for k,a in arrays.items()}
    return dict(prefix_calls=count, original_attempted_steps=case["attempted_steps"],
        failure_call_included=count == case["attempted_steps"],
        original_failure_call_index=case["attempted_steps"]-1,
        residuals=rows, action_jumps=jumps, full_clip_metrics=None,
        note="Same recorded states/history and anchors; privileged future references do not establish a causal failure mechanism.")
