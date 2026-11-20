"""Evaluate two scheduled pushes on one continuous trajectory."""

import numpy as np


def score_repeated(trace, poses, body_floor, kicks, reanchors):
    ticks = np.arange(len(trace))
    gate = trace[:, 5]
    releases = []
    for anchor in reanchors:
        zero = np.flatnonzero((ticks >= anchor) & (gate == 0))
        if len(zero):
            releases.append(int(zero[0]))
    # Each imposed push may start one recovery cycle. A third cycle is a refall.
    engage = np.flatnonzero((gate == 1) & np.r_[True, gate[:-1] != 1])
    phases = []
    for i, start in enumerate(kicks):
        stop = kicks[i + 1] if i + 1 < len(kicks) else len(trace)
        down = np.any(
            body_floor[start + 1 : stop]
            & ((trace[start : stop - 1, 0] < 0.22) | (trace[start : stop - 1, 1] < 0.65))
        )
        handed = [r for r in releases if start <= r < stop]
        # Count every entry to full recovery, including entries that never release.
        takeovers = int(np.sum((engage >= start) & (engage < stop)))
        phases.append(
            dict(
                start=int(start),
                end=int(stop),
                down=bool(down),
                takeovers=takeovers,
                releases=handed,
                clean=bool(
                    len(handed) == takeovers and takeovers <= 1 and (not down or takeovers == 1)
                ),
            )
        )
    tail = trace[-150:]
    stable = len(tail) == 150 and np.all(
        (tail[:, 0] >= 0.35) & (tail[:, 1] > 0.85) & (tail[:, 5] == 0)
    )
    distance = float(np.linalg.norm(poses[-1, :2] - poses[-len(tail), :2]))
    width = float(tail[:, 7].mean())
    oob = int(trace[:, 3].sum())
    return dict(
        phases=phases,
        stable=bool(stable),
        final_body_contacts=int(body_floor[-150:].sum()),
        foot_width_cm=width,
        walk_disp_m=distance,
        pd_oob=oob,
        success=bool(
            len(phases) == 2
            and phases[0]["down"]
            and phases[0]["releases"]
            and all(p["clean"] for p in phases)
            and stable
            and not body_floor[-150:].any()
            and width <= 22
            and distance > 0.25
            and oob == 0
        ),
    )
