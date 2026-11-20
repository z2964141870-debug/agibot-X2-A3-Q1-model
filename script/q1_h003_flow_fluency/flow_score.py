"""Score continuous five-second windows aligned to the first completed handback."""

import numpy as np
from repeated import score_repeated


def measure(trace, poses, loads, body, kicks, anchors, repeat):
    phases = score_repeated(trace, poses, body, kicks, anchors)["phases"]
    gate = trace[:, 5]
    starts = np.flatnonzero((gate > 0) & np.r_[True, gate[:-1] == 0])
    fallen = any(p["down"] for p in phases)
    phase = phases[-1]
    release = next(iter(phase["releases"]), None)
    if repeat and release is None and not phase["down"] and len(phases) == 2:
        start = phase["start"] + 1
    else:
        start = release

    def assess(n):
        if start is None or start + n > len(trace):
            return False
        f = trace[start : start + n]
        return bool(
            np.all((f[:, 0] >= 0.35) & (f[:, 1] > 0.85) & (f[:, 5] == 0))
            and not body[start + 1 : start + n].any()
            and f[:, 7].mean() <= 22
            and np.linalg.norm(poses[start + n - 1, :2] - poses[start, :2]) > 0.25
            and not trace[: start + n, 3].any()
        )

    clean = all(p["clean"] for p in phases)
    refall = any(any(s > r and s < p["end"] for s in starts) for p in phases for r in p["releases"])
    ok = fallen and clean and (not repeat or len(phases) == 2) and not refall
    avoided = (
        not fallen
        and not len(starts)
        and np.all(trace[-250:, 1] > 0.85)
        and np.all(trace[-250:, 0] >= 0.35)
    )
    result = dict(
        down=fallen,
        avoided=bool(avoided),
        pass5=bool(ok and assess(250)),
        phase3=bool(ok and assess(150)),
        refell=bool(refall),
        release=release,
        phases5=phases,
        release_s=None if release is None else (release - phase["start"]) * 0.02,
    )
    if release is None:
        return result
    v = poses[release : release + 250]
    w, x, y, z = v[:, 3:7].T
    roll = np.abs(np.rad2deg(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))
    for label, a, b in [("first", 0, 50), ("three", 0, 150), ("late", 150, 250)]:
        if len(v) >= b:
            result[label + "_roll"] = float(roll[a:b].max())
            result[label + "_speed"] = float(
                np.linalg.norm(np.diff(v[a:b, :2], axis=0), axis=1).mean() * 50
            )
    side = loads[release : release + 250]
    for i in range(len(side) - 14):
        f = side[i : i + 4]
        if any(np.all((f[:, s] > 20) & (f[:, 1 - s] < 5)) for s in (0, 1)):
            result["support_roll"] = float(roll[i : i + 15].max())
            result["support_tick"] = i
            break
    return result
