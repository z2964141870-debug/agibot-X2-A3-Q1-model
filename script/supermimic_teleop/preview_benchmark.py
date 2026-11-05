"""Offline preview-error baselines on saved reference angles; no policy or physics.

At origin t, a buffer b exposes samples only through t+b. Known future slots are
copied and the rest of the 10-slot target is forecast from that visible prefix.
All settings use identical origins. Parameters are fixed, with no fitting across
clips. These eight retargeted upper-body channels are not live human sensor data.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import time


DT = 0.02
HORIZON = 10
METHODS = ("hold", "cv_difference", "cv_ls5", "ca_ls7", "damped_cv_ls5")
CLIPS = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023",
         "walk_forward_stop_002__A024")


def solve3(matrix, rhs):
    rows = [list(map(float, row)) + [float(y)] for row, y in zip(matrix, rhs)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-12:
            raise ValueError("Singular polynomial fit")
        rows[col] = [value / scale for value in rows[col]]
        for r in range(3):
            if r != col:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


def kinematics(history, method):
    if len(history) < 7 or method not in METHODS:
        raise ValueError("Seven visible frames and a known method are required")
    width = len(history[-1])
    velocity, acceleration = [0.0] * width, [0.0] * width
    if method == "hold":
        return velocity, acceleration
    if method == "cv_difference":
        return [(a-b) / DT for a, b in zip(history[-1], history[-2])], acceleration
    if method in ("cv_ls5", "damped_cv_ls5"):
        for j in range(width):
            velocity[j] = sum((i-2) * row[j] for i, row in enumerate(history[-5:])) / (10 * DT)
        return velocity, acceleration
    xs = list(range(-6, 1))
    moments = [sum(x ** p for x in xs) for p in range(5)]
    matrix = [[moments[r+c] for c in range(3)] for r in range(3)]
    for j in range(width):
        rhs = [sum((x ** p) * row[j] for x, row in zip(xs, history[-7:])) for p in range(3)]
        _, first, second = solve3(matrix, rhs)
        velocity[j], acceleration[j] = first / DT, 2 * second / DT ** 2
    return velocity, acceleration


def preview(visible, origin, method, damping_s=0.12):
    """Only a causal prefix is accepted; callers must not pass hidden samples."""
    latest = len(visible) - 1
    if not 0 <= latest-origin <= HORIZON or origin < 6 or damping_s <= 0:
        raise ValueError("Invalid visible prefix or buffer")
    velocity, acceleration = kinematics(visible, method)
    out = []
    for offset in range(1, HORIZON + 1):
        frame = origin + offset
        if frame <= latest:
            out.append(list(visible[frame]))
            continue
        h = (frame-latest) * DT
        travel = damping_s * (1-math.exp(-h/damping_s)) if method == "damped_cv_ls5" else h
        out.append([q + v * travel + .5 * a * h*h
                    for q, v, a in zip(visible[-1], velocity, acceleration)])
    return out


def load_reference(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    names = [name for name in reader.fieldnames if name.startswith("reference_")]
    if len(names) != 8:
        raise ValueError("Expected the eight saved reference shoulder/elbow channels")
    values = []
    for i, row in enumerate(reader):
        if int(row["frame"]) != i or not math.isclose(float(row["time_s"]), i*DT, abs_tol=1e-7):
            raise ValueError("Expected contiguous 50Hz reference frames")
        current = [float(row[name]) for name in names]
        if not all(math.isfinite(x) for x in current):
            raise ValueError("Nonfinite saved reference")
        values.append(current)
    if len(values) <= HORIZON + 7:
        raise ValueError("Reference clip is too short")
    return values, dict(path=str(path.resolve()), bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(), frames=len(values), channels=names)


def evaluate(clean, observed, buffer_frames, method):
    errors, tail_errors, timings = [], [], []
    per_offset = [[] for _ in range(HORIZON)]
    for origin in range(6, len(clean)-HORIZON):
        visible = observed[:origin + buffer_frames + 1]
        started = time.perf_counter_ns()
        predicted = preview(visible, origin, method)
        timings.append((time.perf_counter_ns()-started) / 1e6)
        for i, row in enumerate(predicted):
            for predicted_q, target in zip(row, clean[origin+i+1]):
                error = predicted_q-target
                errors.append(error)
                per_offset[i].append(error)
                if i+1 > buffer_frames:
                    tail_errors.append(error)
    def rmse(rows):
        return math.sqrt(sum(x*x for x in rows) / len(rows)) if rows else None
    absolute = sorted(abs(x) for x in errors)
    ordered_times = sorted(timings)
    return dict(origins=len(timings), scalar_errors=len(errors),
                horizon_rmse_rad=rmse(errors), forecast_only_rmse_rad=rmse(tail_errors),
                far_200ms_rmse_rad=rmse(per_offset[-1]),
                per_offset_rmse_rad=[rmse(x) for x in per_offset],
                abs_error_p95_rad=absolute[math.ceil(.95 * len(absolute))-1],
                abs_error_max_rad=absolute[-1],
                window_compute_median_ms=statistics.median(timings),
                window_compute_p95_ms=ordered_times[math.ceil(.95*len(ordered_times))-1])


def run(root, seed):
    rows, inputs = [], []
    for clip_index, clip in enumerate(CLIPS):
        clean, identity = load_reference(root / (clip+"_s0") / "upper_timing.csv")
        inputs.append(identity)
        for noise_std in (0.0, 0.01):
            rng = random.Random(seed + clip_index)
            observed = [[x+rng.gauss(0, noise_std) for x in row] for row in clean]
            for buffer_frames in (0, 4, 5, 10):
                methods = ("hold",) if buffer_frames == 10 else METHODS
                for method in methods:
                    rows.append(dict(clip=clip, noise_std_rad=noise_std,
                                     buffer_frames=buffer_frames, buffer_ms=round(buffer_frames*DT*1000),
                                     method=method if buffer_frames != 10 else "measured_h10",
                                     **evaluate(clean, observed, buffer_frames, method)))
    summary = []
    for noise in (0.0, 0.01):
        for buffer_frames in (0, 4, 5, 10):
            methods = ("measured_h10",) if buffer_frames == 10 else METHODS
            for method in methods:
                group = [x for x in rows if x["noise_std_rad"] == noise
                         and x["buffer_frames"] == buffer_frames and x["method"] == method]
                count = sum(x["scalar_errors"] for x in group)
                pooled = math.sqrt(sum(x["horizon_rmse_rad"]**2 * x["scalar_errors"] for x in group) / count)
                far = math.sqrt(sum(x["far_200ms_rmse_rad"]**2*x["origins"] for x in group)
                                / sum(x["origins"] for x in group))
                summary.append(dict(noise_std_rad=noise, buffer_frames=buffer_frames,
                                    method=method, frame_weighted_rmse_rad=pooled,
                                    far_200ms_rmse_rad=far,
                                    macro_clip_rmse_rad=statistics.mean(x["horizon_rmse_rad"] for x in group)))
    for item in inputs:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Saved input changed")
    return dict(stage="OFFLINE_REFERENCE_ONLY_NOT_ROBOT_TELEOP_VALIDATION",
                checked_utc=datetime.now(timezone.utc).isoformat(), seed=seed,
                dt_s=DT, target_offsets_ms=list(range(20, 201, 20)),
                damping_s=0.12, fitted_predictors=False, inputs_unchanged=True,
                inputs=inputs, cases=rows, summary=summary,
                limits=["Three existing retargeted upper-body reference trajectories only; eight scalar angle channels.",
                        "No human SE(3), root, contact, policy, simulation or real-robot evaluation.",
                        "No model selection or tuning on these results; noise is synthetic iid Gaussian.",
                        "Origin t and buffer b expose through t+b only; no samples after that enter prediction.",
                        "Fixed buffer time excludes capture, transport, filter, policy and actuator latency.",
                        "Compute timings are local Python operations on eight scalars, not GPU policy latency."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new result file")
    result = run(args.reference_root, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(dict(output=str(args.output.resolve()), cases=len(result["cases"]),
                          inputs_unchanged=True, physics_steps=0, optimizer_updates=0)))


if __name__ == "__main__":
    main()
