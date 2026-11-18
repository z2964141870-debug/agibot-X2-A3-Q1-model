"""Compare isolated S1 stages against the existing S1 engine on recorded IMUs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .gpu_engine import S1GpuFgp, _sha256
from .staged_engine import StagedS1


def timed_replay(staged, raw, calibration, period_ns):
    staged.reset(calibration)
    rows = []
    source_times = []
    origin = time.monotonic_ns()
    for index, frame in enumerate(raw):
        due = origin + index * period_ns if period_ns else index
        source_times.append(due)
        while period_ns and (remaining := (due-time.monotonic_ns())/1e9) > 0:
            if staged.pending is not None and staged.pipe.poll(remaining):
                rows.append(staged.collect())
            else:
                time.sleep(max(0, (due-time.monotonic_ns())/1e9))
        row = staged.submit(frame, due)
        if row is not None:
            rows.append(row)
    if staged.pending is not None:
        rows.append(staged.collect(staged.frame_timeout))
    if [row["source_ns"] for row in rows] != source_times[29:]:
        raise ValueError("Staged replay lost or misattributed a source frame")
    return rows


def describe_delivery(samples):
    received = np.array([received for _, received in samples])
    source = np.array([source for source, _ in samples])
    intervals_ms = np.diff(received) / 1e6
    age_ms = (received-source) / 1e6
    return dict(output_hz=float((len(received)-1)*1e9/(received[-1]-received[0])),
                interval_p95_ms=float(np.percentile(intervals_ms, 95)),
                interval_max_ms=float(intervals_ms.max()),
                intervals_over_33_33ms=int((intervals_ms > 1000/30).sum()),
                source_to_result_p95_ms=float(np.percentile(age_ms, 95)))


def run(args):
    began = time.monotonic()
    length = 240 if args.pace else 93
    period_ns = 1_000_000_000 // 30 if args.pace else 0
    with np.load(args.fixture, allow_pickle=False) as fixture:
        calibration = fixture["calibration"]
        raw = fixture["raw"][:length]
    if raw.shape != (length, 10, 7) or not np.isfinite(raw).all():
        raise ValueError(f"Expected {length} finite recorded IMU frames")

    baseline = S1GpuFgp(args.clotho_root, args.cano_engine)
    try:
        baseline.reset(calibration)
        reference, delivered = [], []
        origin = time.monotonic_ns()
        for index, frame in enumerate(raw):
            due = origin + index * period_ns
            if period_ns:
                time.sleep(max(0, (due-time.monotonic_ns())/1e9))
            value = baseline.step(frame)
            reference.append(value)
            if value is not None:
                delivered.append((due, time.monotonic_ns()))
    finally:
        baseline.close()
    if any(row is not None for row in reference[:29]) or any(row is None for row in reference[29:]):
        raise ValueError("Baseline warmup/output alignment differs")

    stage = StagedS1(args.clotho_root, args.cano_engine)
    try:
        rows = timed_replay(stage, raw, calibration, period_ns)
        # Leave a reconstruction pending across a reset and prove it is discarded.
        stage.submit(raw[-1], 10_000)
        repeat = timed_replay(stage, raw, calibration, period_ns)
    finally:
        stage.close()
    if stage.process.exitcode != 0:
        raise RuntimeError(f"Staged CPU worker did not exit cleanly: {stage.process.exitcode}")
    diffs = {}
    for name in ("pose_mat", "root_translation_m"):
        expected = np.stack([row[name] for row in reference[29:]])
        actual = np.stack([row["output"][name] for row in rows])
        reset = np.stack([row["output"][name] for row in repeat])
        diffs[f"{name}_vs_baseline"] = float(np.max(np.abs(actual - expected)))
        diffs[f"{name}_after_reset"] = float(np.max(np.abs(reset - actual)))
    passed = all(value < 2e-5 for value in diffs.values())
    result = dict(status="PASSED" if passed else "FAILED", samples=len(rows),
                  fixture_sha256=_sha256(args.fixture),
                  hardware_commands=0, native_worker_exitcode=stage.process.exitcode,
                  wall_s=time.monotonic()-began, tolerance=2e-5, differences=diffs)
    if period_ns:
        result["recorded_release_hz"] = 30
        result["delivery"] = dict(monolithic=describe_delivery(delivered),
                                  staged=describe_delivery([(row["source_ns"], row["returned_ns"]) for row in rows]),
                                  staged_after_reset=describe_delivery(
                                      [(row["source_ns"], row["returned_ns"]) for row in repeat]))
        result["staged_cpu_completion"] = describe_delivery(
            [(row["source_ns"], row["cpu_finished_ns"]) for row in rows])
        result["staged_breakdown_p95_ms"] = {
            key: float(np.percentile([row[key] for row in rows], 95))
            for key in ("calibration_ms", "cano_ms", "cpu_queue_ms", "cpu_ms", "inference_ms")
        }
        result["staged_breakdown_p95_ms"]["cpu_finish_to_collect_ms"] = float(np.percentile(
            [(row["returned_ns"]-row["cpu_finished_ns"])/1e6 for row in rows], 95))
    if args.trace is not None:
        with args.trace.open("x") as trace:
            for group, entries in (("staged", rows), ("staged_after_reset", repeat)):
                for row in entries:
                    json.dump(dict(group=group, **{k: v for k, v in row.items() if k != "output"}),
                              trace, allow_nan=False)
                    trace.write("\n")
    with args.output.open("x") as dest:
        json.dump(result, dest, indent=2, allow_nan=False)
        dest.write("\n")
    print(json.dumps(result))
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clotho-root", "cano-engine", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pace", action="store_true", help="release 240 recorded frames at 30 Hz")
    parser.add_argument("--trace", type=Path, help="save per-frame timings after replay")
    args = parser.parse_args()
    if args.output.exists() or (args.trace is not None and
                                (args.trace.exists() or args.trace.resolve() == args.output.resolve())):
        parser.error("Provide distinct new output and trace paths")
    if not run(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
