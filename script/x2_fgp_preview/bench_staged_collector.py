"""Paired Orin comparison of synchronous and independent FGP result collection."""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from .gpu_engine import _sha256
from .staged_engine import StagedS1
from .verify_staged import describe_delivery, timed_replay


def run_round(args, raw, calibration, independent):
    period_ns = 1_000_000_000 // 30
    delivered = []
    caller_tid = threading.get_native_id()

    def sink(row):
        delivered.append(dict(row, observed_ns=time.monotonic_ns(), collector_tid=threading.get_native_id()))

    stage = StagedS1(args.clotho_root, args.cano_engine, on_result=sink if independent else None)
    try:
        if independent:
            stage.reset(calibration)
            origin = time.monotonic_ns()
            for i, frame in enumerate(raw):
                due = origin + i * period_ns
                time.sleep(max(0, (due-time.monotonic_ns())/1e9))
                stage.submit(frame, due)
            stage.collect(stage.frame_timeout)
            if [row["source_ns"] for row in delivered] != [origin+i*period_ns for i in range(29, len(raw))]:
                raise ValueError("Independent sink lost or misattributed a source frame")
            if any(row["collector_tid"] == caller_tid for row in delivered):
                raise ValueError("Independent sink ran in the CANO thread")
            # An earlier callback may be delivered during reset, never after it returns.
            stage.submit(raw[-1], 0)
            stage.reset(calibration)
            if delivered.pop()["source_ns"] != 0 or stage.pending is not None:
                raise ValueError("Pending input did not drain at the reset boundary")
        else:
            delivered = [dict(row, observed_ns=row["returned_ns"], collector_tid=caller_tid)
                         for row in timed_replay(stage, raw, calibration, period_ns)]
    finally:
        stage.close()
    if stage.process.exitcode != 0:
        raise RuntimeError(f"CPU worker did not exit cleanly: {stage.process.exitcode}")
    if len(delivered) != 211 or [row["sequence"] for row in delivered] != list(range(1, 212)):
        raise ValueError("Expected the full stable frame sequence")
    return delivered


def run(args):
    start = time.monotonic()
    with np.load(args.fixture, allow_pickle=False) as fixture:
        raw = fixture["raw"][:240]
        calibration = fixture["calibration"]
    if raw.shape != (240, 10, 7) or not np.isfinite(raw).all():
        raise ValueError("Expected 240 finite recorded frames")
    rounds = []
    stage_rows = []
    reference = None
    for label in ("inline", "independent", "independent", "inline"):
        rows = run_round(args, raw, calibration, label == "independent")
        outputs = {key: np.stack([row["output"][key] for row in rows])
                   for key in ("pose_mat", "root_translation_m")}
        if reference is None:
            reference = outputs
        delivery = describe_delivery([(r["source_ns"], r["observed_ns"]) for r in rows])
        delivery.update(label=label, frames=len(rows), max_abs_vs_first={
            key: float(np.max(np.abs(value - reference[key]))) for key, value in outputs.items()})
        delivery["cpu_finish_to_sink_p95_ms"] = float(np.percentile(
            [(r["observed_ns"]-r["cpu_finished_ns"])/1e6 for r in rows], 95))
        delivery["cpu_completion_interval_p95_ms"] = describe_delivery(
            [(r["source_ns"], r["cpu_finished_ns"]) for r in rows])["interval_p95_ms"]
        rounds.append(delivery)
        stage_rows.append(rows)
        print(json.dumps(delivery), flush=True)

    result = dict(status="PASSED" if all(max(r["max_abs_vs_first"].values()) < 2e-5 for r in rounds) else "FAILED",
                  rounds=rounds, wall_s=time.monotonic()-start, recorded_release_hz=30,
                  fixture_sha256=_sha256(args.fixture), hardware_commands=0)
    with args.trace.open("x") as trace:
        for index, rows in enumerate(stage_rows, start=1):
            for row in rows:
                json.dump(dict(round=index, **{k: v for k, v in row.items() if k != "output"}),
                          trace, allow_nan=False)
                trace.write("\n")
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return result["status"] == "PASSED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clotho-root", "cano-engine", "fixture", "output", "trace"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.trace.exists() or args.output.resolve() == args.trace.resolve():
        parser.error("Provide distinct new output and trace paths")
    if not run(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
