"""Measure optional FGP timing overhead on a fixed IMU prefix, without live I/O."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def stats(values):
    return dict(n=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(max(values)))


def worker(args):
    spec = importlib.util.spec_from_file_location("fgp_probe_engine", args.engine_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        raw, calibration = fixture["raw"][:args.frames], fixture["calibration"]
    if raw.shape != (args.frames, 10, 7) or calibration.shape != (120,) or not np.isfinite(raw).all():
        raise ValueError("expected fixed recorded IMU prefix and an existing 120-value calibration")
    split_cano = args.compare_cano_copy or args.compare_cano_pinned_output
    enabled = args.worker == "on" or split_cano or args.compare_calibration
    recorder = None
    if split_cano:
        from script.x2_joint_opt.profile import Recorder, timed_graph_type
        recorder = Recorder(limit=5000)
        module.CudaGraphSession = timed_graph_type(module.CudaGraphSession, recorder)
    engine = module.S1GpuFgp(args.clotho_root, args.cano_engine, profile_stages=enabled,
                             record_cano_io=not (args.compare_cano_copy and args.worker == "on"),
                             pin_cano_output=args.compare_cano_pinned_output and args.worker == "on",
                             numpy_calibration=args.compare_calibration and args.worker == "on")
    indices, pose, root, times, stages = [], [], [], [], {}
    try:
        engine.reset(calibration)
        for index, values in enumerate(raw):
            span_start = len(recorder.rows) if recorder else 0
            t0 = time.perf_counter_ns()
            result = engine.step(values)
            elapsed = (time.perf_counter_ns() - t0) / 1e6
            if result is None:
                continue
            indices.append(index)
            pose.append(result["pose_mat"])
            root.append(result["root_translation_m"])
            times.append(elapsed)
            if enabled:
                for key, value in engine.last_step_profile.items():
                    if key.endswith("_ms") and value is not None:
                        stages.setdefault(key, []).append(value)
            if recorder is not None:
                span = recorder.rows[span_start:]
                calls = [row for row in span if row["stage"] == "cano_session"]
                if not calls:
                    raise ValueError("FGP output has no corresponding timed CANO call")
                for name in ("h2d_stream_span_ms", "graph_stream_span_ms", "d2h_stream_span_ms"):
                    stages.setdefault(name, []).append(sum(row[name] for row in calls))
                groups = (("host_cuda_sync_ms", "cuda_sync_host", None),
                          ("host_h2d_api_ms", "cuda_memcpy_host", 1),
                          ("host_d2h_api_ms", "cuda_memcpy_host", 2))
                for name, kind, direction in groups:
                    stages.setdefault(name, []).append(sum(
                        row["wall_ms"] for row in span if row["stage"] == kind
                        and (direction is None or row.get("direction") == direction)))
    finally:
        engine.close()
    if not indices:
        raise ValueError("FGP prefix produced no poses")
    np.savez(args.worker_output, indices=indices, pose=pose, root=root, times=times,
             **{f"stage_{key}": value for key, value in stages.items()})


def main():
    cli = argparse.ArgumentParser(__doc__)
    for name in ("engine_source", "clotho_root", "cano_engine", "fixture"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    cli.add_argument("--worker", choices=("off", "on"), help=argparse.SUPPRESS)
    cli.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    cli.add_argument("--frames", type=int, default=120)
    comparison = cli.add_mutually_exclusive_group()
    comparison.add_argument("--compare-cano-copy", action="store_true",
                            help="Time CANO stream stages in both arms; on skips only diagnostic input snapshots")
    comparison.add_argument("--compare-cano-pinned-output", action="store_true",
                            help="Time CANO stream stages in both arms; on pins only the CUDA output destination")
    comparison.add_argument("--compare-calibration", action="store_true",
                            help="Time both arms; on uses the candidate NumPy post-calibration math")
    args = cli.parse_args()
    if not 60 <= args.frames <= 120:
        cli.error("--frames must be between 60 and 120")
    if args.worker:
        worker(args)
        return
    histories = {"on": [], "off": []}
    runs = []
    snapshots = []
    with tempfile.TemporaryDirectory(prefix="fgp-profile-") as temp:
        for index, flag in enumerate(("off", "on", "on", "off")):
            output = Path(temp) / f"run-{index}.npz"
            subprocess.run([sys.executable, "-B", __file__, *sys.argv[1:],
                            "--worker", flag, "--worker-output", str(output)], check=True, timeout=45)
            with np.load(output, allow_pickle=False) as data:
                snapshots.append({key: data[key] for key in ("indices", "pose", "root")})
                histories[flag].extend(data["times"])
                stages = {key.removeprefix("stage_"): stats(data[key])
                          for key in data.files if key.startswith("stage_")}
                runs.append(dict(profile=flag, total=stats(data["times"]), stages=stages))
    for capture in snapshots[1:]:
        np.testing.assert_array_equal(snapshots[0]["indices"], capture["indices"])
    comparisons = []
    for lhs, rhs in ((0, 3), (1, 2), (0, 1), (3, 2)):
        comparisons.append(dict(runs=[lhs, rhs], **{
            key + "_max_abs": float(np.max(np.abs(snapshots[lhs][key] - snapshots[rhs][key])))
            for key in ("pose", "root")
        }))
    treatment = "enable stage counters"
    if args.compare_cano_copy:
        treatment = "skip unconsumed CANO input snapshots"
    if args.compare_cano_pinned_output:
        treatment = "pin CANO output destination"
    if args.compare_calibration:
        treatment = "NumPy per-frame IMU calibration math"
    print(json.dumps(dict(imu_frames=4 * args.frames, order="off,on,on,off (fresh processes)", runs=runs,
        output_max_differences=comparisons,
        off=stats(histories["off"]), on=stats(histories["on"]),
        treatment=treatment,
        scope="recorded sequential processing; no BLE, transport or log serialization",
        hardware_commands=0), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
