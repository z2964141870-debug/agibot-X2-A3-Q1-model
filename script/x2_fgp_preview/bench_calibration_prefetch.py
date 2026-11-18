"""Compare original FGP calibration placement on paced recorded IMU input."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import queue
import resource
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


RATE = 30
FRAME_NS = 1_000_000_000 / RATE
TOLERANCE = 2e-5


@dataclass(frozen=True)
class Packet:
    index: int
    raw: np.ndarray
    due: int
    released: int
    prepared: int
    packed: list | None


class TimedCalibration:
    def __init__(self, calibrator):
        self.calibrator = calibrator
        self.rows = []

    def calibrate(self, raw, acc_scale):
        before = resource.getrusage(resource.RUSAGE_THREAD)
        start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        result = self.calibrator.calibrate(raw, acc_scale)
        cpu = time.thread_time_ns() - cpu_start
        wall = time.perf_counter_ns() - start
        after = resource.getrusage(resource.RUSAGE_THREAD)
        self.rows.append((wall, cpu, after.ru_nvcsw - before.ru_nvcsw,
                          after.ru_nivcsw - before.ru_nivcsw))
        return result


class PreparedCalibration:
    packet: Packet | None = None

    def calibrate(self, raw, acc_scale):
        packet = self.packet
        if packet is None or raw is not packet.raw or acc_scale != 30 or packet.packed is None:
            raise ValueError("calibration must use the prepared packet and original scale")
        self.packet = None
        return packet.packed


def release_packets(engine, raw, calibrator, prefetch, packets, stop, cpu):
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    enqueue_times = []
    anchor = time.perf_counter_ns() + 100_000_000
    with engine.torch.inference_mode():
        for index, frame in enumerate(raw):
            due = anchor + index * 1_000_000_000 // RATE
            while (remaining := due - time.perf_counter_ns()) > 0:
                if stop.wait(remaining / 1e9):
                    return
            if stop.is_set():
                return
            released = time.perf_counter_ns()
            packed = calibrator.calibrate(frame, acc_scale=30) if prefetch else None
            packet = Packet(index, frame, due, released, time.perf_counter_ns(), packed)
            while not stop.is_set():
                try:
                    packets.put(packet, timeout=0.05)
                    enqueue_times.append(time.perf_counter_ns() - packet.prepared)
                    break
                except queue.Full:
                    continue

    return enqueue_times


def worker(args):
    spec = importlib.util.spec_from_file_location("fgp_prefetch_engine", args.engine_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(args.fixture, allow_pickle=False) as fixture:
        raw, calibration = fixture["raw"][:args.frames], fixture["calibration"]
    if (raw.shape != (args.frames, 10, 7) or calibration.shape != (120,)
            or not np.isfinite(raw).all() or not np.isfinite(calibration).all()):
        raise ValueError("invalid recorded IMU prefix or calibration")
    engine = module.S1GpuFgp(args.clotho_root, args.cano_engine)
    original = engine.calibrator_adapter
    prefetch = args.worker == "on"
    prepared = PreparedCalibration()
    calibrator = TimedCalibration(original) if args.cores else original
    packets = queue.Queue(maxsize=2)
    stop = threading.Event()
    rows, indices, pose, root = [], [], [], []
    try:
        engine.reset(calibration)
        if args.cores:
            os.sched_setaffinity(0, {args.cores[0]})
        engine.calibrator_adapter = prepared if prefetch else calibrator
        with ThreadPoolExecutor(max_workers=1) as executor:
            producer = executor.submit(release_packets, engine, raw, calibrator, prefetch, packets, stop,
                                       args.cores[1] if args.cores else None)
            try:
                for expected in range(len(raw)):
                    try:
                        packet = packets.get(timeout=4)
                    except queue.Empty:
                        if producer.done():
                            producer.result()
                        raise
                    if (packet.index != expected or packet.released < packet.due
                            or packet.prepared < packet.released):
                        raise ValueError("unexpected frame order or release time")
                    if prefetch:
                        prepared.packet = packet
                    started = time.perf_counter_ns()
                    result = engine.step(packet.raw)
                    finished = time.perf_counter_ns()
                    if prefetch and prepared.packet is not None:
                        raise ValueError("FGP did not consume the prepared calibration")
                    rows.append((packet.due, packet.released, packet.prepared, started, finished))
                    if result is not None:
                        indices.append(expected)
                        pose.append(result["pose_mat"])
                        root.append(result["root_translation_m"])
                enqueue_times = producer.result(timeout=2)
            finally:
                stop.set()
    finally:
        engine.calibrator_adapter = original
        engine.close()
    if len(indices) < 2 or not np.isfinite(pose).all() or not np.isfinite(root).all():
        raise ValueError("need at least two finite outputs")
    timing = {}
    if args.cores:
        timing["calibration"] = calibrator.rows
        timing["enqueue"] = enqueue_times
        if len(calibrator.rows) != len(raw) or len(enqueue_times) != len(raw):
            raise ValueError("missing frame timings")
    np.savez(args.worker_output, indices=indices, pose=pose, root=root, rows=rows, **timing)


def stats(ns):
    values = ns / 1e6
    return dict(n=len(ns), p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                max_ms=float(np.max(values)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("engine_source", "clotho_root", "cano_engine", "fixture"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    cli.add_argument("--worker", choices=("off", "on"), help=argparse.SUPPRESS)
    cli.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    cli.add_argument("--frames", type=int, default=96)
    cli.add_argument("--cores", nargs=2, type=int, metavar=("CONSUMER", "PRODUCER"),
                     help="Pin the two calling threads and record calibration CPU time and context switches")
    args = cli.parse_args()
    if not 60 <= args.frames <= 120:
        cli.error("--frames must be between 60 and 120")
    if args.cores and (not sys.platform.startswith("linux") or len(set(args.cores)) != 2
                       or not set(args.cores) <= os.sched_getaffinity(0)):
        cli.error("--cores requires two distinct allowed Linux CPUs")
    if args.worker:
        if args.worker_output is None:
            cli.error("--worker requires --worker-output")
        worker(args)
        return
    reports, histories = [], {"on": [], "off": []}
    diffs = dict(pose=0.0, root=0.0)
    baseline = None
    with tempfile.TemporaryDirectory(prefix="fgp-prefetch-") as temp:
        for index, flag in enumerate(("off", "on", "on", "off")):
            output = Path(temp) / f"run-{index}.npz"
            subprocess.run([sys.executable, "-B", __file__, *sys.argv[1:],
                            "--worker", flag, "--worker-output", str(output)], check=True, timeout=45)
            with np.load(output, allow_pickle=False) as data:
                indices = data["indices"]
                snapshot = {key: data[key] for key in ("indices", "pose", "root")}
                if baseline is None:
                    baseline = snapshot
                else:
                    np.testing.assert_array_equal(baseline["indices"], indices)
                    for key in diffs:
                        np.testing.assert_allclose(baseline[key], snapshot[key], atol=TOLERANCE, rtol=0)
                        diffs[key] = max(diffs[key], float(np.max(np.abs(baseline[key] - snapshot[key]))))
                due, released, ready, start, finish = data["rows"][indices].T
                intervals = np.diff(finish)
                fields = dict(source_to_output=finish - due, interval=intervals, consumer=finish - start,
                              release_late=released - due, ready_to_consumer=start - ready)
                if flag == "on":
                    fields["producer_calibration"] = ready - released
                switches = {}
                if args.cores:
                    wall, cpu, voluntary, involuntary = data["calibration"][indices].T
                    fields.update(calibration_wall=wall, calibration_cpu=cpu,
                                  calibration_wall_minus_cpu=wall - cpu,
                                  enqueue_call=data["enqueue"][indices])
                    slow = np.argsort(finish - due)[-5:][::-1]
                    switches = dict(calibration_voluntary_switches=int(voluntary.sum()),
                                    calibration_involuntary_switches=int(involuntary.sum()),
                                    slowest_outputs=[
                                        dict(index=int(indices[i]), source_ms=float((finish[i] - due[i]) / 1e6),
                                             calibration_wall_ms=float(wall[i] / 1e6),
                                             calibration_cpu_ms=float(cpu[i] / 1e6),
                                             voluntary_switches=int(voluntary[i]),
                                             involuntary_switches=int(involuntary[i]),
                                             enqueue_call_ms=float(fields["enqueue_call"][i] / 1e6),
                                             ready_to_consumer_ms=float(fields["ready_to_consumer"][i] / 1e6))
                                        for i in slow])
                histories[flag].append(fields)
                reports.append(dict(prefetch=flag, outputs=len(indices),
                                    intervals_over_budget=int(np.count_nonzero(intervals > FRAME_NS)),
                                    **switches, **{key: stats(values) for key, values in fields.items()}))
    pooled = {}
    for flag, batches in histories.items():
        pools = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
        pooled[flag] = dict(**{key: stats(values) for key, values in pools.items()},
                           intervals_over_budget=int(np.count_nonzero(pools["interval"] > FRAME_NS)))
    print(json.dumps(dict(scope="recorded IMU paced at 30 Hz, bounded FIFO, no live I/O",
                          imu_frames=4 * args.frames, cores=args.cores, runs=reports, pooled=pooled,
                          output_max_abs_difference=diffs, hardware_commands=0), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
