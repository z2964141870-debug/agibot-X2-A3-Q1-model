"""Compare ORT thread counts while recorded IMU passes through FGP and loopback."""

import argparse
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue
import re
import subprocess
from threading import Thread
import time
from types import SimpleNamespace

import numpy as np
import zmq

from . import __path__ as package_paths
from .artifacts import activate, sha
from .bench_human_ingress import CLOCK, SharedHumanBuffer
from .bench_zmq_ingress import fixture_runtime, request_frames, serve, socket
from .core import Calibration, DT_NS, HumanBuffer, SmplAdapter
from .replay_window_pacing import distribution
from .runtime import synthetic_state


FRAMES = 180
FGP_WARMUP = 29
STATE_TICKS = 450


def read_when_written(path, process=None, timeout_s=15):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"FGP process stopped before {path.name}: {process.returncode}")
        time.sleep(0.025)
    raise TimeoutError(f"wait for {path.name}")


def publish_json(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def fgp_main(args):
    from script.x2_fgp_preview.gpu_engine import S1GpuFgp

    with np.load(args.imu, allow_pickle=False) as packet:
        raw, calibration = packet["raw"][:args.frames], packet["calibration"]
    if raw.shape != (args.frames, 10, 7) or calibration.shape != (120,):
        raise ValueError("recorded IMU layout changed")
    engine = S1GpuFgp(args.clotho_root, args.cano_engine)
    try:
        engine.reset(calibration)
        cal = Calibration(**json.loads(args.calibration.read_text()))
        adapter = SmplAdapter(args.policy_root / "upstream/deploy/vr_teleop/human_skeleton.json", cal)
        publish_json(args.output / "worker_ready.json", {"frames": len(raw)})
        start = read_when_written(args.output / "start.json")
        pending = deque()
        captured, poses, roots = [], [], []
        max_inflight = 0

        def forward(frame, due, begin, end, pending_start, pose):
            sending = time.monotonic_ns()
            packet = adapter.packet_from_matrices(
                pose["pose_mat"], pose["root_translation_m"], sequence=frame,
                capture_ns=due, capture_clock=CLOCK, arrival_ns=end, arrival_clock=CLOCK)
            sent = time.monotonic_ns()
            sender.send_json(packet)
            ack = sender.recv_json()["result"]
            confirmed = time.monotonic_ns()
            if ack["accepted_sequence"] != frame:
                raise ValueError("unexpected HumanFrame acknowledgement")
            return dict(frame=frame, input_ns=due, credit_wait_ns=begin-pending_start, begin_ns=begin,
                        inferred_ns=end, sending_ns=sending, sent_ns=sent,
                        accepted_ns=ack["accepted_ns"], confirmed_ns=confirmed)

        with zmq.Context() as context, ThreadPoolExecutor(max_workers=1) as pool:
            sender = pool.submit(socket, context, zmq.REQ).result(timeout=3)
            pool.submit(sender.connect, start["endpoint"]).result(timeout=3)
            try:
                for frame, values in enumerate(raw):
                    due = start["anchor_ns"] + frame * 1_000_000_000 // 30
                    while (remaining := due - time.monotonic_ns()) > 0:
                        time.sleep(remaining / 1e9)
                    pending_start = time.monotonic_ns()
                    while pending and (pending[0].done() or len(pending) == args.inflight):
                        captured.append(pending.popleft().result(timeout=3))
                    max_inflight = max(max_inflight, len(pending) + 1)
                    begin = time.monotonic_ns()
                    pose = engine.step(values)
                    end = time.monotonic_ns()
                    if pose is None:
                        continue
                    poses.append(pose["pose_mat"])
                    roots.append(pose["root_translation_m"])
                    pending.append(pool.submit(forward, frame, due, begin, end, pending_start, pose))
                while pending:
                    captured.append(pending.popleft().result(timeout=3))
            finally:
                pool.submit(sender.close).result(timeout=3)
        if [row["frame"] for row in captured] != list(range(FGP_WARMUP, args.frames)):
            raise ValueError("FGP changed output frame coverage")
        np.savez(args.output / "fgp_outputs.npz", pose=poses, root=roots)
        publish_json(args.output / "fgp_result.json", dict(frames=captured, peak_inflight=max_inflight))
    finally:
        engine.close()


@contextmanager
def sample_system(output, enabled):
    samples = []
    if not enabled:
        yield samples
        return
    with subprocess.Popen(["tegrastats", "--interval", "500"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as monitor:
        def collect():
            for line in monitor.stdout:
                samples.append((time.monotonic_ns(), line.strip()))

        reader = Thread(target=collect, name="tegrastats-reader")
        reader.start()
        try:
            yield samples
        finally:
            if monitor.poll() is None:
                monitor.terminate()
            try:
                monitor.wait(timeout=3)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            reader.join(timeout=3)
            if reader.is_alive():
                raise RuntimeError("telemetry reader did not exit")
            publish_json(output / "tegrastats.json", samples)


def describe_system(samples, start, end):
    window = [line for stamp, line in samples if start <= stamp <= end]
    gpu, cpu, memory, temperatures = [], [], [], []
    for line in window:
        activity = re.search(r"GR3D_FREQ (\d+)%", line)
        processors = re.search(r"CPU \[([^\]]+)\]", line)
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        gpu_temp = re.search(r"gpu@([\d.]+)C", line)
        if not (activity and processors and ram and gpu_temp):
            raise ValueError(f"unknown tegrastats format: {line}")
        gpu.append(int(activity[1]))
        cpu.append([0 if core == "off" else int(core.split("%", 1)[0])
                    for core in processors[1].split(",")])
        memory.append(int(ram[1]))
        temperatures.append(float(gpu_temp[1]))
    if len(window) < 5:
        raise ValueError("insufficient resource samples inside FGP output span")
    return dict(samples=len(window), cpu_per_core_median_percent=np.median(cpu, axis=0).tolist(),
                cpu_per_core_max_percent=np.max(cpu, axis=0).tolist(),
                gpu_engine_median_percent=float(np.median(gpu)), gpu_engine_max_percent=max(gpu),
                ram_usage_max_mb=max(memory), gpu_temp_max_c=max(temperatures))



def runtime_affinity(root, threads, mask):
    previous = os.sched_getaffinity(0)
    existing = set(Path("/proc/self/task").iterdir())
    try:
        if mask is not None:
            os.sched_setaffinity(0, mask)
        runtime = fixture_runtime(root, threads)
        created = set(Path("/proc/self/task").iterdir()) - existing
        bindings = {int(tid.name): sorted(os.sched_getaffinity(int(tid.name))) for tid in created}
        if not bindings or (mask is not None and any(value != sorted(mask) for value in bindings.values())):
            raise ValueError(f"ORT worker binding failed: {bindings}")
        return runtime, bindings
    finally:
        os.sched_setaffinity(0, previous)


def case(args, threads, output, policy_mask=None):
    output.mkdir()
    bindings = None
    if threads == 0:
        skeleton = activate(args.policy_root) / "upstream/deploy/vr_teleop/human_skeleton.json"
        runtime = SimpleNamespace(buffer=HumanBuffer(sha(skeleton), max_age_ns=120_000_000,
                                                     max_interp_gap_ns=100_000_000))
    elif args.affinity_compare:
        runtime, bindings = runtime_affinity(args.policy_root, threads, policy_mask)
    else:
        runtime = fixture_runtime(args.policy_root, threads)
    runtime.buffer = SharedHumanBuffer(runtime.buffer)
    options = ("imu", "policy_root", "clotho_root", "cano_engine", "calibration")
    worker = [str(args.fgp_python), "-B", "-c",
              "import json,runpy,sys,script.x2_morph_human as m; "
              "m.__path__[:]=json.loads(sys.argv.pop(1)); sys.path.append(sys.argv.pop(1)); "
              "runpy.run_module('script.x2_morph_human.bench_fgp_ort_joint',run_name='__main__')",
              json.dumps(list(package_paths)), str(Path(zmq.__file__).resolve().parents[1]),
              "--worker", "--output", str(output), "--frames", str(args.frames),
              "--inflight", str(args.inflight)]
    for name in options:
        worker.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    env = None
    if args.staged_worker is not None:
        env = dict(os.environ)
        env["X2_STAGED_PROBE_DEADLINE"] = str(time.time() + 95)
        env["PYTHONPATH"] = str(args.staged_worker) + os.pathsep + env.get("PYTHONPATH", "")
        worker = [str(args.fgp_python), "-B", "-m", "script.x2_fgp_staged_probe.worker", "fgp",
                  "--endpoint", "local_benchmark_pending", "--output", str(output),
                  "--source-root", str(args.clotho_root), "--build-root", str(args.cano_engine.parent),
                  "--fixture", str(args.imu)]
    state_ticks = (STATE_TICKS if args.frames == FRAMES else 650) if threads else 0
    sample = args.resource_compare or args.affinity_compare
    service_cpus = None

    def run_service(*values):
        nonlocal service_cpus
        if policy_mask is not None:
            os.sched_setaffinity(0, policy_mask)
        if args.affinity_compare:
            service_cpus = sorted(os.sched_getaffinity(0))
            if policy_mask is not None and service_cpus != sorted(policy_mask):
                raise ValueError("state service thread binding failed")
        return serve(*values)

    with (output / "fgp.log").open("x") as log, sample_system(output, sample) as samples:
        process = subprocess.Popen(worker, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            read_when_written(output / "worker_ready.json", process, 40 if args.staged_worker else 15)
            with zmq.Context() as context, ThreadPoolExecutor(max_workers=1) as worker_pool:
                endpoints = Queue()
                guard = worker_pool.submit(run_service, context, endpoints, runtime,
                                           state_ticks + args.frames - FGP_WARMUP, None, False)
                endpoint = endpoints.get(timeout=2)
                anchor = time.monotonic_ns() + 200_000_000
                states = [(tick * DT_NS, json.dumps(synthetic_state(runtime, tick, 0)).encode())
                          for tick in range(state_ticks)]
                publish_json(output / "start.json", dict(endpoint=endpoint, anchor_ns=anchor))
                replies = request_frames(context, endpoint, states, anchor)
                guard.result(timeout=18 if not threads else 3)
            if process.wait(timeout=3) != 0:
                raise RuntimeError("FGP worker failed; see round log")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if args.staged_worker is not None:
        status = json.loads((output / "fgp_summary.json").read_text())
        reaped = json.loads((output / "cpu_reap.json").read_text())
        if (status["outputs"], status["dropped_input_frames"], reaped["completed"], reaped["exitcode"], reaped["alive"]) != (
                args.frames - FGP_WARMUP, 0, args.frames - FGP_WARMUP, 0, False):
            raise ValueError("staged FGP coverage or clean CPU exit mismatch")
        frames = [json.loads(line) for line in (output / "fgp.jsonl").read_text().splitlines()]
        captures = [dict(r, input_ns=r["capture_ns"], begin_ns=r["started_ns"],
                         sending_ns=r["inferred_ns"] + round(r["sender_queue_ms"] * 1e6),
                         sent_ns=r["acknowledged_ns"] - round(r["human_rpc_ms"] * 1e6),
                         confirmed_ns=r["acknowledged_ns"], credit_wait_ns=r["reserve_ns"])
                    for r in frames if r["output"]]
        publish_json(output / "fgp_result.json", dict(frames=captures, peak_inflight=status["max_in_flight"]))
    capture_doc = json.loads((output / "fgp_result.json").read_text())
    captures = capture_doc["frames"]
    accepted = {r["input_ns"]: r["accepted_ns"] for r in captures}
    ready = []
    for seq, reply in enumerate(replies):
        result = reply["result"]
        if result["ready"]:
            if result["state_sequence"] != seq or any(
                    accepted[stamp] > result["cutoff_ns"] for stamp in result["timing"]["raw_support_end_ns"]):
                raise ValueError("bad reference visibility or state sequence")
            ready.append(seq)
    if threads and not ready:
        raise ValueError("no shared FGP and policy cycles")
    active = replies[ready[0]:ready[-1] + 1] if ready else []
    consecutive = [float(np.rad2deg(np.max(np.abs(np.subtract(
        active[i]["result"]["q_target_rad"], active[i-1]["result"]["q_target_rad"])))))
                   for i in range(1, len(active))
                   if active[i]["result"]["ready"] and active[i-1]["result"]["ready"]]
    longest = max((b - a - 1 for a, b in zip(ready, ready[1:])), default=0)

    def detailed(values):
        values = np.asarray(values)
        return dict(**distribution(values), p50=float(np.median(values)),
                    p99=float(np.percentile(values, 99)),
                    above_33_33ms=int(np.count_nonzero(values > 1000/30)),
                    above_50ms=int(np.count_nonzero(values > 50)),
                    above_100ms=int(np.count_nonzero(values > 100)))

    def frame_durations(a, b):
        return detailed([(r[b] - r[a]) / 1e6 for r in captures])

    def frame_spacing(key):
        return detailed(np.diff([r[key] for r in captures]) / 1e6)

    rpc = [(replies[i]["finished_ns"] - replies[i]["sent_ns"]) / 1e6 for i in ready]
    ages = [(replies[i]["result"]["cutoff_ns"] -
             replies[i]["result"]["timing"]["slot_capture_ns"][0]) / 1e6 for i in ready]
    release_ages = [(r["confirmed_ns"] - r["input_ns"]) / 1e6 for r in captures]
    return dict(threads=threads, frames=len(captures), state_ticks=state_ticks,
                peak_inflight=capture_doc["peak_inflight"],
                resources=describe_system(samples, captures[0]["inferred_ns"], captures[-1]["inferred_ns"])
                if sample else None,
                policy_init_new_threads=bindings, service_thread_cpus=service_cpus,
                first_ready_tick=ready[0] if ready else None, last_ready_tick=ready[-1] if ready else None,
                ready=len(ready), unready_within_ready_range=len(active) - len(ready),
                longest_ready_gap_ms=longest * DT_NS / 1e6,
                fgp_output_gap_ms=frame_spacing("inferred_ns"), acknowledgement_gap_ms=frame_spacing("confirmed_ns"),
                fgp_step_ms=frame_durations("begin_ns", "inferred_ns"),
                source_to_ack_ms=detailed(release_ages),
                first_last_quarter_source_to_ack_p50_ms=[float(np.median(part)) for part in
                    (release_ages[:len(release_ages)//4], release_ages[-len(release_ages)//4:])],
                gpu_finish_to_sender_ms=frame_durations("inferred_ns", "sending_ns"),
                sender_rpc_ms=frame_durations("sent_ns", "confirmed_ns"),
                producer_credit_wait_ms=detailed([r["credit_wait_ns"] / 1e6 for r in captures]),
                oldest_reference_age_ms=detailed(ages) if ages else None, state_rpc_ms=detailed(rpc) if rpc else None,
                state_rpc_over_20_ms=sum(v > 20 for v in rpc),
                adjacent_target_step_deg=distribution(consecutive) if consecutive else None)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("imu", "policy_root", "clotho_root", "cano_engine", "calibration", "output"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    cli.add_argument("--fgp-python", type=Path)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    cli.add_argument("--frames", type=int, choices=(FRAMES, 240), default=FRAMES)
    cli.add_argument("--inflight", type=int, choices=(1, 2), default=1)
    cli.add_argument("--staged-worker", type=Path, help="path to the isolated script namespace")
    comparison = cli.add_mutually_exclusive_group()
    comparison.add_argument("--resource-compare", action="store_true", help="compare FGP alone with 2-thread Morph")
    comparison.add_argument("--affinity-compare", action="store_true", help="compare 2-thread Morph unrestricted and on CPUs 6,7")
    args = cli.parse_args()
    if args.worker:
        fgp_main(args)
        return
    if args.fgp_python is None:
        cli.error("--fgp-python is required")
    if args.staged_worker is not None and (args.frames, args.inflight) != (240, 2):
        cli.error("staged worker requires 240 frames and two in flight")
    if (args.resource_compare or args.affinity_compare) and args.staged_worker is None:
        cli.error("resource comparison requires the staged FGP fixture")
    assets = {name: getattr(args, name) for name in ("imu", "cano_engine", "calibration")}
    before = {name: sha(path) for name, path in assets.items()}
    args.output.mkdir()
    counts = (0, 2, 2, 0) if args.resource_compare else ((2, 2, 2, 2) if args.affinity_compare else (4, 2, 2, 4))
    masks = (None, {6, 7}, {6, 7}, None) if args.affinity_compare else (None,) * 4
    if args.affinity_compare and not {6, 7} <= os.sched_getaffinity(0):
        cli.error("CPUs 6 and 7 must both be available")
    rounds = [case(args, threads, args.output / f"r{i}", mask)
              for i, (threads, mask) in enumerate(zip(counts, masks, strict=True))]
    if before != {name: sha(path) for name, path in assets.items()}:
        raise ValueError("source contents changed")
    pose_deltas = []
    with np.load(args.output / "r0" / "fgp_outputs.npz", allow_pickle=False) as reference:
        for i in range(1, len(rounds)):
            with np.load(args.output / f"r{i}" / "fgp_outputs.npz", allow_pickle=False) as candidate:
                differences = {name: float(np.max(np.abs(reference[name] - candidate[name])))
                               for name in ("pose", "translation" if args.staged_worker is not None else "root")}
            if any(value > 2e-5 for value in differences.values()):
                raise ValueError("FGP numerical outputs differ beyond tolerance")
            pose_deltas.append(differences)
    summary = dict(hardware_commands=0, input_hashes=before, imu_frames_per_round=args.frames,
                   inflight_limit=args.inflight, rounds=rounds, fgp_numerical_deltas=pose_deltas,
                   scope="recorded IMU and fixed synthetic feedback, shared loopback",
                   fgp_structure="separate GPU, CPU process, sender" if args.staged_worker is not None
                   else "FGP with separate sender")
    publish_json(args.output / "result.json", summary)


if __name__ == "__main__":
    main()
