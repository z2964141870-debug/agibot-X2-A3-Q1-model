"""Compare ORT thread counts while recorded IMU passes through FGP and loopback."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue
import subprocess
import time

import numpy as np
import zmq

from . import __path__ as package_paths
from .artifacts import sha
from .bench_human_ingress import CLOCK, SharedHumanBuffer
from .bench_zmq_ingress import fixture_runtime, request_frames, serve, socket
from .core import Calibration, DT_NS, SmplAdapter
from .replay_window_pacing import distribution
from .runtime import synthetic_state


FRAMES = 180
FGP_WARMUP = 29
STATE_TICKS = 450


def read_when_written(path, process=None):
    for _ in range(600):
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
        raw, calibration = packet["raw"][:FRAMES], packet["calibration"]
    if raw.shape != (FRAMES, 10, 7) or calibration.shape != (120,):
        raise ValueError("recorded IMU layout changed")
    engine = S1GpuFgp(args.clotho_root, args.cano_engine)
    try:
        engine.reset(calibration)
        cal = Calibration(**json.loads(args.calibration.read_text()))
        adapter = SmplAdapter(args.policy_root / "upstream/deploy/vr_teleop/human_skeleton.json", cal)
        publish_json(args.output / "worker_ready.json", {"frames": len(raw)})
        start = read_when_written(args.output / "start.json")
        captured = []
        with zmq.Context() as context, socket(context, zmq.REQ) as client:
            client.connect(start["endpoint"])
            for frame, values in enumerate(raw):
                due = start["anchor_ns"] + frame * 1_000_000_000 // 30
                while (remaining := due - time.monotonic_ns()) > 0:
                    time.sleep(remaining / 1e9)
                begin = time.monotonic_ns()
                pose = engine.step(values)
                end = time.monotonic_ns()
                if pose is None:
                    continue
                packet = adapter.packet_from_matrices(
                    pose["pose_mat"], pose["root_translation_m"], sequence=frame,
                    capture_ns=due, capture_clock=CLOCK, arrival_ns=end, arrival_clock=CLOCK)
                sent = time.monotonic_ns()
                client.send_json(packet)
                ack = client.recv_json()["result"]
                confirmed = time.monotonic_ns()
                if ack["accepted_sequence"] != frame:
                    raise ValueError("unexpected HumanFrame acknowledgement")
                captured.append(dict(frame=frame, input_ns=due, begin_ns=begin, inferred_ns=end,
                                     sent_ns=sent, accepted_ns=ack["accepted_ns"], confirmed_ns=confirmed))
        if [row["frame"] for row in captured] != list(range(FGP_WARMUP, FRAMES)):
            raise ValueError("FGP changed output frame coverage")
        publish_json(args.output / "fgp_result.json", captured)
    finally:
        engine.close()


def case(args, threads, output):
    output.mkdir()
    runtime = fixture_runtime(args.policy_root, threads)
    runtime.buffer = SharedHumanBuffer(runtime.buffer)
    options = ("imu", "policy_root", "clotho_root", "cano_engine", "calibration")
    worker = [str(args.fgp_python), "-B", "-c",
              "import json,runpy,sys,script.x2_morph_human as m; "
              "m.__path__[:]=json.loads(sys.argv.pop(1)); sys.path.append(sys.argv.pop(1)); "
              "runpy.run_module('script.x2_morph_human.bench_fgp_ort_joint',run_name='__main__')",
              json.dumps(list(package_paths)), str(Path(zmq.__file__).resolve().parents[1]),
              "--worker", "--output", str(output)]
    for name in options:
        worker.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    with (output / "fgp.log").open("x") as log:
        process = subprocess.Popen(worker, stdout=log, stderr=subprocess.STDOUT)
        try:
            read_when_written(output / "worker_ready.json", process)
            with zmq.Context() as context, ThreadPoolExecutor(max_workers=1) as worker_pool:
                endpoints = Queue()
                guard = worker_pool.submit(serve, context, endpoints, runtime,
                                           STATE_TICKS + FRAMES - FGP_WARMUP, None, False)
                endpoint = endpoints.get(timeout=2)
                anchor = time.monotonic_ns() + 200_000_000
                states = [(tick * DT_NS, json.dumps(synthetic_state(runtime, tick, 0)).encode())
                          for tick in range(STATE_TICKS)]
                publish_json(output / "start.json", dict(endpoint=endpoint, anchor_ns=anchor))
                replies = request_frames(context, endpoint, states, anchor)
                guard.result(timeout=3)
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
    captures = json.loads((output / "fgp_result.json").read_text())
    accepted = {r["input_ns"]: r["accepted_ns"] for r in captures}
    ready = []
    for seq, reply in enumerate(replies):
        result = reply["result"]
        if result["ready"]:
            if result["state_sequence"] != seq or any(
                    accepted[stamp] > result["cutoff_ns"] for stamp in result["timing"]["raw_support_end_ns"]):
                raise ValueError("bad reference visibility or state sequence")
            ready.append(seq)
    if not ready:
        raise ValueError("no shared FGP and policy cycles")
    active = replies[ready[0]:ready[-1] + 1]
    consecutive = [float(np.rad2deg(np.max(np.abs(np.subtract(
        active[i]["result"]["q_target_rad"], active[i-1]["result"]["q_target_rad"])))))
                   for i in range(1, len(active))
                   if active[i]["result"]["ready"] and active[i-1]["result"]["ready"]]
    longest = max((b - a - 1 for a, b in zip(ready, ready[1:])), default=0)

    def frame_durations(a, b):
        return distribution([(r[b] - r[a]) / 1e6 for r in captures])

    def frame_spacing(key):
        return distribution(np.diff([r[key] for r in captures]) / 1e6)

    rpc = [(replies[i]["finished_ns"] - replies[i]["sent_ns"]) / 1e6 for i in ready]
    return dict(threads=threads, frames=len(captures), first_ready_tick=ready[0], last_ready_tick=ready[-1],
                ready=len(ready), unready_within_ready_range=len(active) - len(ready),
                longest_ready_gap_ms=longest * DT_NS / 1e6,
                fgp_output_gap_ms=frame_spacing("inferred_ns"), acknowledgement_gap_ms=frame_spacing("confirmed_ns"),
                fgp_step_ms=frame_durations("begin_ns", "inferred_ns"),
                source_to_ack_ms=frame_durations("input_ns", "confirmed_ns"), state_rpc_ms=distribution(rpc),
                state_rpc_over_20_ms=sum(v > 20 for v in rpc),
                adjacent_target_step_deg=distribution(consecutive))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("imu", "policy_root", "clotho_root", "cano_engine", "calibration", "output"):
        cli.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    cli.add_argument("--fgp-python", type=Path)
    cli.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.worker:
        fgp_main(args)
        return
    if args.fgp_python is None:
        cli.error("--fgp-python is required")
    assets = {name: getattr(args, name) for name in ("imu", "cano_engine", "calibration")}
    before = {name: sha(path) for name, path in assets.items()}
    args.output.mkdir()
    rounds = [case(args, threads, args.output / f"r{i}") for i, threads in enumerate((4, 2, 2, 4))]
    if before != {name: sha(path) for name, path in assets.items()}:
        raise ValueError("source contents changed")
    summary = dict(hardware_commands=0, input_hashes=before, imu_frames_per_round=FRAMES,
                   state_ticks_per_round=STATE_TICKS, rounds=rounds,
                   scope="GPU FGP with recorded IMU, shared loopback, fixed synthetic feedback, serial FGP and send")
    publish_json(args.output / "result.json", summary)


if __name__ == "__main__":
    main()
