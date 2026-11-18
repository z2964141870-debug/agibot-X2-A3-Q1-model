"""Collect and calibrate SmartWear from a terminal, then send advisory HumanFrames."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time


@contextmanager
def capture_log(path):
    with path.open("x", encoding="utf-8") as out:
        remaining = 16 * 1024 * 1024
        warned = False

        def write(event):
            nonlocal remaining, warned
            if warned:
                return
            try:
                line = json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n"
                size = len(line.encode("utf-8"))
                if size > remaining:
                    raise ValueError("16 MiB diagnostic log limit reached")
                out.write(line)
                remaining -= size
            except (OSError, ValueError) as error:
                warned = True
                print(f"{path}: {error}", file=sys.stderr, flush=True)

        yield write


def forward(sock, endpoint, calibration, output):
    from script.x2_morph_human.artifacts import home
    from script.x2_morph_human.client import Client
    from script.x2_morph_human.core import Calibration, SmplAdapter

    adapter = SmplAdapter(home() / "upstream/deploy/vr_teleop/human_skeleton.json",
                          Calibration(**json.loads(calibration.read_text())))
    clock = f"fgp_terminal_monotonic:{socket.gethostname()}:{time.monotonic_ns()}"
    client = None
    with capture_log(output) as log:
        try:
            while True:
                payload = sock.recv(65536)
                replaced = 0
                while select.select([sock], [], [], 0)[0]:
                    payload = sock.recv(65536)
                    replaced += 1
                frame = json.loads(payload)
                now = time.monotonic_ns()
                row = {k: frame[k] for k in ("sequence", "received_ns", "output_ns")}
                row.update(dequeued_ns=now, superseded=replaced)
                if now - frame["received_ns"] > 150_000_000:
                    log(dict(row, status="expired_before_send"))
                    continue
                try:
                    packet = adapter.packet_from_matrices(
                        frame["pose_mat"], frame["root_translation_m"], sequence=frame["sequence"],
                        capture_ns=frame["received_ns"], capture_clock=frame["source_clock"],
                        arrival_ns=now, arrival_clock=clock)
                    if client is None:
                        client = Client(endpoint, timeout_ms=200)
                    row["send_ns"] = time.monotonic_ns()
                    reply = client.request(packet)
                    accepted = reply.get("accepted_sequence") == frame["sequence"]
                    log(dict(row, status="accepted" if accepted else "rejected",
                             ack_ns=time.monotonic_ns(), reply=reply))
                except Exception as error:
                    if client is not None:
                        client.close()
                        client = None
                    log(dict(row, status="forward_error", end_ns=time.monotonic_ns(), error=repr(error)))
        finally:
            if client is not None:
                client.close()


def calibrate(preview, scan_seconds):
    while True:
        deadline = time.monotonic() + scan_seconds + 60
        while preview.source.pair() is None:
            devices = preview.source.snapshot()
            print("  ".join(f'{r}={s["state"]} ({s["packets"]} packets)' for r, s in devices.items()),
                  flush=True)
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for synchronous upper and lower BLE frames")
            preview.stop_event.wait(1)
        input("收到上下衣数据。按 Enter，3 秒后开始采集 T-pose：")
        try:
            preview.command({"action": "calibrate"})
        except ValueError as error:
            print(error, flush=True)
            continue
        while True:
            with preview.lock:
                state = dict(preview.calibration)
                error = preview.error
            name = state["state"]
            print(f'{name}: {state.get("remaining_s", 0):.1f}s, {state.get("samples", 0)} samples',
                  flush=True)
            if name == "static_passed":
                return
            if name not in ("preparing", "collecting"):
                print(f"T-pose 未通过：{error or name}。", flush=True)
                break
            preview.stop_event.wait(.5)


def run(args):
    from .gpu_engine import S1GpuFgp
    from .server import Preview

    args.output.mkdir(parents=True, exist_ok=False)
    with capture_log(args.output / "capture.jsonl") as log:
        engine = S1GpuFgp(args.clotho_root, args.cano_engine, profile_stages=args.profile_stages)
        engine.torch.set_num_threads(1)
        preview = Preview("ble", args.output, {"joints": [], "vertices": []},
                          scan_seconds=args.scan_seconds, engine=engine,
                          live_mesh=False, log_smpl=False, inference_trace=log)
        sender = child = process = None
        preview.start()
        try:
            calibrate(preview, args.scan_seconds)
            sender, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
            sender.setblocking(False)
            # At most the newest frame is used when the advisory receiver is busy.
            process = subprocess.Popen([
                str(args.bridge_python), "-B", "-m", __spec__.name, "--forward-fd", str(child.fileno()),
                "--endpoint", args.endpoint, "--calibration", str(args.calibration),
                "--output", str(args.output / "forward.jsonl")], pass_fds=(child.fileno(),))
            child.close()
            print(f"T-pose 通过。FGP 积累历史帧后开始转发，日志：{args.output}", flush=True)
            last = -1
            while process.poll() is None:
                state = preview.smpl_state()
                with preview.lock:
                    error = preview.error
                    calibration = preview.calibration["state"]
                if calibration == "epoch_changed" or error:
                    raise RuntimeError(error or "衣服重连后需要重新进行 T-pose")
                frame = state["sample"]
                if state["realtime_valid"] and frame["sequence"] > last:
                    last = frame["sequence"]
                    try:
                        sender.send(json.dumps(frame, separators=(",", ":"), allow_nan=False).encode())
                    except BlockingIOError:
                        print(f"丢弃 {last}：转发队列已满", file=sys.stderr, flush=True)
                preview.stop_event.wait(.005)
            raise RuntimeError(f"HumanFrame 转发进程退出：{process.returncode}")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for sock in (sender, child):
                if sock is not None:
                    sock.close()
            preview.stop()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--clotho-root", type=Path)
    p.add_argument("--cano-engine", type=Path)
    p.add_argument("--bridge-python", type=Path)
    p.add_argument("--lock", type=Path, default=Path("data/runtime/x2-fgp-viewer/preview.lock"))
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5776")
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--scan-seconds", type=float, default=180)
    p.add_argument("--profile-stages", action="store_true",
                   help="Include per-frame CANO, LIP, GP and CPU timing in capture.jsonl")
    p.add_argument("--forward-fd", type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.endpoint.startswith("tcp://127.0.0.1:"):
        p.error("只允许本机策略端点")
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    if a.forward_fd is not None:
        try:
            with socket.socket(fileno=a.forward_fd) as sock:
                forward(sock, a.endpoint, a.calibration, a.output)
        except KeyboardInterrupt:
            pass
        return
    if not all((a.clotho_root, a.cano_engine, a.bridge_python)):
        p.error("--clotho-root, --cano-engine and --bridge-python are required")
    try:
        with a.lock.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                p.error("衣服采集已在运行，请先在原终端退出")
            run(a)
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()
