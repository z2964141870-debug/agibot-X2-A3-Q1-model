"""Read-only SMPL replay and SmartWear calibration preview. No robot SDK imports."""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from .ble import BleSource
from .calibration import summarize_window


class ReplayBody:
    def __init__(self, clotho_root, recording):
        from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment
        prepare_clotho_environment(clotho_root)
        import torch
        import articulate as art
        self.torch = torch
        torch.set_num_threads(4)
        self.body = art.ParametricModel(str(clotho_root / "models/SMPL_male.pkl"), device="cpu")
        self.faces = np.asarray(self.body.face, dtype=np.int32).ravel().tolist()
        self.parents = [-1 if parent is None else int(parent) for parent in self.body.parent]
        self.data = np.load(recording, allow_pickle=False) if recording else None
        self.length = len(self.data["pose_mat"]) if self.data is not None else 1
        self.times = (self.data["capture_ns"]-self.data["capture_ns"][0])/1e9 if self.data is not None else np.zeros(1)
        if self.data is not None and (self.length < 2 or not np.isfinite(self.times).all() or (np.diff(self.times) <= 0).any()):
            raise ValueError("recording timestamps must increase")
        self.duration = float(self.times[-1]+np.median(np.diff(self.times))) if self.length > 1 else 1.0
        self.name = recording.name if recording else "SMPL template"

    def frame(self, index=None):
        if index is None:
            pose, tran = np.tile(np.eye(3), (24, 1, 1)), np.zeros(3)
        else:
            pose, tran = self.data["pose_mat"][index], self.data["root_translation_m"][index]
        return self.from_pose(pose, tran)

    def from_pose(self, pose, tran):
        torch = self.torch
        with torch.inference_mode():
            _, joints, vertices = self.body.forward_kinematics(
                torch.as_tensor(pose, dtype=torch.float32)[None],
                tran=torch.as_tensor(tran, dtype=torch.float32)[None], calc_mesh=True)
        return np.concatenate((joints.numpy().ravel(), vertices.numpy().ravel())).astype("<f4")

    def template(self):
        frame = self.frame()
        return dict(faces=self.faces, parents=self.parents, joints=frame[:72].tolist(),
                    vertices=frame[72:].tolist(), source="SMPL_male zero-shape T-pose template")


class Preview:
    def __init__(self, mode, output, template, replay=None, scan_seconds=120, engine=None, live_body=None):
        self.mode, self.output, self.template, self.replay = mode, output, template, replay
        self.source = BleSource(scan_seconds=scan_seconds) if mode == "ble" else None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.calibration = dict(state="not_captured", tpose_geometry_verified=False)
        self.calibration_started = None
        self.calibration_rows = []
        self.calibration_epoch = None
        self.last_pair = None
        self.playing = True
        self.index = 0
        self.playhead_s = 0.0
        self.last_tick = time.monotonic()
        self.frame_seq = 0
        self.mesh_bytes = np.asarray(template["joints"] + template["vertices"], dtype="<f4").tobytes()
        self.error = None
        self.engine, self.live_body = engine, live_body
        self.smpl = None
        self.sample_log_bytes = 0
        self.sample_log_limit_bytes = 64*1024*1024
        self.live_status = "waiting_calibration" if engine else "not_enabled"
        self.inference_thread = threading.Thread(target=self.run_inference, daemon=True) if engine else None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        if self.source:
            self.source.start()
        self.thread.start()
        if self.inference_thread:
            self.inference_thread.start()

    def run_inference(self):
        calibration_id, previous_pair, epoch = None, None, None
        initialized = False
        stale_reset = False
        next_tick = time.monotonic()
        while not self.stop_event.wait(max(0, next_tick-time.monotonic())):
            next_tick = max(next_tick+1/30, time.monotonic())
            try:
                with self.lock:
                    calibration = dict(self.calibration)
                if calibration.get("state") != "static_passed":
                    continue
                pair = self.source.pair()
                if pair is None:
                    with self.lock:
                        self.live_status = "stale_hold"
                    if initialized and not stale_reset:
                        self.engine.reset()
                        stale_reset = True
                    continue
                if calibration_id != calibration["captured_unix_ns"]:
                    self.engine.reset(calibration["calibration_data"])
                    calibration_id, epoch = calibration["captured_unix_ns"], pair["epoch"]
                    initialized, stale_reset = True, False
                if pair["epoch"] != epoch:
                    with self.lock:
                        self.calibration["state"] = "epoch_changed"
                        self.live_status = "waiting_calibration"
                    continue
                if previous_pair == pair["received_ns"]:
                    continue
                previous_pair = pair["received_ns"]
                stale_reset = False
                result = self.engine.step(self.source.active_values(pair))
                if result is None:
                    with self.lock:
                        self.live_status = "history_warmup"
                    continue
                mesh = self.live_body.from_pose(result["pose_mat"], result["root_translation_m"])
                pose = result["pose_mat"].astype(np.float64)
                so3 = float(np.max(np.abs(np.swapaxes(pose, -1, -2)@pose-np.eye(3))))
                with self.lock:
                    self.frame_seq += 1
                    self.mesh_bytes = mesh.tobytes()
                    self.live_status = "receiving"
                    self.smpl = dict(sequence=self.frame_seq, received_ns=pair["received_ns"], output_ns=time.monotonic_ns(),
                        pose_mat=pose.tolist(), root_translation_m=result["root_translation_m"].tolist(),
                        joints_yup_m=mesh[:72].reshape(24, 3).tolist(), inference_ms=result["inference_ms"],
                        rotation_orthogonality_max_error=so3, source_clock="receiver monotonic proxy",
                        source="LATEST_CLOTHO_U100H_GLOBALPOSE_CPU", hardware_commands=0,
                        globalpose_integration_dt_s=float(self.engine.gp.dt), nominal_input_hz=30)
                    snapshot = dict(self.smpl)
                self.output.mkdir(parents=True, exist_ok=True)
                encoded = (json.dumps(snapshot, allow_nan=False)+"\n").encode()
                if self.sample_log_bytes+len(encoded) <= self.sample_log_limit_bytes:
                    with (self.output / "smpl.jsonl").open("ab") as stream:
                        stream.write(encoded)
                    self.sample_log_bytes += len(encoded)
            except Exception as error:
                with self.lock:
                    self.error = f"{type(error).__name__}: {error}"
                    self.live_status = "error"
                    self.calibration["state"] = "error"

    def run(self):
        while not self.stop_event.wait(1/30):
            try:
                with self.lock:
                    now = time.monotonic()
                    if self.replay and self.playing and now-self.last_tick >= 1/15:
                        elapsed = now-self.last_tick
                        self.playhead_s = (self.playhead_s+elapsed)%self.replay.duration
                        self.index = min(self.replay.length-1, max(0, int(np.searchsorted(self.replay.times, self.playhead_s, side="right")-1)))
                        self.mesh_bytes = self.replay.frame(self.index).tobytes()
                        self.frame_seq += 1
                        self.last_tick = now
                    if self.calibration_started is not None:
                        self.collect_calibration(now)
            except Exception as error:
                with self.lock:
                    self.error = f"{type(error).__name__}: {error}"
                    self.calibration_started = None
                    self.calibration["state"] = "error"

    def collect_calibration(self, now):
        pair = self.source.pair()
        if pair and pair["received_ns"] != self.last_pair:
            if self.calibration_epoch is None:
                self.calibration_epoch = pair["epoch"]
            if self.calibration_epoch != pair["epoch"]:
                raise ValueError("garment counter epoch changed during calibration")
            self.calibration_rows.append(self.source.active_values(pair))
            self.last_pair = pair["received_ns"]
        elapsed = now-self.calibration_started
        self.calibration.update(samples=len(self.calibration_rows), remaining_s=max(0, 3-elapsed))
        if elapsed >= 3:
            self.calibration_started = None
            result = summarize_window(self.calibration_rows)
            result.update(state="static_passed" if result["static_window_passed"] else "unstable",
                          source="LIVE_BLE_RECEIVER_MONOTONIC", captured_unix_ns=time.time_ns())
            self.calibration = result
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / f"calibration_{time.time_ns()}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")

    def state(self):
        with self.lock:
            sample = self.smpl_state()
            return dict(mode=self.mode, source_name=self.replay.name if self.replay else "SmartWear BLE",
                        reconstruction="RECORDED_LATEST_CLOTHO_GLOBALPOSE" if self.replay else self.live_status,
                        devices=self.source.snapshot() if self.source else {},
                        calibration=self.calibration, frame=self.index, frame_count=self.replay.length if self.replay else 0,
                        playing=self.playing, frame_seq=self.frame_seq, error=self.error,
                        replay_playhead_s=self.playhead_s if self.replay else None,
                        replay_duration_s=self.replay.duration if self.replay else None,
                        smpl_age_ms=sample["age_ms"], smpl_realtime_valid=sample["realtime_valid"],
                        smpl_inference_ms=self.smpl["inference_ms"] if self.smpl else None,
                        smpl_so3_error=self.smpl["rotation_orthogonality_max_error"] if self.smpl else None,
                        model_profile=self.engine.model_profile if self.engine else None,
                        sample_log_bytes=self.sample_log_bytes, sample_log_limit_bytes=self.sample_log_limit_bytes,
                        hardware_commands=0, feedback_subscriptions=0, publish_allowed=False,
                        source_clock="recorded stream" if self.replay else "receiver monotonic proxy, not hardware capture")

    def smpl_state(self):
        with self.lock:
            sample = dict(self.smpl) if self.smpl else None
            return dict(status=self.live_status, sample=sample,
                        age_ms=(time.monotonic_ns()-sample["received_ns"])/1e6 if sample else None,
                        realtime_valid=self.live_status == "receiving" and sample is not None and time.monotonic_ns()-sample["received_ns"] <= 150_000_000)

    def command(self, command):
        with self.lock:
            if command.get("action") == "calibrate":
                if not self.source or self.source.pair() is None:
                    raise ValueError("both garments must have fresh synchronized IMU data")
                if self.calibration_started is not None:
                    raise ValueError("calibration already collecting")
                self.calibration_started = time.monotonic()
                self.calibration_rows, self.calibration_epoch, self.last_pair = [], None, None
                self.calibration = dict(state="collecting", samples=0, remaining_s=3, tpose_geometry_verified=False)
                self.error = None
            elif command.get("action") == "play" and self.replay:
                if not isinstance(command.get("playing"), bool):
                    raise ValueError("playing must be a boolean")
                self.playing = command["playing"]
                self.last_tick = time.monotonic()
            elif command.get("action") == "seek" and self.replay:
                index = int(command["frame"])
                if not 0 <= index < self.replay.length:
                    raise ValueError("frame out of range")
                self.index, self.last_tick = index, time.monotonic()
                self.playhead_s = float(self.replay.times[index])
                self.mesh_bytes = self.replay.frame(index).tobytes()
                self.frame_seq += 1
            else:
                raise ValueError("unsupported preview command")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.inference_thread:
            self.inference_thread.join(timeout=15)
        return self.source.stop() if self.source else not self.thread.is_alive()


def make_handler(preview, three_root, lucide_path):
    web = Path(__file__).parent / "web"
    token = secrets.token_urlsafe(32)
    assets = {"/": web/"index.html", "/app.js": web/"app.js", "/style.css": web/"style.css",
              "/vendor/three.module.js": three_root/"build/three.module.js",
              "/vendor/three.core.js": three_root/"build/three.core.js",
              "/vendor/OrbitControls.js": three_root/"examples/jsm/controls/OrbitControls.js",
              "/vendor/lucide.js": lucide_path}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, kind="application/json", status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/state":
                self.send(preview.state())
            elif path == "/api/config":
                self.send(dict(token=token, parents=preview.template["parents"], faces=preview.template["faces"]))
            elif path == "/api/smpl":
                self.send(preview.smpl_state())
            elif path == "/api/frame":
                with preview.lock:
                    frame = preview.mesh_bytes
                self.send(frame, "application/octet-stream")
            elif path == "/favicon.ico":
                self.send(b"", status=204)
            elif path in assets and assets[path].is_file():
                file = assets[path]
                self.send(file.read_bytes(), "text/javascript" if file.suffix == ".js" else mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            else:
                self.send(dict(error="not found"), status=404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/command" or not secrets.compare_digest(self.headers.get("X-Preview-Token", ""), token):
                self.send(dict(error="invalid preview request"), status=403)
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4096:
                    raise ValueError("request too large or empty")
                preview.command(json.loads(self.rfile.read(size)))
                self.send(dict(ok=True))
            except (ValueError, TypeError, KeyError) as error:
                self.send(dict(error=str(error)), status=400)

    return Handler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("ble", "replay"), default="replay")
    parser.add_argument("--clotho-root", type=Path)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--export-template", type=Path)
    parser.add_argument("--three-root", type=Path)
    parser.add_argument("--lucide", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/analysis/X2_FGP_PREVIEW"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--scan-seconds", type=float, default=120)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()
    replay = engine = live_body = None
    if args.export_template:
        if not args.clotho_root:
            parser.error("template export needs --clotho-root")
        template = ReplayBody(args.clotho_root, None).template()
        args.export_template.parent.mkdir(parents=True, exist_ok=True)
        args.export_template.write_text(json.dumps(template, separators=(",", ":"), allow_nan=False))
        return
    if args.mode == "replay":
        if not args.clotho_root or not args.recording:
            parser.error("replay requires --clotho-root and --recording")
        replay = ReplayBody(args.clotho_root, args.recording)
        template = replay.template()
    else:
        if args.reconstruct:
            if not args.clotho_root:
                parser.error("reconstruction needs --clotho-root")
            from .engine import LatestFgp
            engine = LatestFgp(args.clotho_root)
            live_body = ReplayBody(args.clotho_root, None)
            template = live_body.template()
        elif not args.template:
            parser.error("BLE transport preview requires --template")
        else:
            template = json.loads(args.template.read_text())
    if not args.three_root or not args.lucide:
        parser.error("local Three.js and Lucide assets required")
    preview = Preview(args.mode, args.output, template, replay, args.scan_seconds, engine, live_body)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(preview, args.three_root, args.lucide))
    preview.start()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "session.json").write_text(json.dumps(dict(mode=args.mode, started_unix_ns=time.time_ns(),
        reconstruction_enabled=bool(engine), hardware_commands=0, feedback_subscriptions=0,
        publish_allowed=False), indent=2)+"\n")
    print(json.dumps(dict(url=f"http://{args.host}:{args.port}", mode=args.mode, hardware_commands=0)), flush=True)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(json.dumps(dict(owned_threads_stopped=preview.stop())), flush=True)


if __name__ == "__main__":
    main()
