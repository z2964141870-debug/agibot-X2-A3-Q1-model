"""Real FGP GPU/CPU replay, or instrumented unchanged Morph advisory runtime."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

from script.x2_fgp_realtime.assets import file_digest
from .timing import reference_ages


def policy(args):
    import zmq
    from script.x2_morph_human.runtime import Runtime
    runtime = Runtime(max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    clock = "joint_replay_soc1_monotonic"
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.LINGER, 0)
    server.setsockopt(zmq.MAXMSGSIZE, 256*1024)
    server.bind(args.endpoint)
    print(json.dumps(dict(status="ADVISORY_ONLY", publish_allowed=False)), flush=True)
    try:
        with (args.output / "service.jsonl").open("x") as tape:
            while True:
                message = server.recv_json()
                received = time.monotonic_ns()
                try:
                    reply = runtime.handle(message, received, clock)
                    ready_at = time.monotonic_ns()
                    ages = reference_ages(reply, ready_at, clock)
                    if ages:
                        reply["joint_test_timing"] = dict(ages, service_received_ns=received,
                                                         reply_ready_ns=ready_at)
                except Exception as exc:
                    runtime.reset()
                    reply = dict(ready=False, error=str(exc), reset=True, publish_allowed=False)
                server.send_json(reply)
                if message["kind"] == "x2_measurement":
                    tape.write(json.dumps(dict(sequence=message["sequence"], received_ns=received,
                                              sent_ns=time.monotonic_ns(), reply=reply), allow_nan=False)+"\n")
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        context.term()


def fgp(args):
    from script.x2_fgp_accel.graph_session import CudaGraphSession
    from script.x2_fgp_accel.feature_cache import ExactFeatureCache, FEATURE_SOURCES
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator
    from script.x2_morph_human.core import Calibration, SmplAdapter
    from script.x2_morph_human.artifacts import home, verify
    # Both project environments are CPython 3.10 ARM64. Reuse only the existing
    # ZMQ package; keep this process's numpy/torch and all global paths unchanged.
    communication_site = str(Path.cwd()/"data/environments/orin-py310/lib/python3.10/site-packages")
    sys.path.append(communication_site)
    try:
        from script.x2_morph_human.client import Client
    finally:
        sys.path.remove(communication_site)
    source = args.source_root
    if any(file_digest(source / "stage_tran_gp" / name) != digest for name, digest in FEATURE_SOURCES.items()):
        raise ValueError("Stateless feature source changed")
    build = json.loads((args.build_root / "build_receipt.json").read_text())
    engine = args.build_root / "cano.plan"
    if (build["status"] != "BUILD_PASSED" or build["precision"] != "mixed_gemm"
            or file_digest(engine) != build["engine_sha256"]
            or file_digest(source / "models/onnx_s1/cano.onnx") != build["onnx_sha256"]):
        raise ValueError("Original accurate engine identity changed")
    verify()
    cal_doc = json.loads(Path("script/x2_morph_human/config/clotho_fgp_yup.json").read_text())
    adapter = SmplAdapter(home()/"upstream/deploy/vr_teleop/human_skeleton.json", Calibration(**cal_doc))
    prepare_clotho_environment(source)
    import torch
    import articulate as art
    from Aplus.tools.smpl_light import SMPLPose
    from Utils.config import bulid_rot
    from onnx_inference import OnnxBundle, OnnxFlowModel
    from stage_tran_gp.tracker import GlobalPoseTracker
    from stage_tran_gp.features import valid_rotations
    torch.set_num_threads(4)
    fixture = np.load(args.fixture, allow_pickle=False)
    raw = fixture["raw"]
    if raw.shape != (929, 10, 7) or not np.isfinite(raw).all():
        raise ValueError("Expected 929 contiguous IMU frames")
    cal = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
    cal.set_calibration_120(fixture["calibration"])
    initialized = time.monotonic_ns()
    session = CudaGraphSession(engine)
    original_flow, original_tracker = OnnxBundle.flow_model, GlobalPoseTracker.__init__
    caches = []

    # Reuse the audited executor while bounding retained diagnostics to one call.
    # Its original 32-call probe stays unchanged; this separate replay caps at 900.
    class ReplaySession:
        calls = 0

        def run(self, names, feed):
            if self.calls >= 900:
                raise RuntimeError("900-call replay budget exhausted")
            result = session.run(names, feed)
            session.recorded.clear()
            self.calls += 1
            return result

    replay = ReplaySession()
    def flow(bundle):
        if bundle.directory.resolve() != (source/"models/onnx_s1").resolve():
            raise ValueError("Unexpected bundle")
        return OnnxFlowModel(replay, bundle.window_size, latest_only=True)

    def tracker(instance, *positional, **keywords):
        original_tracker(instance, *positional, **keywords)
        if instance.network.fps != 30:
            raise ValueError("Expected original 30Hz GP")
        instance.features = ExactFeatureCache(instance.features)
        caches.append(instance.features)

    pipeline = client = None
    poses, roots, captures, rows = [], [], [], []
    try:
        OnnxBundle.flow_model = flow
        GlobalPoseTracker.__init__ = tracker
        spec = importlib.util.spec_from_file_location("joint_author", source/"online_fgp+LIP_realtime_onnx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.parse_args(["--onnx-dir", str(source/"models/onnx_s1"), "--onnx-threads", "4"])
        pipeline = module.RealtimePipeline(config)
        client = Client(args.endpoint)
        print(json.dumps(dict(status="FGP_READY", initialization_ms=(time.monotonic_ns()-initialized)/1e6)),
              file=sys.__stdout__, flush=True)
        start = int(sys.stdin.readline())
        skipped = 0
        with (args.output / "fgp.jsonl").open("x") as tape, torch.inference_mode():
            index = 0
            while index < len(raw):
                if args.scheduling == "latest":
                    newest_due = min(len(raw)-1, max(0, (time.monotonic_ns()-start)*30//1_000_000_000))
                    if newest_due > index:
                        skipped += newest_due-index
                        index = newest_due
                item = raw[index]
                due = start + index * 1_000_000_000 // 30
                remaining = (due-time.monotonic_ns())/1e9
                if remaining > 0:
                    time.sleep(remaining)
                began = time.monotonic_ns()
                acc, rot = cal.calibrate_ordered_10x7(item)
                calibrated = torch.cat((acc.flatten()/30, rot.flatten())).tolist()
                result = pipeline.step(calibrated, SimpleNamespace(calibrate=lambda raw, acc_scale: raw))
                inferred = time.monotonic_ns()
                row = dict(frame=index, capture_ns=due, started_ns=began, inferred_ns=inferred,
                           queue_ms=(began-due)/1e6, fgp_ms=(inferred-began)/1e6, output=result is not None)
                if result is not None:
                    native = pipeline.gp.last_result
                    pose, root = native["pose"].numpy(), native["translation"].numpy()
                    if not valid_rotations(pose, 24) or not np.isfinite(root).all():
                        raise ValueError("Invalid full FGP output")
                    packet = adapter.packet_from_matrices(pose, root, sequence=index,
                        capture_ns=due, capture_clock="joint_replay_soc1_monotonic",
                        arrival_ns=inferred, arrival_clock="joint_replay_soc1_monotonic")
                    converted = time.monotonic_ns()
                    acknowledgement = client.request(packet)
                    acknowledged = time.monotonic_ns()
                    if acknowledgement.get("accepted_sequence") != index:
                        raise ValueError("HumanFrame acknowledgement failed")
                    row.update(bridge_ms=(converted-inferred)/1e6, human_rpc_ms=(acknowledged-converted)/1e6,
                               software_input_age_at_ack_ms=(acknowledged-due)/1e6,
                               acknowledged_ns=acknowledged, stages=dict(pipeline.perf.frame))
                    poses.append(pose.copy()); roots.append(root.copy()); captures.append(due)
                rows.append(row)
                tape.write(json.dumps(row, allow_nan=False)+"\n")
                index += 1
        if (not 1 <= replay.calls <= 900 or len(caches) != 1 or caches[0].hits != replay.calls
                or (args.scheduling == "fifo" and replay.calls != 900)):
            raise ValueError("Unexpected FGP call/feature reuse count")
        np.savez_compressed(args.output / "fgp_outputs.npz", pose=np.asarray(poses),
                            translation=np.asarray(roots), capture_ns=np.asarray(captures))
        (args.output / "fgp_summary.json").write_text(json.dumps(dict(status="FGP_REPLAY_COMPLETED",
            input_frames=len(rows), scheduled_input_frames=929, outputs=len(poses),
            gpu_calls=replay.calls, feature_hits=caches[0].hits,
            dropped_input_frames=skipped, queue_discipline=args.scheduling,
            training_rate_matches_processed_rate=(skipped == 0),
            fixture_sha256=file_digest(args.fixture), engine_sha256=file_digest(engine),
            native_physics=True, hardware_commands=0, ble_connections=0), indent=2)+"\n")
    finally:
        OnnxBundle.flow_model, GlobalPoseTracker.__init__ = original_flow, original_tracker
        if pipeline is not None:
            pipeline.close()
        session.close()
        if client is not None:
            client.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("policy", "fgp"))
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--build-root", type=Path)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--scheduling", choices=("fifo", "latest"), default="fifo")
    a = p.parse_args()
    if a.mode == "policy":
        policy(a)
    else:
        with contextlib.redirect_stdout(sys.stderr):
            fgp(a)


if __name__ == "__main__":
    main()
