"""One GPU owner, ordered author CPU state, and a separate HumanFrame sender."""
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
from .pipeline import BoundedPipeline, PreparedDenoiser


def tune_sessions():
    import onnxruntime as ort
    original = ort.InferenceSession
    def configured(model, sess_options=None, *args, **kwargs):
        options = sess_options or ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return original(model, options, *args, **kwargs)
    ort.InferenceSession = configured


def fgp(args):
    tune_sessions()
    from script.x2_fgp_accel.graph_session import CudaGraphSession
    from script.x2_fgp_accel.feature_cache import ExactFeatureCache, FEATURE_SOURCES
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator
    from script.x2_morph_human.core import Calibration
    from script.x2_joint_perf.adapter import BatchedSmplAdapter
    from script.x2_morph_human.artifacts import home, verify
    communication_site = str(Path.cwd()/"data/environments/orin-py310/lib/python3.10/site-packages")
    sys.path.append(communication_site)
    try:
        from script.x2_morph_human.client import Client
    finally:
        sys.path.remove(communication_site)
    source = args.source_root
    if any(file_digest(source/"stage_tran_gp"/name) != sha for name, sha in FEATURE_SOURCES.items()):
        raise ValueError("Feature source changed")
    build = json.loads((args.build_root/"build_receipt.json").read_text())
    engine = args.build_root/"cano.plan"
    if (build["status"] != "BUILD_PASSED" or build["precision"] != "mixed_gemm"
            or file_digest(engine) != build["engine_sha256"]
            or file_digest(source/"models/onnx_s1/cano.onnx") != build["onnx_sha256"]):
        raise ValueError("Accurate engine identity changed")
    verify()
    calibration = Calibration(**json.loads(Path("script/x2_morph_human/config/clotho_fgp_yup.json").read_text()))
    adapter = BatchedSmplAdapter(home()/"upstream/deploy/vr_teleop/human_skeleton.json", calibration)
    prepare_clotho_environment(source)
    import torch
    import articulate as art
    from Aplus.tools.smpl_light import SMPLPose
    from Utils.config import bulid_rot
    from onnx_inference import OnnxBundle, OnnxFlowModel
    from stage_tran_gp.tracker import GlobalPoseTracker
    from stage_tran_gp.features import valid_rotations
    torch.set_num_threads(1)
    fixture = np.load(args.fixture, allow_pickle=False)
    raw = fixture["raw"]
    if raw.shape != (929, 10, 7) or not np.isfinite(raw).all():
        raise ValueError("Expected the original 929-frame fixture")
    cal = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
    cal.set_calibration_120(fixture["calibration"])
    reference = None
    indices = None
    if args.parity_reference:
        reference = args.parity_reference
        rows = [json.loads(line) for line in (reference/"fgp.jsonl").read_text().splitlines()]
        selected = [r for r in rows if r["output"]][:64]
        if len(selected) != 64:
            raise ValueError("Require 64 saved reference outputs")
        indices = [r["frame"] for r in rows if r["frame"] <= selected[-1]["frame"]]
        if len(indices) != 93 or len(set(indices)) != 93:
            raise ValueError("Expected exact 29-history plus 64-output input prefix")
    session = CudaGraphSession(engine)
    old_flow, old_tracker = OnnxBundle.flow_model, GlobalPoseTracker.__init__
    caches, calls = [], [0]
    class ReplaySession:
        def run(self, names, feed):
            if calls[0] >= (64 if reference else 900):
                raise RuntimeError("Pipeline GPU call bound exhausted")
            result = session.run(names, feed)
            session.recorded.clear()
            calls[0] += 1
            return result
    def flow(bundle):
        if bundle.directory.resolve() != (source/"models/onnx_s1").resolve():
            raise ValueError("Unexpected bundle")
        return OnnxFlowModel(ReplaySession(), bundle.window_size, latest_only=True)
    def tracker(instance, *args, **kwargs):
        old_tracker(instance, *args, **kwargs)
        if instance.network.fps != 30:
            raise ValueError("Original GP must remain 30Hz")
        instance.features = ExactFeatureCache(instance.features)
        caches.append(instance.features)
    pipeline = scheduler = None
    client = [None]
    poses, roots, captures, frames, rows = [], [], [], [], []
    try:
        OnnxBundle.flow_model, GlobalPoseTracker.__init__ = flow, tracker
        spec = importlib.util.spec_from_file_location("pipeline_author", source/"online_fgp+LIP_realtime_onnx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.parse_args(["--onnx-dir", str(source/"models/onnx_s1"), "--onnx-threads", "1"])
        pipeline = module.RealtimePipeline(config)
        denoiser = pipeline.denoiser
        prepared = PreparedDenoiser()
        pipeline.denoiser = prepared
        identity = SimpleNamespace(calibrate=lambda raw, acc_scale: raw)

        @torch.inference_mode()
        def process(item):
            row, calibrated, acc, rot = item
            row["cpu_started_ns"] = time.monotonic_ns()
            prepared.bind(acc, rot)
            result = pipeline.step(calibrated, identity)
            inferred = time.monotonic_ns()
            if result is None or prepared.value is not None:
                raise ValueError("Author CPU path did not consume exactly one prepared frame")
            native = pipeline.gp.last_result
            pose, root = native["pose"].numpy().copy(), native["translation"].numpy().copy()
            if not valid_rotations(pose, 24) or not np.isfinite(root).all():
                raise ValueError("Invalid author CPU output")
            row.update(inferred_ns=inferred, fgp_ms=(inferred-row["started_ns"])/1e6,
                       cpu_stage_ms=(inferred-row["cpu_started_ns"])/1e6,
                       cpu_queue_ms=(row["cpu_started_ns"]-row["denoised_ns"])/1e6,
                       stages=dict(pipeline.perf.frame))
            return row, pose, root

        def send(item):
            row, pose, root = item
            send_started = time.monotonic_ns()
            packet = adapter.packet_from_matrices(pose, root, sequence=row["frame"],
                capture_ns=row["capture_ns"], capture_clock="joint_replay_soc1_monotonic",
                arrival_ns=row["inferred_ns"], arrival_clock="joint_replay_soc1_monotonic")
            converted = time.monotonic_ns()
            if reference:
                acknowledgement = dict(accepted_sequence=row["frame"])
            else:
                if client[0] is None:
                    client[0] = Client(args.endpoint)
                acknowledgement = client[0].request(packet)
            acknowledged = time.monotonic_ns()
            if acknowledgement.get("accepted_sequence") != row["frame"]:
                raise ValueError("HumanFrame acknowledgement failed")
            row.update(bridge_ms=(converted-send_started)/1e6,
                       sender_queue_ms=(send_started-row["inferred_ns"])/1e6,
                       human_rpc_ms=(acknowledged-converted)/1e6,
                       software_input_age_at_ack_ms=(acknowledged-row["capture_ns"])/1e6,
                       acknowledged_ns=acknowledged)
            poses.append(pose); roots.append(root); captures.append(row["capture_ns"]); frames.append(row["frame"])
            rows.append(row)
        def close_sender():
            if client[0] is not None:
                client[0].close()
        scheduler = BoundedPipeline(process, send, close_sender=close_sender)
        print(json.dumps(dict(status="FGP_READY", in_flight_limit=2)), file=sys.__stdout__, flush=True)
        start = time.monotonic_ns() if reference else int(sys.stdin.readline())
        skipped = index = cursor = 0
        with torch.inference_mode():
            while (cursor < len(indices)) if reference else (index < len(raw)):
                scheduler.reserve()
                handed_off = False
                try:
                    if reference:
                        index = indices[cursor]
                    else:
                        newest = min(len(raw)-1, max(0, (time.monotonic_ns()-start)*30//1_000_000_000))
                        if newest > index:
                            skipped += newest-index
                            index = newest
                    due = time.monotonic_ns() if reference else start+index*1_000_000_000//30
                    remaining = (due-time.monotonic_ns())/1e9
                    if remaining > 0:
                        time.sleep(remaining)
                    began = time.monotonic_ns()
                    acc, rot = cal.calibrate_ordered_10x7(raw[index])
                    calibrated = torch.cat((acc.flatten()/30, rot.flatten())).tolist()
                    # Preserve the author's calibrated float32 list round trip.
                    acc = torch.as_tensor(calibrated[:30], dtype=torch.float32).reshape(10, 3)*30
                    rot = torch.as_tensor(calibrated[30:], dtype=torch.float32).reshape(10, 3, 3)
                    acc, rot = denoiser.step(acc, rot)
                    denoised = time.monotonic_ns()
                    ready = len(denoiser.acc_buffer) == denoiser.window_size
                    row = dict(frame=index, capture_ns=due, started_ns=began,
                               denoised_ns=denoised, gpu_stage_ms=(denoised-began)/1e6,
                               queue_ms=(began-due)/1e6, output=ready)
                    if ready:
                        scheduler.submit((row, calibrated, acc, rot))
                        handed_off = True
                    else:
                        row.update(inferred_ns=denoised, fgp_ms=(denoised-began)/1e6)
                        rows.append(row)
                finally:
                    if not handed_off:
                        scheduler.release()
                index += 1
                cursor += 1
        scheduler.close()
        if len(caches) != 1 or caches[0].hits != calls[0] or len(poses) != calls[0]:
            raise ValueError("GPU, feature, CPU, and sender counts must agree")
        if reference:
            with np.load(reference/"fgp_outputs.npz", allow_pickle=False) as original:
                differences = {k:float(np.max(np.abs(np.asarray(value)-original[k][:64])))
                               for k,value in (("pose",poses),("translation",roots))}
            if frames != [r["frame"] for r in selected] or max(differences.values()) > 2e-5:
                raise ValueError("Full CPU/GPU pipeline prefix differs: "+str(differences))
            (args.output/"parity.json").write_text(json.dumps(dict(
                status="PIPELINE_PREFIX_EQUIVALENCE_PASSED", outputs=64,
                input_frames=indices, reference=str(reference),
                reference_pose_sha256=file_digest(reference/"fgp_outputs.npz"),
                reference_timing_sha256=file_digest(reference/"fgp.jsonl"),
                fixture_sha256=file_digest(args.fixture), engine_sha256=file_digest(engine),
                max_absolute_difference=differences, tolerance=2e-5,
                real_policy_calls=0, hardware_commands=0, wall_clock_online_tested=False), indent=2)+"\n")
        (args.output/"fgp_summary.json").write_text(json.dumps(dict(
            status="FGP_REPLAY_COMPLETED", input_frames=len(rows), scheduled_input_frames=929,
            outputs=len(poses), gpu_calls=calls[0], feature_hits=caches[0].hits,
            dropped_input_frames=skipped, queue_discipline="fixed_prefix" if reference else "latest",
            training_rate_matches_processed_rate=(skipped == 0 and not reference),
            in_flight_limit=2, max_in_flight=scheduler.maximum,
            submitted=scheduler.submitted, completed=scheduler.completed, dropped_after_denoising=0,
            fixture_sha256=file_digest(args.fixture), engine_sha256=file_digest(engine),
            native_physics=True, hardware_commands=0, ble_connections=0), indent=2)+"\n")
    finally:
        try:
            if scheduler is not None:
                scheduler.close()
        finally:
            np.savez_compressed(args.output/"fgp_outputs.npz", pose=np.asarray(poses),
                translation=np.asarray(roots), capture_ns=np.asarray(captures), frame=np.asarray(frames))
            with (args.output/"fgp.jsonl").open("x") as f:
                for row in sorted(rows,key=lambda r:r["frame"]):
                    f.write(json.dumps(row,allow_nan=False)+"\n")
            OnnxBundle.flow_model, GlobalPoseTracker.__init__ = old_flow, old_tracker
            if pipeline is not None:
                pipeline.close()
            session.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("fgp",))
    p.add_argument("--endpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--scheduling", choices=("latest",), default="latest")
    p.add_argument("--parity-reference", type=Path)
    a = p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        fgp(a)


if __name__ == "__main__":
    main()
