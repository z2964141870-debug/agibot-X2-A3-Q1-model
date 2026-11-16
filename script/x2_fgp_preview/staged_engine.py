"""Split S1 CANO and ordered CPU reconstruction for headless capture."""

from contextlib import contextmanager
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from .gpu_engine import _CalibratorAdapter, _load_realtime_module, _sha256
from script.x2_joint_pipeline.pipeline import PreparedDenoiser


def author_args(module, root):
    models = root / "models"
    return module.parse_args([
        "--onnx-dir", str(models / "onnx_s1"), "--gp-onnx-dir", str(models / "onnx_s1"),
        "--onnx-threads", "1", "--gp-onnx-threads", "1", "--backend", "onnx",
        "--device", "cpu", "--translation-mode", "gp", "--smpl-model", str(models / "SMPL_male.pkl"),
    ])


class ResettableDenoiser(PreparedDenoiser):
    def reset(self):
        self.value = None


@contextmanager
def cpu_pipeline(root):
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment
    from script.x2_fgp_accel.feature_cache import ExactFeatureCache, FEATURE_SOURCES

    if any(_sha256(root / "stage_tran_gp" / name) != digest for name, digest in FEATURE_SOURCES.items()):
        raise ValueError("Feature source identity changed")
    prepare_clotho_environment(root)
    import torch
    from onnx_inference import OnnxBundle
    from stage_tran_gp.backend import NativePhysics
    from stage_tran_gp.tracker import GlobalPoseTracker

    torch.set_num_threads(1)
    module = _load_realtime_module(root)
    prepared = ResettableDenoiser()
    original_flow = OnnxBundle.flow_model
    original_init = GlobalPoseTracker.__init__

    def cache_features(tracker, *args, **kwargs):
        original_init(tracker, *args, **kwargs)
        tracker.features = ExactFeatureCache(tracker.features)

    try:
        OnnxBundle.flow_model = lambda bundle: None
        GlobalPoseTracker.__init__ = cache_features
        module.RealtimeIMUDenoiser = lambda *args, **kwargs: prepared
        pipeline = module.RealtimePipeline(author_args(module, root))
    finally:
        OnnxBundle.flow_model = original_flow
        GlobalPoseTracker.__init__ = original_init
    try:
        if not isinstance(pipeline.gp.tracker.physics, NativePhysics):
            raise RuntimeError("Expected native GlobalPose physics")
        yield pipeline, prepared, torch
    finally:
        pipeline.close()


def reconstruction_worker(pipe, root):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        with pipe, cpu_pipeline(root) as (pipeline, prepared, torch):
            identity = SimpleNamespace(calibrate=lambda raw, acc_scale: raw)
            pipe.send(("ready", os.getpid()))
            while True:
                try:
                    operation, packet = pipe.recv()
                except EOFError:
                    break
                if operation == "close":
                    break
                if operation == "reset":
                    pipeline.reset()
                    pipe.send(("reset", None))
                    continue
                if operation != "frame":
                    raise ValueError("Unknown reconstruction operation")
                sequence, calibrated, acc, rot = packet
                started = time.monotonic_ns()
                before = pipeline.gp.tracker.last_result
                with torch.inference_mode():
                    prepared.bind(torch.from_numpy(acc), torch.from_numpy(rot))
                    result = pipeline.step(calibrated, identity)
                if prepared.value is not None:
                    raise ValueError("Author pipeline did not consume the denoised frame")
                finished = time.monotonic_ns()
                stages = dict(pipeline.perf.frame)
                current = pipeline.gp.tracker.last_result
                if current is not None and current is not before:
                    diagnostics = current["diagnostics"]
                    stages.update(gp_ms=diagnostics["gp_ms"], native_physics_ms=diagnostics["physics_ms"])
                output = None
                if result is not None:
                    native = pipeline.gp.last_result
                    output = dict(pose_mat=native["pose"].detach().numpy().copy(),
                                  root_translation_m=native["translation"].detach().numpy().copy())
                    if not all(np.isfinite(values).all() for values in output.values()):
                        raise ValueError("Non-finite S1 reconstruction")
                pipe.send(("frame", (sequence, output, started, finished, stages)))
    except (BrokenPipeError, ConnectionResetError):
        return


class StagedS1:
    """One CANO frame and one reconstruction may be in progress at once.

    Output carries the original frame's sequence and source timestamp. Call
    collect(0) independently of new input so a completed frame can leave promptly.
    """

    def __init__(self, root, engine, *, startup_timeout=35, frame_timeout=1):
        root, engine = Path(root).resolve(), Path(engine).resolve()
        if _sha256(engine) != "89cf34659df7d686abd4230fcd6b1c17a47b05b9d46dda4ecdbad5381d0f3f08":
            raise ValueError("Audited mixed_gemm engine identity mismatch")
        self.frame_timeout = frame_timeout
        self.pending = None
        self.sequence = 0
        self.session = None
        self.closed = False
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=reconstruction_worker, args=(child, root), name="fgp-reconstruction")
        try:
            self.process.start()
        except BaseException:
            self.pipe.close()
            child.close()
            raise
        child.close()
        try:
            self._receive("ready", startup_timeout)
            self._init_cano(root, engine)
        except BaseException:
            self.close()
            raise

    def _init_cano(self, root, engine):
        from script.x2_fgp_accel.graph_session import CudaGraphSession
        from script.x2_morph_human.clotho_fgp_runner import ClothoCalibrator, prepare_clotho_environment

        prepare_clotho_environment(root)
        import articulate as art
        from Aplus.tools.smpl_light import SMPLPose
        from Utils.config import bulid_rot
        from onnx_inference import OnnxBundle, OnnxFlowModel
        import torch

        torch.set_num_threads(1)
        self.torch = torch
        self.calibrator = _CalibratorAdapter(ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori), torch)
        module = _load_realtime_module(root)
        args = author_args(module, root)
        bundle = OnnxBundle(Path(args.onnx_dir), threads=1)
        self.session = CudaGraphSession(engine)
        session = self.session

        class LiveSession:
            def run(self, names, feed):
                result = session.run(names, feed)
                session.recorded.clear()
                return result

        self.denoiser = module.RealtimeIMUDenoiser(
            ckpt_path=str(args.cano_ckpt), model_size=bundle.model_size,
            window_size=bundle.window_size, chunk_size=4, stride=2, steps=1, interval=1,
            output_mode="causal_ema", ema_alpha=.08, acc_scale=30., use_fp16=module.device.type == "cuda",
            every_frame=True, torch_compile=False,
            onnx_model=OnnxFlowModel(LiveSession(), bundle.window_size, latest_only=True),
        )

    def _receive(self, expected, timeout):
        if not self.pipe.poll(timeout):
            raise TimeoutError(f"FGP CPU stage timed out waiting for {expected}")
        kind, value = self.pipe.recv()
        if kind != expected:
            raise RuntimeError(f"FGP CPU stage returned {kind}, expected {expected}")
        return value

    def reset(self, calibration=None):
        if self.pending is not None:
            self.collect(self.frame_timeout)
        self.pipe.send(("reset", None))
        self._receive("reset", self.frame_timeout)
        if calibration is not None:
            self.calibrator.calibrator.set_calibration_120(calibration)
        self.denoiser.reset()

    def submit(self, raw, source_ns):
        started = time.monotonic_ns()
        with self.torch.inference_mode():
            calibrated = self.calibrator.calibrate(raw)
            acc = self.torch.as_tensor(calibrated[:30], dtype=self.torch.float32).reshape(10, 3) * 30
            rot = self.torch.as_tensor(calibrated[30:], dtype=self.torch.float32).reshape(10, 3, 3)
            cano_started = time.monotonic_ns()
            acc, rot = self.denoiser.step(acc, rot)
        denoised = time.monotonic_ns()
        previous = self.collect(self.frame_timeout) if self.pending is not None else None
        if len(self.denoiser.acc_buffer) >= self.denoiser.window_size:
            self.sequence += 1
            self.pending = dict(sequence=self.sequence, source_ns=source_ns, started_ns=started,
                                calibration_ms=(cano_started-started)/1e6, cano_ms=(denoised-cano_started)/1e6,
                                denoised_ns=denoised)
            self.pipe.send(("frame", (self.sequence, calibrated, acc.numpy(), rot.numpy())))
        return previous

    def collect(self, timeout=0):
        if self.pending is None or (timeout == 0 and not self.pipe.poll()):
            return None
        sequence, result, started, finished, stages = self._receive("frame", timeout)
        row, self.pending = self.pending, None
        if sequence != row["sequence"]:
            raise ValueError("FGP CPU stage returned an out-of-order frame")
        returned = time.monotonic_ns()
        row.update(cpu_started_ns=started, cpu_finished_ns=finished, returned_ns=returned,
                   cpu_queue_ms=(started-row["denoised_ns"])/1e6, cpu_ms=(finished-started)/1e6,
                   inference_ms=(returned-row["started_ns"])/1e6, stages=stages, output=result)
        return row

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                try:
                    self.pipe.send(("close", None))
                except OSError:
                    pass
                self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        finally:
            self.pipe.close()
            if self.session is not None:
                self.session.close()
