"""S1 FGP preview adapter backed by the audited TensorRT engines."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

from script.x2_fgp_accel.graph_session import CudaGraphSession
from script.x2_fgp_accel.trt_session import TrtSession


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_realtime_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "clotho_fgp_realtime_entry", root / "online_fgp+LIP_realtime_onnx.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load CLOTHO realtime entry point")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CalibratorAdapter:
    def __init__(self, calibrator, torch, *, profile_stages=False, numpy_candidate=False):
        self.calibrator = calibrator
        self.torch = torch
        self.profile_stages = profile_stages
        self.elapsed_ms = None
        self.numpy_candidate = numpy_candidate
        self.numpy_constants = None

    def calibrate(self, raw, acc_scale=30):
        started = time.perf_counter_ns() if self.profile_stages else None
        values = np.asarray(raw, dtype=np.float32)
        packed = self._numpy(values, acc_scale) if self.numpy_candidate else None
        if packed is None:
            acc, rot = self.calibrator.calibrate_ordered_10x7(values)
            packed = self.torch.cat((acc.flatten() / float(acc_scale), rot.flatten())).tolist()
        if started is not None:
            self.elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        return packed

    def _numpy(self, values, scale):
        if (values.shape != (10, 7) or not np.isfinite(values).all()
                or self.calibrator.smpl2imu is None or not np.isfinite(scale) or scale == 0):
            return None
        q = values[:, 3:]
        norm = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
        if (norm <= 0).any() or not np.isfinite(norm).all():
            return None
        if self.numpy_constants is None:
            self.numpy_constants = tuple(t.detach().cpu().numpy() for t in
                (self.calibrator.smpl2imu, self.calibrator.device2bone, self.calibrator.acc_offset))
        world, bone, gravity = self.numpy_constants
        a, b, c, d = (q / norm).T
        rot = np.stack((
            -2*c*c - 2*d*d + 1, 2*b*c - 2*a*d, 2*a*c + 2*b*d,
            2*b*c + 2*a*d, -2*b*b - 2*d*d + 1, 2*c*d - 2*a*b,
            2*b*d - 2*a*c, 2*a*b + 2*c*d, -2*b*b - 2*c*c + 1,
        ), axis=1).reshape(10, 3, 3)
        acc = world @ (rot @ values[:, :3, None] - gravity)
        return np.concatenate((acc.reshape(-1) / np.float32(scale), (world @ rot @ bone).reshape(-1))).tolist()


class S1GpuFgp:
    """Expose the preview engine's ``reset/step/close`` contract for S1 GPU."""

    def __init__(self, clotho_root: Path, cano_engine: Path, gp_engine: Path | None = None, *,
                 cuda_graph=True, profile_stages=False, record_cano_io=True, pin_cano_output=False,
                 numpy_calibration=False):
        self.clotho_root = Path(clotho_root).resolve()
        self.cano_engine = Path(cano_engine).resolve()
        self.gp_engine = None
        for path in (self.clotho_root, self.cano_root, self.cano_engine):
            if not path.exists():
                raise FileNotFoundError(path)
        if self.cano_engine.stat().st_size == 0:
            raise ValueError("TensorRT engine is empty")

        if _sha256(self.cano_engine) != "89cf34659df7d686abd4230fcd6b1c17a47b05b9d46dda4ecdbad5381d0f3f08":
            raise ValueError("Audited mixed_gemm engine identity mismatch")

        from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator

        prepare_clotho_environment(self.clotho_root)
        import articulate as art
        import torch
        from Aplus.tools.smpl_light import SMPLPose
        from Utils.config import bulid_rot
        from onnx_inference import OnnxBundle, OnnxFlowModel
        from script.x2_fgp_accel.feature_cache import ExactFeatureCache, FEATURE_SOURCES
        from stage_tran_gp.tracker import GlobalPoseTracker
        if any(_sha256(self.clotho_root / "stage_tran_gp" / name) != digest for name, digest in FEATURE_SOURCES.items()):
            raise ValueError("Feature source identity changed")
        torch.set_num_threads(1)

        self.torch = torch
        self.art = art
        self.source = _load_realtime_module(self.clotho_root)
        onnx_root = self.clotho_root / "models" / "onnx_s1"
        self.cano_session = (CudaGraphSession if cuda_graph else TrtSession)(
            self.cano_engine, record_io=record_cano_io, pin_output=pin_cano_output)



        original_flow_model = OnnxBundle.flow_model
        original_tracker_init = GlobalPoseTracker.__init__
        session = self.cano_session
        class LiveSession:
            def run(self, names, feed):
                result = session.run(names, feed)
                session.recorded.clear()
                return result

        def flow_model(bundle):
            if bundle.directory.resolve() != onnx_root.resolve():
                raise ValueError("Unexpected S1 ONNX directory")
            return OnnxFlowModel(LiveSession(), bundle.window_size, latest_only=True)

        def cached_tracker(tracker, *args, **kwargs):
            original_tracker_init(tracker, *args, **kwargs)
            tracker.features = ExactFeatureCache(tracker.features)

        OnnxBundle.flow_model = flow_model
        GlobalPoseTracker.__init__ = cached_tracker
        try:
            args = self.source.parse_args([
                "--onnx-dir", str(onnx_root),
                "--gp-onnx-dir", str(onnx_root),
                "--onnx-threads", "1",
                "--gp-onnx-threads", "1",
                "--backend", "onnx",
                "--device", "cpu",
                "--translation-mode", "gp",
                "--smpl-model", str(self.clotho_root / "models" / "SMPL_male.pkl"),
            ])
            self.pipeline = self.source.RealtimePipeline(args)
        except BaseException:
            self.cano_session.close()
            raise
        finally:
            OnnxBundle.flow_model = original_flow_model
            GlobalPoseTracker.__init__ = original_tracker_init

        self.calibrator = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
        self.calibrator_adapter = _CalibratorAdapter(self.calibrator, torch, profile_stages=profile_stages,
                                                     numpy_candidate=numpy_calibration)
        self.profile_stages = profile_stages
        self.last_step_profile = None
        self.model_profile = dict(
            selection="CLOTHO fgp_realtime S1 bundle",
            backend="TensorRT CUDA Graph",
            device="GPU",
            cano="onnx_s1 / audited mixed_gemm engine",
            gp="onnx_s1 ONNX Runtime CPU",
            lip="onnx_s1 ONNX Runtime CPU",
            physics="native carticulate CPU",
            cano_engine_sha256=_sha256(self.cano_engine),
            cano_engine=str(self.cano_engine),
            denoise_window_frames=30,
            nominal_input_hz=30,
            hardware_commands=0,
        )
        self.reset()

    @property
    def cano_root(self):
        return self.clotho_root / "models"

    def reset(self, calibration=None):
        if calibration is not None:
            self.calibrator.set_calibration_120(calibration)
            self.calibrator_adapter.numpy_constants = None
        self.pipeline.reset()

    def step(self, values):
        began = time.perf_counter()
        if self.profile_stages:
            self.last_step_profile = None
            self.calibrator_adapter.elapsed_ms = None
            cpu_started = time.thread_time_ns()
            previous_tracker_result = self.pipeline.gp.tracker.last_result
        with self.torch.inference_mode():
            result = self.pipeline.step(values, self.calibrator_adapter, measured=True)
        if self.profile_stages:
            pipeline_ms = (time.perf_counter() - began) * 1000
            spans = self.pipeline.perf.frame
            row = dict(pipeline_wall_ms=pipeline_ms,
                       pipeline_thread_cpu_ms=(time.thread_time_ns() - cpu_started) / 1e6,
                       calibration_ms=self.calibrator_adapter.elapsed_ms,
                       cano_host_ms=spans.get("cano"),
                       lip_host_ms=spans.get("pose"),
                       gp_adapter_excluding_lip_ms=spans.get("physics"))
            current = self.pipeline.gp.tracker.last_result
            if current is not None and current is not previous_tracker_result:
                diagnostics = current["diagnostics"]
                row.update(gp_tracker_ms=diagnostics["gp_ms"],
                           native_physics_ms=diagnostics["physics_ms"],
                           physics_updates=diagnostics["physics_updates"],
                           gp_status=diagnostics["status"])
            self.last_step_profile = row
        if result is None:
            return None
        pose, tran = result
        pose = pose.detach().cpu().numpy().astype(np.float32)
        native = self.pipeline.gp.last_result
        pose = native["pose"].detach().cpu().numpy().astype(np.float32)
        tran = native["translation"].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(pose).all() or not np.isfinite(tran).all():
            raise ValueError("non-finite S1 GPU output")
        return dict(pose_mat=pose, root_translation_m=tran,
                    inference_ms=(time.perf_counter() - began) * 1000)

    def close(self):
        try:
            self.pipeline.close()
        finally:
            self.cano_session.close()
