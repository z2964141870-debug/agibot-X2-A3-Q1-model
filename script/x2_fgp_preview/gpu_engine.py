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
    def __init__(self, calibrator, torch):
        self.calibrator = calibrator
        self.torch = torch

    def calibrate(self, raw, acc_scale=30):
        acc, rot = self.calibrator.calibrate_ordered_10x7(np.asarray(raw, dtype=np.float32))
        return self.torch.cat((acc.flatten() / float(acc_scale), rot.flatten())).tolist()


class S1GpuFgp:
    """Expose the preview engine's ``reset/step/close`` contract for S1 GPU."""

    def __init__(self, clotho_root: Path, cano_engine: Path, gp_engine: Path | None = None, *, cuda_graph=True):
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
        self.cano_session = (CudaGraphSession if cuda_graph else TrtSession)(self.cano_engine)



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
        self.calibrator_adapter = _CalibratorAdapter(self.calibrator, torch)
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
        self.pipeline.reset()

    def step(self, values):
        began = time.perf_counter()
        with self.torch.inference_mode():
            result = self.pipeline.step(values, self.calibrator_adapter, measured=True)
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
