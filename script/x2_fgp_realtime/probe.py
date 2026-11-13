"""Bounded recorded-input author neural/full-physics probe; no BLE or SDK."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np

from .assets import file_digest


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(count=len(a), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                max=float(a.max()), over_33ms=int((a > 1000 / 30).sum()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--bundle", choices=("onnx_s1", "onnx_cano_s_ep120_fast"), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    args.output.mkdir(parents=True)
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator
    prepare_clotho_environment(args.source_root)
    import torch
    import articulate as art
    from Aplus.tools.smpl_light import SMPLPose
    from Utils.config import bulid_rot
    from onnx_inference import OnnxBundle
    from stage_tran_gp.onnx_network import OnnxGPNetwork
    from stage_tran_gp.features import PoseFeatures, angular_velocity_world, valid_rotations

    torch.set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("author_realtime", args.source_root / "online_fgp+LIP_realtime_onnx.py")
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    fixture = np.load(args.fixture, allow_pickle=False)
    raw = fixture["raw"]
    if raw.shape != (61, 10, 7) or not np.isfinite(raw).all():
        raise ValueError("Expected the existing 61-frame recorded IMU fixture")
    cal = ClothoCalibrator(torch, art, bulid_rot, SMPLPose.t_pose_ori)
    cal.set_calibration_120(fixture["calibration"])
    directory = args.source_root / "models" / args.bundle
    began = time.perf_counter()
    pipeline = None
    if args.full:
        config = source.parse_args(["--onnx-dir", str(directory), "--onnx-threads", str(args.threads)])
        pipeline = source.RealtimePipeline(config)
    else:
        bundle = OnnxBundle(directory, threads=args.threads)
        denoise = source.RealtimeIMUDenoiser(None, window_size=30, steps=1,
            every_frame=True, ema_alpha=.08, onnx_model=bundle.flow_model())
        lip = source.RealtimeLIPPredictor(runner=bundle.lip_runner())
        gp = OnnxGPNetwork(directory, threads=1)
        features = PoseFeatures(args.source_root / "models/SMPL_male.pkl")
    initialize_s = time.perf_counter() - began
    rows, poses, translations, gp_outputs = [], [], [], []
    previous_rot = None
    try:
        with torch.inference_mode():
            for index, item in enumerate(raw):
                start = time.perf_counter()
                acc, rot = cal.calibrate_ordered_10x7(item)
                calibrated_at = time.perf_counter()
                if args.full:
                    calibrated = torch.cat((acc.flatten() / 30, rot.flatten())).tolist()
                    result = pipeline.step(calibrated, SimpleNamespace(calibrate=lambda raw, acc_scale: raw))
                    if result is None:
                        continue
                    native = pipeline.gp.last_result
                    pose, tran = native["pose"], native["translation"]
                    translations.append(tran.numpy().copy())
                    extra = dict(status=native["diagnostics"]["status"],
                                 physics_ms=native["diagnostics"]["physics_ms"],
                                 author_exclusive_stage_ms=dict(pipeline.perf.frame))
                else:
                    acc, rot = denoise.step(acc, rot)
                    cano_at = time.perf_counter()
                    if len(denoise.acc_buffer) < 30:
                        continue
                    pose, _ = lip.step(acc, rot)
                    lip_at = time.perf_counter()
                    gyro = np.zeros((10, 3)) if previous_rot is None else angular_velocity_world(previous_rot, rot.numpy(), 1/30)
                    previous_rot = rot.numpy().copy()
                    feature, _, _ = features(pose, acc, gyro)
                    feature_at = time.perf_counter()
                    output = gp.step(feature)
                    gp_outputs.append(output.numpy().copy())
                    extra = dict(cano_ms=(cano_at-calibrated_at)*1000,
                                 lip_ms=(lip_at-cano_at)*1000, features_ms=(feature_at-lip_at)*1000,
                                 gp_ms=(time.perf_counter()-feature_at)*1000)
                if not valid_rotations(pose.numpy(), 24):
                    raise ValueError("Nonfinite or nonrotation pose output")
                if args.full and (not torch.isfinite(tran).all()):
                    raise ValueError("Nonfinite native translation")
                poses.append(pose.numpy().copy())
                rows.append(dict(frame=index, total_ms=(time.perf_counter()-start)*1000,
                                 calibration_ms=(calibrated_at-start)*1000, **extra))
    finally:
        if pipeline is not None:
            pipeline.close()
    if len(rows) != 32:
        raise ValueError("Expected exactly 32 outputs after 29 history frames")
    report = dict(status="RECORDED_FULL_PIPELINE_PASSED" if args.full else "RECORDED_NEURAL_STAGES_PASSED",
        machine=platform.machine(), bundle=args.bundle, threads=args.threads,
        torch_version=torch.__version__, fixture_sha256=file_digest(args.fixture),
        manifest_sha256=file_digest(directory / "manifest.json"), initialize_s=initialize_s,
        input_frames=61, history_only_frames=29, output_frames=32,
        timing_all_ms=stats([r["total_ms"] for r in rows]),
        timing_after_first2_ms=stats([r["total_ms"] for r in rows[2:]]), rows=rows,
        native_physics=args.full, ble_connections=0, feedback_subscriptions=0,
        hardware_commands=0, publish_allowed=False, training_updates=0,
        accuracy_ground_truth_evaluated=False, wall_clock_online_tested=False,
        limitation="Short recorded-input diagnostic; no BLE, browser, network or hardware latency. Neural mode has no root output and does not validate physics. Full mode uses native translation without Unity display offset.")
    np.savez_compressed(args.output / "outputs.npz", pose=np.stack(poses),
                        translation=np.asarray(translations), gp=np.asarray(gp_outputs))
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
