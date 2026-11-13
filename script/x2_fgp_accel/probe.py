"""Replace only CANO execution in the existing recorded-input full-chain probe."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from script.x2_fgp_realtime.assets import file_digest
from .build import require_resources, resources
from .trt_session import TrtSession


LIMITS = {
    "fp32": dict(residual_max_abs=2e-4, residual_relative_l2=1e-3, pose_max_deg=.5, root_max_m=.005),
    "fp16": dict(residual_max_abs=5e-3, residual_relative_l2=.02, pose_max_deg=1., root_max_m=.01),
}
LIMITS["mixed"] = LIMITS["fp16"].copy()
LIMITS["mixed_gemm"] = LIMITS["fp16"].copy()
LIMITS["tf32"] = LIMITS["fp32"].copy()


def differences(pose, root, reference_pose, reference_root):
    if pose.shape != reference_pose.shape or root.shape != reference_root.shape:
        raise ValueError("Full-chain output shape mismatch")
    relative = np.matmul(pose.astype(np.float64), np.swapaxes(reference_pose.astype(np.float64), -1, -2))
    angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    displacement = np.linalg.norm(root-reference_root, axis=-1)
    return dict(pose_max_deg=float(angle.max()), pose_mean_deg=float(angle.mean()),
                root_max_m=float(displacement.max()), root_mean_m=float(displacement.mean()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing results")
    require_resources(resources())
    build = json.loads((args.build_root / "build_receipt.json").read_text())
    engine_path = args.build_root / "cano.plan"
    bundle_root = args.source_root / "models/onnx_s1"
    if (build["status"] != "BUILD_PASSED" or build["onnx_sha256"] != file_digest(bundle_root / "cano.onnx")
            or build["engine_sha256"] != file_digest(engine_path)):
        raise ValueError("Engine/source provenance mismatch")
    reference_report = json.loads((args.reference / "report.json").read_text())
    if (reference_report["bundle"] != "onnx_s1" or not reference_report["native_physics"]
            or reference_report["fixture_sha256"] != file_digest(args.fixture)
            or reference_report["manifest_sha256"] != file_digest(bundle_root / "manifest.json")):
        raise ValueError("Reference identity mismatch")
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment
    prepare_clotho_environment(args.source_root)
    from onnx_inference import OnnxBundle, OnnxFlowModel, cpu_session
    from script.x2_fgp_realtime import probe as baseline
    session = TrtSession(engine_path)
    original_flow = OnnxBundle.flow_model
    argv = sys.argv
    try:
        def accelerated_flow(bundle):
            if bundle.directory.resolve() != bundle_root.resolve() or bundle.manifest.get("cano_output") != "latest":
                raise ValueError("Unexpected CANO bundle")
            return OnnxFlowModel(session, bundle.window_size, latest_only=True)
        OnnxBundle.flow_model = accelerated_flow
        sys.argv = ["recorded_probe", "--source-root", str(args.source_root), "--fixture", str(args.fixture),
                    "--bundle", "onnx_s1", "--threads", "4", "--full", "--output", str(args.output)]
        baseline.main()
    finally:
        OnnxBundle.flow_model = original_flow
        sys.argv = argv
        session.close()
    if len(session.recorded) != 32:
        raise ValueError("Expected exactly 32 CANO calls")
    # CPU comparison follows the timed GPU run; it cannot inflate GPU timings.
    reference = cpu_session(bundle_root / "cano.onnx", threads=4)
    gpu_outputs = np.stack([output for _, output in session.recorded])
    cpu_outputs = np.stack([reference.run(["residual"], feed)[0] for feed, _ in session.recorded])
    if not np.isfinite(cpu_outputs).all():
        raise ValueError("Nonfinite CPU reference")
    delta = gpu_outputs-cpu_outputs
    metrics = dict(residual_max_abs=float(np.abs(delta).max()),
                   residual_relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(cpu_outputs), 1e-12)))
    with np.load(args.output / "outputs.npz", allow_pickle=False) as saved, np.load(args.reference / "outputs.npz", allow_pickle=False) as ref:
        metrics.update(differences(saved["pose"], saved["translation"], ref["pose"], ref["translation"]))
    limits = LIMITS[build["precision"]]
    accepted = all(metrics[k] <= value for k, value in limits.items())
    np.savez_compressed(args.output / "cano_comparison.npz", gpu=gpu_outputs, cpu=cpu_outputs,
        x_t=np.stack([feed["x_t"] for feed, _ in session.recorded]),
        x_cond=np.stack([feed["x_cond"] for feed, _ in session.recorded]))
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text())
    report.update(acceleration="CANO TensorRT only; all other author stages unchanged",
        precision=build["precision"], tensorrt_version=session.version,
        engine_sha256=build["engine_sha256"], onnx_sha256=build["onnx_sha256"],
        agreement_status="PASSED" if accepted else "FAILED", agreement_metrics=metrics,
        agreement_limits=limits, screening_limits_are_hardware_safety_certification=False,
        reference_report_sha256=file_digest(args.reference / "report.json"),
        comparison_after_timed_run=True, input_recording_included_in_timed_run=True,
        resource_after=resources())
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
