#!/usr/bin/env python3
"""TEST_ONLY CLOTHO IMU -> installed FGP canonicalizer/LIP/velocity -> SMPL."""

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from inspect_egolocate import SMPL_PARENTS


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def read_sequence(archive, sequence):
    prefix = f"train_released/{sequence}/"
    arrays, entries = {}, []
    with zipfile.ZipFile(archive) as source:
        for name in ("acc.pt", "rot.pt", "device2bone.pt", "pose.pt", "joint.pt"):
            raw = source.read(prefix + name)
            tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"Expected finite tensor: {name}")
            arrays[name] = tensor.float()
            entries.append({"entry": prefix + name, "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "shape": list(tensor.shape)})
        with source.open(prefix + "sensor_data.txt") as stream:
            raw_rows = sum(1 for line in stream if line.strip())
    acc, rot = arrays["acc.pt"].squeeze(-1), arrays["rot.pt"]
    if (acc.ndim != 3 or acc.shape[1:] != (11, 3)
            or rot.shape != acc.shape[:2] + (3, 3)):
        raise ValueError("Expected the released 11-IMU acc/rot tensor contract")
    if any(arrays[name].shape != (len(acc), 24, 3) for name in ("pose.pt", "joint.pt")):
        raise ValueError("Released comparison labels are not aligned with IMU tensors")
    if arrays["device2bone.pt"].shape != (11, 3, 3):
        raise ValueError("Unexpected recorded calibration shape")
    ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    return acc[:, ids].contiguous(), rot[:, ids].contiguous(), arrays, entries, raw_rows


def canonicalize(model, acc, rot, device, batch_size, project, to_r6d, from_r6d):
    """Evaluate causal 30-frame contexts; retain only each context's last frame."""
    window = 30
    raw_rot = rot.clone()
    orthogonality = torch.linalg.matrix_norm(rot.transpose(-1, -2) @ rot - torch.eye(3))
    determinant = torch.linalg.det(rot)
    if (orthogonality.max() > 0.05 or (determinant <= 0).any()
            or (determinant - 1).abs().max() > 0.05):
        raise ValueError("IMU rotations exceed the explicit 0.05 repair budget")
    rot = project(rot)
    checks = {"source_max_orthogonality_error": orthogonality.max().item(),
              "source_max_determinant_error": (determinant - 1).abs().max().item(),
              "max_matrix_repair_frobenius_change": torch.linalg.matrix_norm(rot - raw_rot).max().item(),
              "repair_budget": 0.05}
    features = torch.cat([acc / 30, to_r6d(rot)], dim=-1)
    clean = features.clone()
    contexts = features.unfold(0, window, 1).permute(0, 3, 1, 2)
    selected = sorted(set([0, len(contexts) // 2, len(contexts) - 1]))
    value = contexts[selected].to(device)
    def infer(value):
        ids = torch.arange(10, device=device)[None].expand(len(value), -1)
        mask = torch.zeros(len(value), window * 10, device=device, dtype=torch.bool)
        return model(x_t_9=value, x_n_9=value, vis_ids=ids, mem_pad_mask=mask)
    batched = infer(value)
    individual = torch.cat([infer(row[None]) for row in value])
    checks["batch_vs_individual_context_indices"] = selected
    checks["batch_vs_individual_max_abs_difference"] = (batched - individual).abs().max().item()
    if not torch.allclose(batched, individual, atol=1e-4, rtol=1e-3):
        raise ValueError("Canonicalizer batch differs from independent causal windows")
    del batched, individual
    for start in range(0, len(contexts), batch_size):
        value = contexts[start:start + batch_size].to(device)
        residual = infer(value)
        clean[start + window - 1:start + window - 1 + len(value)] = (value + residual)[:, -1].cpu()
        if start % (batch_size * 20) == 0:
            print(f"canonicalization {start}/{len(contexts)}", flush=True)
    result_acc, result_rot = clean[..., :3] * 30, from_r6d(clean[..., 3:])
    # Match the installed causal_ema setting, with a fresh state at the first full context.
    for index in range(window, len(clean)):
        result_acc[index] = 0.7 * result_acc[index - 1] + 0.3 * result_acc[index]
        result_rot[index] = project(0.7 * result_rot[index - 1] + 0.3 * result_rot[index])
    return result_acc[window - 1:], result_rot[window - 1:], checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--fgp-root", type=Path, required=True)
    parser.add_argument("--velocity-root", type=Path, required=True)
    parser.add_argument("--velocity-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; preserve prior results")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    args.output_dir.mkdir(parents=True)
    for path in (args.fgp_root, args.velocity_root, args.velocity_root / "scripts"):
        sys.path.insert(0, str(path.resolve()))
    from neural_backbone import IMUCanonicalization
    from Aplus.tools.smpl_light import SMPLight
    from evaluate_deployment_orientation import build_lip, predict_lip_pose_rotation
    from fgp_velocity.features import pack_lip_features
    from fgp_velocity.rotations import project_to_rotation_matrix, matrix_to_r6d, r6d_to_matrix
    from fgp_velocity.runtime import FGPVelocityRuntime

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    print(f"Loading {args.sequence} on {device}", flush=True)
    acc, rot, comparison, entries, raw_rows = read_sequence(args.archive, args.sequence)
    print(f"Loaded {len(acc)} aligned IMU frames", flush=True)
    if len(acc) <= 60:
        raise ValueError("Sequence does not cover both warmup periods")
    cano_path = args.fgp_root / "checkpoints/canonicalization/U100hL7.07h_residual.pth"
    lip_path = args.fgp_root / "checkpoints/LIP/LIP_10_real_denoised_7.07_10imu_10.pth"
    model = IMUCanonicalization(
        pretrained_ae=None, cin_state=9, cin_cond=9, cout=9, depth=6,
        nhead=4, ffn=512, dropout=0.1, max_T=512, t_embed_dim=256,
        enc_d_model=384, enc_layers=4, enc_nhead=8).to(device).eval()
    payload = torch.load(cano_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("model", payload), strict=True)
    print("Canonicalizer checkpoint loaded", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        clean_acc, clean_rot, canonical_checks = canonicalize(
            model, acc, rot, device, args.batch_size,
            project_to_rotation_matrix, matrix_to_r6d, r6d_to_matrix)
        print(f"Canonicalization complete: {canonical_checks}", flush=True)
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        features, _ = pack_lip_features(clean_acc, clean_rot)
        lip = build_lip(device, lip_path)
        pose = predict_lip_pose_rotation(lip, features, device=device, chunk_len=512)
        print("LIP pose reconstruction complete", flush=True)
        body = SMPLight()
        _, local_joints = body.forward_kinematics(pose, calc_joint=True)
        velocity = FGPVelocityRuntime(args.velocity_checkpoint, device=device, fps=30, warmup_frames=30)
        # Both neural stages run without using the archive's pose/joint comparison labels.
        outputs = []
        velocity.reset()
        for frame in range(len(pose)):
            output = velocity.step(clean_acc[frame], clean_rot[frame], pose[frame],
                                   local_joint_position=local_joints[frame])
            outputs.append(output.as_dict())
            if frame % 300 == 0:
                print(f"velocity {frame}/{len(pose)}", flush=True)
    fields = {name: torch.stack([row[name] for row in outputs]).numpy()
              for name in outputs[0]}
    np.savez_compressed(args.output_dir / "fgp_trace.npz", canonical_acc=clean_acc.numpy(),
                        canonical_rot=clean_rot.numpy(), local_pose=pose.numpy(),
                        frame_index=np.arange(29, len(acc)),
                        **fields, usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
    # Drop the first 29 raw contexts and 30 cold velocity outputs from Q1 input.
    first_frame = 59
    local = pose[30:].numpy()
    translation = fields["root_translation"][30:]
    joints = local_joints[30:].numpy() + translation[:, None]
    axis_angle = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 72)
    bundle = args.output_dir / "body_only.npz"
    archive_id = identity(args.archive)
    np.savez_compressed(bundle, pose_matrix=local.reshape(-1, 24, 9), axis_angle=axis_angle,
                        trans=translation, joints=joints, parents=SMPL_PARENTS,
                        source_time_s=np.arange(len(local), dtype=float) / 30,
                        frame_index=np.arange(first_frame, len(acc)),
                        source_sha256=np.asarray(archive_id["sha256"]),
                        usage=np.asarray("TEST_ONLY"), training_allowed=np.asarray(False))
    input_id = identity(bundle)
    inspection = {"usage": "TEST_ONLY", "training_allowed": False,
                  "source_kind": "REINFERRED_CLOTHO_IMU_FGP_LIP_EXISTING_VELOCITY_HEAD",
                  "reference_source_kind": "TEST_ONLY_REINFERRED_CLOTHO_FGP_LIP_ESTIMATED_ROOT",
                  "source_sequence": args.sequence, "archive": archive_id,
                  "entries": entries, "bundle": input_id,
                  "timestamps": "PUBLISHED_ALIGNED_PT_FIXED_30_HZ_NOT_RAW_TEXT_TIMESTAMPS",
                  "root_translation_kind": "LEARNED_EXISTING_VELOCITY_HEAD_NOT_MEASURED_GROUND_TRUTH"}
    (args.output_dir / "inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    truth_rotation = Rotation.from_rotvec(comparison["pose.pt"][29:].numpy().reshape(-1, 3))
    angular = (Rotation.from_matrix(pose.numpy().reshape(-1, 3, 3)) * truth_rotation.inv()).magnitude()
    relative_pred = local_joints.numpy() - local_joints[:, 0:1].numpy()
    truth_joints = comparison["joint.pt"].numpy()
    relative_truth = truth_joints - truth_joints[:, 0:1]
    position_error = np.linalg.norm(relative_pred[30:] - relative_truth[first_frame:], axis=-1)
    root_truth = truth_joints[first_frame:, 0] - truth_joints[first_frame, 0]
    root_estimate = translation - translation[0]
    dependencies = [Path(__file__).resolve(), args.fgp_root / "neural_backbone.py",
                    args.fgp_root / "Aplus/tools/smpl_light.py",
                    args.fgp_root / "PoseReconstruction/LIP_model.py",
                    args.velocity_root / "scripts/evaluate_deployment_orientation.py",
                    args.velocity_root / "fgp_velocity/runtime.py",
                    args.velocity_root / "fgp_velocity/features.py",
                    args.velocity_root / "fgp_velocity/rotations.py",
                    args.velocity_root / "fgp_velocity/model.py"]
    report = {
        **inspection, "frames_input": len(acc), "frames_exported": len(local),
        "raw_sensor_text_rows": raw_rows, "raw_text_alignment_verified": False,
        "input_semantics": "PUBLISHED_PREPROCESSED_CLOTH_IMU_ACC_ROT_11_CHANNELS",
        "channel_selection": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
        "calibration": "RELEASED_TENSOR_CONTRACT_NO_RECALIBRATION_FROM_UNALIGNED_SENSOR_TEXT",
        "canonicalization": "30_FRAME_CAUSAL_LAST_OUTPUT_RESIDUAL_EULER_1_EMA_0_7_BATCHED",
        "canonicalizer_checks": canonical_checks,
        "lip_reset_at_first_complete_canonical_context": True,
        "velocity_valid_all_exported": bool(fields["velocity_valid"][30:].all()),
        "removed_warmup_input_frames": first_frame,
        "supplied_pose_joint_used_for_inference": False,
        "label_comparison_is_independent_benchmark": False,
        "existing_velocity_training_overlap_possible": True,
        "supplied_label_local_rotation_error_p95_rad": float(np.quantile(angular.reshape(-1, 24)[30:], .95)),
        "supplied_label_rootrelative_joint_error_p95_m": float(np.quantile(position_error, .95)),
        "supplied_label_relative_root_trajectory_rmse_m": float(np.sqrt(np.mean(np.sum((root_estimate - root_truth) ** 2, axis=1)))),
        "estimated_root_range_m": np.ptp(translation, axis=0).tolist(),
        "supplied_label_root_range_m": np.ptp(root_truth, axis=0).tolist(),
        "estimated_root_path_length_m": float(np.linalg.norm(np.diff(translation, axis=0), axis=-1).sum()),
        "wall_s": time.perf_counter() - started,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy")},
        "device": str(device), "batch_size": args.batch_size,
        "source_files": [identity(path) for path in dependencies],
        "weights": [identity(path) for path in (cano_path, lip_path, args.velocity_checkpoint)],
        "training_started": False, "hardware_control": False, "baidu_backup_status": "LOCAL_ONLY",
        "artifacts": [identity(path) for path in sorted(args.output_dir.iterdir())],
    }
    if not all(np.isfinite(array).all() for array in (local, translation, joints, position_error)):
        raise ValueError("Non-finite pipeline output")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
