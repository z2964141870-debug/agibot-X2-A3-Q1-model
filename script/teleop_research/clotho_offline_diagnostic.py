#!/usr/bin/env python3
"""Bounded CPU diagnosis of the user-supplied CLOTHO release, without its UI entry.

Third-party source and tensors stay outside Git. Only selected definitions from
the fingerprinted online entry are evaluated; its top-level code is never run.
"""

import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
import zipfile


SOURCE_SHA = "974bd8d94c5384cc47b7021d0e1f13092f34948d480a933603c8aeb6db57977b"
CANO_SHA = "b613eb8226b26b9dc82904c22f4aec78773992503337c3cbde674c998f61ce12"
LIP_SHA = "efc4bb8583bf053890d9ae754e97415c8658dba237893982aa51b9541edbb9a2"
DATA_SHA = "8015e969dc2ef9629e3a4e02d16d2846449d7db2ebd2eb94ca063c814358f952"
SEQUENCES = (
    "G_0113_zj_train/183926_zj_walk",
    "G_0109_yy_train/193310_yy_oneLegStand",
    "G_0113_zj_train/192333_zj_warmUp",
)
SLOTS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
DEFINITIONS = {
    "load_flowdit", "flow_sampler_euler", "RealtimeIMUDenoiser",
    "pack_lip_imu_frame", "RealtimeLIPPredictor", "build_lip_model",
}


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require_identity(path, expected):
    result = identity(path)
    if result["sha256"] != expected:
        raise ValueError(f"Fingerprint mismatch: {path}")
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def prepare(args):
    archive = require_identity(args.archive, SOURCE_SHA)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    with zipfile.ZipFile(args.archive) as package:
        for entry in package.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive member")
            target = destination.joinpath(*name.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = package.read(entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
            files.append({"entry": entry.filename, "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
    write_json(destination / "source_manifest.json", {"archive": archive, "files": files,
               "distribution": "user_supplied_reference_only_not_redistributed"})
    print(json.dumps({"prepared": str(destination), "files": len(files), "archive": archive}), flush=True)


def runtime(source):
    # CPU is an explicit contract; no CUDA discovery or shared GPU allocation.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261022)
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
        raise RuntimeError("Vendor load_flowdit requires the weights_only=True default (PyTorch >= 2.6)")

    def deny_external(event, unused_args):
        if event in {"socket.connect", "socket.bind", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise RuntimeError(f"Offline execution denied: {event}")

    sys.addaudithook(deny_external)
    sys.path.insert(0, str(source))
    import articulate as art
    from Aplus.models import EasyLSTM
    from PoseReconstruction.LIP_model import BiPoser
    from neural_backbone import IMUCanonicalization

    entry = source / "online_fgp+LIP.py"
    tree = ast.parse(entry.read_text(encoding="utf-8-sig"), filename=str(entry))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in DEFINITIONS]
    if {node.name for node in nodes} != DEFINITIONS:
        raise ValueError("Online definition contract changed")
    namespace = dict(torch=torch, art=art, EasyLSTM=EasyLSTM, BiPoser=BiPoser,
                     IMUCanonicalization=IMUCanonicalization, deque=deque,
                     device=torch.device("cpu"), LIP_IMU_NUM=10, LIP_ACC_SCALE=30.0, LIP_NUM_JOINTS=24)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(entry), "exec"), namespace)
    return np, torch, Rotation, namespace


def load_sequence(package, sequence, frames, torch):
    tensors, records = {}, []
    for name in ("acc.pt", "rot.pt", "pose.pt", "joint.pt", "device2bone.pt"):
        member = f"train_released/{sequence}/{name}"
        payload = package.read(member)
        value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid released tensor: {member}")
        tensors[name] = value.float()
        records.append({"entry": member, "bytes": len(payload), "shape": list(value.shape),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    count = len(tensors["acc.pt"])
    shapes = {"acc.pt": (count, 11, 3, 1), "rot.pt": (count, 11, 3, 3),
              "pose.pt": (count, 24, 3), "joint.pt": (count, 24, 3)}
    if count < frames or any(tuple(tensors[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError(f"Released tensor shape mismatch: {sequence}")
    acc = tensors["acc.pt"][:frames, SLOTS, :, 0].contiguous()
    rot = tensors["rot.pt"][:frames, SLOTS].contiguous()
    eye = torch.eye(3)
    orth = float(torch.linalg.matrix_norm(rot.transpose(-1, -2) @ rot - eye).max())
    if orth > 0.05 or float(torch.linalg.det(rot).min()) <= 0:
        raise ValueError("IMU rotation input is outside the rotation contract")
    return acc, rot, tensors["pose.pt"][:frames].numpy(), tensors["joint.pt"][:frames].numpy(), records, orth


def globals_from_pose(pose, Rotation, np):
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    world = np.empty_like(local)
    for joint, parent in enumerate(PARENTS):
        world[:, joint] = local[:, joint] if parent < 0 else world[:, parent] @ local[:, joint]
    return world


def joints_from_globals(world, rest, np):
    joints = np.zeros((len(world), 24, 3))
    for joint in range(1, 24):
        joints[:, joint] = joints[:, PARENTS[joint]] + np.einsum(
            "tij,j->ti", world[:, PARENTS[joint]], rest[joint - 1])
    return joints


def summarize_pose(prediction, label, rest, score_start, Rotation, np):
    pred_global = globals_from_pose(prediction, Rotation, np)
    label_global = globals_from_pose(label, Rotation, np)
    pred_joints = joints_from_globals(pred_global, rest, np)
    label_joints = joints_from_globals(label_global, rest, np)
    angles = (Rotation.from_rotvec(prediction[score_start:].reshape(-1, 3)) *
              Rotation.from_rotvec(label[score_start:].reshape(-1, 3)).inv()).magnitude().reshape(-1, 24)
    error = np.linalg.norm(pred_joints[score_start:] - label_joints[score_start:], axis=-1)
    # Root-orientation substitution is a diagnostic oracle, never an inference input.
    oracle_pose = prediction.copy()
    oracle_pose[:, 0] = label[:, 0]
    oracle_joints = joints_from_globals(globals_from_pose(oracle_pose, Rotation, np), rest, np)
    oracle_error = np.linalg.norm(oracle_joints[score_start:] - label_joints[score_start:], axis=-1)
    acceleration_error = np.diff(pred_joints, n=2, axis=0) - np.diff(label_joints, n=2, axis=0)
    acceleration_error = np.linalg.norm(acceleration_error[max(0, score_start - 2):], axis=-1) * 30**2
    metrics = dict(local_rotation_mean_rad=float(angles.mean()), local_rotation_p95_rad=float(np.quantile(angles, .95)),
                   local_rotation_per_joint_mean_rad=angles.mean(0).tolist(),
                   rootrelative_mpjpe_m=float(error.mean()), rootrelative_p95_m=float(np.quantile(error, .95)),
                   wrist_mean_m=float(error[:, [20, 21]].mean()), ankle_mean_m=float(error[:, [7, 8]].mean()),
                   torso_mean_m=float(error[:, [3, 6, 9]].mean()), per_joint_mean_m=error.mean(0).tolist(),
                   oracle_label_root_orientation_mpjpe_m=float(oracle_error.mean()),
                   rootrelative_acceleration_error_m_s2=float(acceleration_error.mean()),
                   world_root_metrics=None)
    return metrics, pred_joints, label_joints


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = Path(args.source).resolve()
    source_manifest = json.loads((source.parent / "source_manifest.json").read_text())
    if source_manifest["archive"]["sha256"] != SOURCE_SHA:
        raise ValueError("Source package identity mismatch")
    for record in source_manifest["files"]:
        require_identity(source.parent / record["entry"], record["sha256"])
    inputs = dict(source_archive=source_manifest["archive"], data=require_identity(args.data, DATA_SHA),
                  canonicalizer=require_identity(args.canonicalizer, CANO_SHA), lip=require_identity(args.lip, LIP_SHA),
                  driver=identity(__file__))
    write_json(out / "admission.json", {"inputs": inputs, "status": "ADMITTED_NOT_COMPLETED",
               "frames": args.frames, "sequences": list(SEQUENCES), "source_files_verified": len(source_manifest["files"]),
               "utc": datetime.now(timezone.utc).isoformat()})
    np, torch, Rotation, definitions = runtime(source)
    lip = definitions["build_lip_model"]()
    state = torch.load(args.lip, map_location="cpu", weights_only=True)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    loaded = lip.load_state_dict(state, strict=True)
    lip.eval()
    if any(not torch.isfinite(v).all() for v in lip.state_dict().values()):
        raise ValueError("Nonfinite LIP weights")
    denoiser = definitions["RealtimeIMUDenoiser"](
        ckpt_path=args.canonicalizer, window_size=30, chunk_size=4, stride=2, steps=1,
        interval=1, output_mode="causal_ema", ema_alpha=.08, acc_scale=30., use_fp16=False,
        every_frame=True, torch_compile=False)
    if any(not torch.isfinite(v).all() for v in denoiser.model.state_dict().values()):
        raise ValueError("Nonfinite canonicalizer weights")
    print("STRICT_LOAD_OK CPU canonicalizer + LIP", flush=True)
    rows = []
    with zipfile.ZipFile(args.data) as package, torch.inference_mode():
        for sequence in SEQUENCES:
            acc, rot, label, joints, records, orth = load_sequence(package, sequence, args.frames, torch)
            denoiser.reset()
            den_acc, den_rot, den_ms, indexes = [], [], [], []
            for frame in range(args.frames):
                start = time.perf_counter()
                a, r = denoiser.step(acc[frame], rot[frame])
                if frame < 29:
                    continue
                if not torch.isfinite(a).all() or not torch.isfinite(r).all():
                    raise ValueError("Nonfinite canonicalizer output")
                den_ms.append((time.perf_counter() - start) * 1000)
                den_acc.append(a.clone())
                den_rot.append(r.clone())
                indexes.append(frame)
                if frame % 30 == 29:
                    print(f"DENOISE {sequence} {frame + 1}/{args.frames}", flush=True)
            den_acc, den_rot = torch.stack(den_acc), torch.stack(den_rot)
            frame_index = np.asarray(indexes)
            truth = label[frame_index]
            truth_world = globals_from_pose(label, Rotation, np)
            offsets = joints[:, 1:] - joints[:, np.asarray(PARENTS[1:])]
            rest = np.median(np.einsum("tbji,tbj->tbi", truth_world[:, np.asarray(PARENTS[1:])], offsets)[:30], axis=0)
            label_fk = joints_from_globals(truth_world, rest, np)
            consistency = np.linalg.norm(label_fk - (joints - joints[:, :1]), axis=-1)
            if float(np.quantile(consistency, .95)) > .005:
                raise ValueError("Label pose/joint common-skeleton consistency failed")
            score_start = 30  # First LIP output is source frame 29; score source frames 59 onward.
            arms, arrays = {}, {}
            for activation in ("tanh", "relu"):
                for net in (lip.net_s1, lip.net_s2):
                    net.act_func = torch.nn.Tanh() if activation == "tanh" else torch.nn.ReLU()
                predictor = definitions["RealtimeLIPPredictor"](lip)
                poses, forward_ms = [], []
                for a, r in zip(den_acc, den_rot):
                    start = time.perf_counter()
                    pose, translation = predictor.step(a, r)
                    forward_ms.append((time.perf_counter() - start) * 1000)
                    if not torch.isfinite(pose).all() or not torch.equal(translation, torch.zeros(3)):
                        raise ValueError("Invalid pose-only prediction")
                    poses.append(pose.numpy().copy())
                matrices = np.stack(poses)
                if np.max(np.linalg.norm(matrices.transpose(0, 1, 3, 2) @ matrices - np.eye(3), axis=(-2, -1))) > .001:
                    raise ValueError("Invalid predicted rotation matrices")
                prediction = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 24, 3)
                metrics, predicted_joints, common_joints = summarize_pose(prediction, truth, rest, score_start, Rotation, np)
                predictor.reset()
                replay, _ = predictor.step(den_acc[0], den_rot[0])
                reset_error = float(np.max(np.abs(replay.numpy() - matrices[0])))
                if reset_error > 1e-6:
                    raise ValueError("Recurrent state reset is not reproducible")
                arms[activation] = {**metrics, "forward_median_ms": float(np.median(forward_ms)),
                                    "reset_first_frame_maxabs": reset_error, "finite": True}
                arrays[f"{activation}_pose"] = prediction
                arrays[f"{activation}_rootrelative_joints"] = predicted_joints
                print("POSE", sequence, activation, json.dumps(metrics), flush=True)
            arrays.update(label_pose=truth, label_rootrelative_joints=common_joints, rest_bone_vectors=rest,
                          frame_index=frame_index, denoised_acc=den_acc.numpy(), denoised_rot=den_rot.numpy(),
                          scored_source_frames=np.asarray([59, args.frames]), training_allowed=np.asarray(False))
            denoiser.reset()
            for frame in range(30):
                reset_acc, reset_rot = denoiser.step(acc[frame], rot[frame])
            denoiser_reset_error = max(float((reset_acc - den_acc[0]).abs().max()),
                                      float((reset_rot - den_rot[0]).abs().max()))
            if denoiser_reset_error > 1e-6:
                raise ValueError("Canonicalizer reset/prefix reproduction failed")
            artifact = out / (sequence.split("/")[-1] + ".npz")
            np.savez_compressed(artifact, **arrays)
            row = dict(sequence=sequence, source_entries=records, source_rotation_orthogonality_max=orth,
                       label_fk_consistency_p95_m=float(np.quantile(consistency, .95)),
                       denoiser_reset_first_output_maxabs=denoiser_reset_error,
                       denoiser_forward_median_ms=float(np.median(den_ms)),
                       denoiser_forward_p95_ms=float(np.quantile(den_ms, .95)),
                       arms=arms, artifact=identity(artifact))
            rows.append(row)
            write_json(out / (sequence.split("/")[-1] + "_receipt.json"), row)
    report = dict(experiment=args.experiment, utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                  host=platform.node(), python=sys.version, torch=torch.__version__, numpy=np.__version__,
                  cpu_threads=2, device="cpu", sequences=rows, source_frames=[0, args.frames],
                  first_output_source_frame=29, scored_source_frames=[59, args.frames], assumed_fps=30,
                  time_alignment="released_PT_same_index_no_future_IMU_and_no_label_input_to_predictor",
                  source_semantics="released_PT_used_as_in_cano_eval; online_raw_calibration_equivalence_unverified",
                  slots=SLOTS, root_imu_after_selection=9, acc_units="released_acc_assumed_m_s2_scale_30",
                  common_skeleton="label_calibrated_first_30_frames_offline_oracle",
                  stage_tran="off_missing_two_weights", display_postprocessing=False,
                  label_independent_test=False, development_only=True, world_root_metrics=None,
                  training_started=False, policy_loaded=False, dynamics_step_executed=False,
                  hardware_connected=False, network_audit_guard=True, strict_load_missing=list(loaded.missing_keys),
                  strict_load_unexpected=list(loaded.unexpected_keys), wall_s=time.perf_counter() - started)
    write_json(out / "receipt.json", report)
    print(json.dumps({"completed": args.experiment, "wall_s": report["wall_s"], "receipt": str(out / "receipt.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--archive", required=True)
    prep.add_argument("--destination", required=True)
    evaluate = commands.add_parser("run")
    for name in ("source", "data", "canonicalizer", "lip", "output", "experiment"):
        evaluate.add_argument("--" + name, required=True)
    evaluate.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    if args.command == "run" and not 90 <= args.frames <= 300:
        parser.error("The development diagnosis is bounded to 90..300 source frames per sequence")
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
