#!/usr/bin/env python3
"""Recompute saved T11 measurements without importing the inference driver."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(record):
    path = Path(record["path"])
    if path.stat().st_size != record["bytes"] or fingerprint(path) != record["sha256"]:
        raise ValueError(f"Changed artifact: {path}")


def fk(pose, rest):
    # Homogeneous transforms independently check the driver's vector recursion.
    local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    world = np.tile(np.eye(4), (len(pose), 24, 1, 1))
    for joint, parent in enumerate(PARENTS):
        transform = np.tile(np.eye(4), (len(pose), 1, 1))
        transform[:, :3, :3] = local[:, joint]
        if parent < 0:
            world[:, joint] = transform
        else:
            transform[:, :3, 3] = rest[joint - 1]
            world[:, joint] = world[:, parent] @ transform
    return world[:, :, :3, 3]


def close(actual, expected, name, tolerance=1e-9):
    if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
        raise ValueError(f"Recomputed measurement differs: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--prior-onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    old = json.loads(Path(args.prior_onnx).read_text())
    for record in receipt["inputs"].values():
        verify_file(record)
    if receipt["strict_load_missing"] or receipt["strict_load_unexpected"]:
        raise ValueError("Strict loading failed")
    if receipt["device"] != "cpu" or receipt["training_started"] or receipt["dynamics_step_executed"]:
        raise ValueError("Execution scope changed")
    rows = []
    for case in receipt["sequences"]:
        verify_file(case["artifact"])
        prior = next(item for item in old["sequences"] if item["sequence"] == case["sequence"])
        verify_file(prior["artifact"])
        for source in prior["source_entries"]:
            counterpart = next(item for item in case["source_entries"] if item["entry"] == source["entry"])
            if counterpart["sha256"] != source["sha256"]:
                raise ValueError("Prior ONNX comparison does not use the same tensors")
        with np.load(case["artifact"]["path"], allow_pickle=False) as saved:
            index = saved["frame_index"]
            close(index, np.arange(29, receipt["source_frames"][1]), "frame alignment", 0)
            score = index >= 59
            label = saved["label_pose"]
            rest = saved["rest_bone_vectors"]
            label_joints = fk(label, rest)
            close(label_joints, saved["label_rootrelative_joints"], "label FK")
            arms = {}
            for name in ("tanh", "relu"):
                pred = saved[name + "_pose"]
                positions = fk(pred, rest)
                close(positions, saved[name + "_rootrelative_joints"], name + " FK")
                errors = np.linalg.norm(positions[score] - label_joints[score], axis=-1)
                expected = case["arms"][name]
                for metric, value in (("rootrelative_mpjpe_m", errors.mean()),
                                      ("rootrelative_p95_m", np.quantile(errors, .95)),
                                      ("wrist_mean_m", errors[:, [20, 21]].mean()),
                                      ("ankle_mean_m", errors[:, [7, 8]].mean())):
                    close(value, expected[metric], name + " " + metric)
                rp = Rotation.from_rotvec(pred[score].reshape(-1, 3)).as_matrix()
                rt = Rotation.from_rotvec(label[score].reshape(-1, 3)).as_matrix()
                relative = rp @ rt.transpose(0, 2, 1)
                angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
                close(angles.mean(), expected["local_rotation_mean_rad"], name + " angle")
                if expected["reset_first_frame_maxabs"] > 1e-6:
                    raise ValueError("LIP state reset failed")
                arms[name] = {k: expected[k] for k in ("rootrelative_mpjpe_m", "wrist_mean_m", "ankle_mean_m",
                              "local_rotation_mean_rad", "rootrelative_acceleration_error_m_s2")}
            with np.load(prior["artifact"]["path"], allow_pickle=False) as previous:
                close(previous["label_pose"][index], label, "ONNX common label")
                close(previous["rest_bone_vectors"], rest, "ONNX common skeleton")
                onnx_joints = fk(previous["pred_pose"][index], rest)
                error = np.linalg.norm(onnx_joints[score] - label_joints[score], axis=-1)
                arms["prior_onnx_matched_frames"] = dict(rootrelative_mpjpe_m=float(error.mean()),
                    wrist_mean_m=float(error[:, [20, 21]].mean()), ankle_mean_m=float(error[:, [7, 8]].mean()))
            if case["denoiser_reset_first_output_maxabs"] > 1e-6:
                raise ValueError("Canonicalizer reset failed")
            rows.append(dict(sequence=case["sequence"], scored_frames=int(score.sum()), arms=arms,
                tanh_vs_relu_mpjpe_reduction=1 - arms["tanh"]["rootrelative_mpjpe_m"] / arms["relu"]["rootrelative_mpjpe_m"],
                denoiser_median_ms=case["denoiser_forward_median_ms"], denoiser_p95_ms=case["denoiser_forward_p95_ms"],
                lip_median_ms=case["arms"]["tanh"]["forward_median_ms"]))
    report = dict(verified=True, receipt_sha256=fingerprint(args.receipt), verifier_sha256=fingerprint(__file__),
        inputs_rehashed=True, homogeneous_fk_recomputed=True, rotation_trace_recomputed=True,
        common_prior_onnx_inputs_verified=True, scored_source_frames=receipt["scored_source_frames"],
        comparisons_are_development_only=True, independent_test=False, rows=rows)
    with Path(args.output).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
