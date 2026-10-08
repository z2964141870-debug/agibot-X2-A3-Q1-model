#!/usr/bin/env python3
"""Stream recorded JSONL into a compact, lossless body-only TEST_ONLY bundle."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SMPL_PARENTS = np.asarray([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                           13, 14, 16, 17, 18, 19, 20, 21])
BODY_FIELDS = {"pose_matrix": (24, 9), "axis_angle": (72,), "trans": (3,), "joints": (24, 3)}


def body_record(record):
    payload = record["payload"]
    if payload.get("type") != "pose" or payload.get("source") != "egolocate_fgp":
        raise ValueError("Expected an egolocate_fgp pose payload")
    if not np.array_equal(payload["parents"], SMPL_PARENTS):
        raise ValueError("Unexpected SMPL parent tree")
    values = {}
    for name, shape in BODY_FIELDS.items():
        value = np.asarray(payload[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} shape or non-finite value")
        values[name] = value
    for name, value in (("host_monotonic_s", record["host_monotonic_s"]),
                        ("elapsed_s", record["elapsed_s"]), ("source_time_s", payload["time"]),
                        ("reported_fps", payload["fps"])):
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {name}")
        values[name] = float(value)
    index = payload["frame_index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("frame_index must be an integer")
    values["frame_index"] = index
    fused = np.asarray(payload["fused_pose_matrix"], dtype=np.float64)
    if fused.shape != (24, 9) or not np.isfinite(fused).all():
        raise ValueError("Invalid fused_pose_matrix")
    values["fused_delta"] = float(np.abs(fused - values["pose_matrix"]).max())
    return values


def global_rotations(local):
    local = np.asarray(local).reshape(-1, 24, 3, 3)
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    for joint, parent in enumerate(SMPL_PARENTS[1:], 1):
        result[:, joint] = result[:, parent] @ local[:, joint]
    return result


def interval_summary(times):
    delta = np.diff(times)
    positive = delta[delta > 0]
    return {
        "duration_s": float(times[-1] - times[0]),
        "nonincreasing_intervals": int(np.count_nonzero(delta <= 0)),
        "median_dt_s": float(np.median(positive)) if len(positive) else None,
        "median_hz": float(1 / np.median(positive)) if len(positive) else None,
        "max_gap_s": float(delta.max()) if len(delta) else None,
        "gaps_over_0_2_s": int(np.count_nonzero(delta > 0.2)),
        "dt_p95_s": float(np.quantile(positive, 0.95)) if len(positive) else None,
    }


def inspect(path, output):
    if output.exists():
        raise ValueError("Use a new output directory; preserve prior results")
    before = path.stat()
    digest = hashlib.sha256()
    rows, errors, error_examples = [], Counter(), []
    lines = 0
    with path.open("rb") as stream:
        for lines, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                values = body_record(json.loads(raw))
                values["source_line"] = lines
                rows.append(values)
            except (ValueError, KeyError, TypeError) as exc:
                reason = f"{type(exc).__name__}: {exc}"
                errors[reason] += 1
                if len(error_examples) < 30:
                    error_examples.append({"line": lines, "reason": reason})
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed while reading")
    if len(rows) < 2:
        raise ValueError("Fewer than two valid body frames")
    arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    compact_dtypes = {}
    for name in BODY_FIELDS:
        smaller = arrays[name].astype(np.float32)
        if np.array_equal(smaller.astype(np.float64), arrays[name]):
            arrays[name] = smaller
        compact_dtypes[name] = str(arrays[name].dtype)
    local = arrays["pose_matrix"].astype(float).reshape(-1, 24, 3, 3)
    orthogonal_error = np.linalg.norm(local.transpose(0, 1, 3, 2) @ local - np.eye(3), axis=(-2, -1))
    determinants = np.linalg.det(local)
    global_pose = global_rotations(local)
    joints = arrays["joints"].astype(float)
    bone_vectors = joints[:, 1:] - joints[:, SMPL_PARENTS[1:]]
    bone_lengths = np.linalg.norm(bone_vectors, axis=-1)
    rest_vectors = np.einsum("fbji,fbj->fbi", global_pose[:, SMPL_PARENTS[1:]], bone_vectors)
    index_delta = np.diff(arrays["frame_index"])
    report = {
        "schema_version": 1, "usage": "TEST_ONLY", "training_allowed": False,
        "source_path": str(path.resolve()), "source_bytes": after.st_size,
        "source_sha256": digest.hexdigest(), "source_lines": lines,
        "valid_body_frames": len(rows), "invalid_lines": lines - len(rows),
        "invalid_reasons": dict(errors), "invalid_examples": error_examples,
        "storage": "BODY_FIELDS_ONLY_FULL_FRAME_COVERAGE_OF_VALID_RECORDS",
        "source_kind": "REAL_CLOTH_FGP_SMPL_RECORDING",
        "omitted_fields": ["camera_sensor", "hands", "hand_meshes", "camera_diagnostics"],
        "body_dtypes": compact_dtypes, "lossless_numeric_body_extraction": True,
        "source_timing": interval_summary(arrays["source_time_s"]),
        "host_timing": interval_summary(arrays["host_monotonic_s"]),
        "duplicate_frame_index_intervals": int(np.count_nonzero(index_delta == 0)),
        "backward_frame_index_intervals": int(np.count_nonzero(index_delta < 0)),
        "missing_frame_index_count": int(np.maximum(index_delta - 1, 0).sum()),
        "rotation_max_orthogonality_error": float(orthogonal_error.max()),
        "rotation_max_det_error": float(np.abs(determinants - 1).max()),
        "fused_vs_raw_rotation_max_abs_delta": float(arrays["fused_delta"].max()),
        "root_translation_range_m": np.ptp(arrays["trans"], axis=0).tolist(),
        "root_joint_vs_trans_max_error_m": float(np.abs(joints[:, 0] - arrays["trans"]).max()),
        "max_bone_length_range_m": float(np.ptp(bone_lengths, axis=0).max()),
        "max_rest_bone_vector_deviation_m": float(np.linalg.norm(
            rest_vectors - np.median(rest_vectors, axis=0), axis=-1).max()),
        "rest_bone_vectors_median_m": np.median(rest_vectors, axis=0).tolist(),
        "source_up_direction_evidence": "SMPL rest bone vectors; inspect before selecting basis",
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_raw_on_server": False, "baidu_backup_status": "LOCAL_ONLY",
    }
    output.mkdir(parents=True)
    bundle = output / "body_only.npz"
    np.savez_compressed(bundle, **arrays, parents=SMPL_PARENTS, usage=np.asarray("TEST_ONLY"),
                        training_allowed=np.asarray(False), source_sha256=np.asarray(digest.hexdigest()))
    report["bundle"] = {"name": bundle.name, "bytes": bundle.stat().st_size,
                        "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()}
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.input, args.output_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "rest_bone_vectors_median_m"}, indent=2))


if __name__ == "__main__":
    main()
