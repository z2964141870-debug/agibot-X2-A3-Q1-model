#!/usr/bin/env python3
"""Index AMASS archive headers without loading motion or marker tensors."""

import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def header(archive, key):
    with archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY header version {version}")
    if dtype.hasobject:
        raise ValueError(f"Object array is not accepted: {key}")
    return {"shape": list(shape), "dtype": dtype.str, "fortran_order": bool(order)}


def scalar(archive, key):
    if archive.getinfo(key + ".npy").file_size > 4096:
        raise ValueError(f"Scalar metadata too large: {key}")
    with archive.open(key + ".npy") as stream:
        value = np.lib.format.read_array(stream, allow_pickle=False)
    if value.size != 1:
        raise ValueError(f"Scalar metadata has multiple values: {key}")
    result = value.item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def inspect_motion(path, root):
    info = path.stat()
    relative = path.relative_to(root)
    row = {"relative_path": relative.as_posix(), "collection": relative.parts[0],
           "bytes": info.st_size, "allocated_bytes": info.st_blocks * 512,
           "mtime_ns": info.st_mtime_ns, "inode": info.st_ino, "device": info.st_dev,
           "hard_links": info.st_nlink, "qualified_for_q1_training": False}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("Duplicate ZIP members")
        keys = set(members)
        if "poses.npy" not in keys and "trans.npy" not in keys:
            row.update(kind="shape_or_static_asset", schema_valid=False)
            return row
        pose = header(archive, "poses")
        trans = header(archive, "trans")
        betas = header(archive, "betas")
        rates = {key: float(scalar(archive, key)) for key in
                 ("mocap_frame_rate", "mocap_framerate") if key + ".npy" in keys}
        if not rates:
            raise ValueError("Missing explicit source frame rate; no default is allowed")
        fps = next(iter(rates.values()))
        if not math.isfinite(fps) or fps <= 0 or any(value != fps for value in rates.values()):
            raise ValueError("Invalid or conflicting source frame rates")
        if (len(pose["shape"]) != 2 or pose["shape"][1] != 165
                or pose["shape"][0] < 2 or trans["shape"] != [pose["shape"][0], 3]
                or len(betas["shape"]) != 1 or betas["shape"][0] < 1):
            raise ValueError("Unsupported SMPL-X motion array shapes")
        model_type = scalar(archive, "surface_model_type")
        gender = scalar(archive, "gender")
        if model_type != "smplx" or gender not in {"male", "female", "neutral"}:
            raise ValueError("Unsupported body model or gender metadata")
        frames = pose["shape"][0]
        row.update(kind="smplx_motion", schema_valid=True, source_frames=frames,
                   source_hz=fps, source_hz_fields=list(rates),
                   nominal_duration_s=(frames - 1) / fps,
                   gender=gender, surface_model_type=model_type,
                   headers={"poses": pose, "trans": trans, "betas": betas})
    return row


def health(expected_oom):
    base = Path("/sys/fs/cgroup/memory")
    if not base.exists():
        return {"available": False}
    current = int((base / "memory.usage_in_bytes").read_text())
    oom = int(dict(line.split() for line in
                   (base / "memory.oom_control").read_text().splitlines())["oom_kill"])
    if expected_oom is not None and oom != expected_oom:
        raise RuntimeError("Container OOM count differs from the admitted snapshot")
    if current + 256 * 1024 ** 2 >= 30 * 1024 ** 3:
        raise RuntimeError("Inventory memory protection")
    return {"available": True, "memory_bytes": current, "oom_kill": oom}


def run(args):
    root = args.source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source root must be a directory")
    if args.output_dir.resolve().is_relative_to(root):
        raise ValueError("Inventory output must be outside the read-only source tree")
    baseline = health(args.expected_oom_kill)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts, fps_counts, genders = (collections.Counter() for _ in range(3))
    by_collection = {}
    unique_files = set()
    sample_paths = {}
    logical_bytes = allocated_bytes = unique_logical_bytes = 0
    duration = 0.
    errors = []
    index_path = args.output_dir / "all_files.jsonl"
    with index_path.open("x") as output:
        for index, path in enumerate(sorted(root.rglob("*.npz"))):
            info = path.stat()
            logical_bytes += info.st_size
            allocated_bytes += info.st_blocks * 512
            file_key = (info.st_dev, info.st_ino)
            if file_key not in unique_files:
                unique_logical_bytes += info.st_size
                unique_files.add(file_key)
            try:
                row = inspect_motion(path, root)
            except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
                row = {"relative_path": path.relative_to(root).as_posix(),
                       "kind": "rejected_metadata", "schema_valid": False,
                       "qualified_for_q1_training": False,
                       "error": f"{type(exc).__name__}: {exc}"}
                errors.append(row)
            counts[row["kind"]] += 1
            if row["schema_valid"]:
                seconds = row["nominal_duration_s"]
                duration += seconds
                fps_counts[str(row["source_hz"])] += 1
                genders[row["gender"]] += 1
                part = by_collection.setdefault(row["collection"], {"motions": 0, "nominal_duration_s": 0.})
                part["motions"] += 1
                part["nominal_duration_s"] += seconds
                sample_paths.setdefault(row["collection"], path)
            output.write(json.dumps(row, allow_nan=False) + "\n")
            if (index + 1) % 1000 == 0:
                health(baseline.get("oom_kill"))
                print(json.dumps({"indexed_files": index + 1, "counts": dict(counts)}), flush=True)
    summary = {"status": "AMASS_HEADER_INVENTORY_COMPLETED",
               "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "source_root": str(root), "counts": dict(counts),
               "npz_files": sum(counts.values()), "logical_bytes": logical_bytes,
               "allocated_bytes_summed_per_path": allocated_bytes,
               "unique_inode_count": len(unique_files), "unique_inode_logical_bytes": unique_logical_bytes,
               "motion_nominal_duration_s": duration, "motion_nominal_hours": duration / 3600,
               "source_hz_counts": dict(fps_counts), "gender_counts": dict(genders),
               "collections": by_collection, "errors": errors,
               "index": identity(index_path), "script": identity(Path(__file__).resolve()),
               "sample_file_identities": [identity(p) for p in sample_paths.values()],
               "health_before": baseline, "health_after": health(baseline.get("oom_kill")),
               "elapsed_s": time.monotonic() - started,
               "new_retargeting": 0, "new_physics_steps": 0, "new_optimizer_updates": 0,
               "storage_status": "LOCAL_ONLY",
               "limits": "Archive headers and bounded scalar metadata only; sample SHA per collection. No full-corpus SHA/CRC or numeric validity, deduplication, cross-dataset identity, splits, Q1 reference quality or tracking qualification."}
    with (args.output_dir / "summary.json").open("x") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int)
    run(parser.parse_args())
