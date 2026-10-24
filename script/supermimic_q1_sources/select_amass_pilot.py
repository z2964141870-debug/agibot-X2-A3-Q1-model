#!/usr/bin/env python3
"""Freeze bounded AMASS walking windows for diagnostics, without training admission."""

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

import numpy as np

from inventory_amass import identity, inspect_motion


def check_window(path, indexed):
    current = inspect_motion(path, path.parents[len(PurePosixPath(indexed["relative_path"]).parts) - 1])
    for key in ("relative_path", "source_frames", "source_hz", "gender", "headers", "bytes", "mtime_ns"):
        if current[key] != indexed[key]:
            raise ValueError(f"AMASS index changed: {key}")
    if current["source_hz"] != 120. or current["headers"]["betas"]["shape"] != [16]:
        raise ValueError("This pilot requires explicit 120 Hz and all sixteen betas")
    frames = 481
    if current["source_frames"] < frames or current["source_frames"] > 2500:
        raise ValueError("Source frame count exceeds the bounded four-second pilot")
    with np.load(path, allow_pickle=False) as source:
        poses = source["poses"][:frames].copy()
        trans = source["trans"][:frames].copy()
        betas = source["betas"].copy()
        if (poses.shape != (frames, 165) or trans.shape != (frames, 3) or betas.shape != (16,)
                or not all(np.isfinite(value).all() for value in (poses, trans, betas))):
            raise ValueError("Nonfinite or inconsistent source window arrays")
        for key, start, stop in (("root_orient", 0, 3), ("pose_body", 3, 66),
                                 ("pose_jaw", 66, 69), ("pose_eye", 69, 75), ("pose_hand", 75, 165)):
            if not np.allclose(source[key][:frames], poses[:, start:stop], atol=1e-8, rtol=0):
                raise ValueError(f"Full pose disagrees with {key}")
    horizontal = trans[:, :2] - trans[:1, :2]
    speed = np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=-1) * 120.
    statistics = {
        "root_max_horizontal_displacement_m": float(np.linalg.norm(horizontal, axis=-1).max()),
        "root_horizontal_path_m": float(np.linalg.norm(np.diff(trans[:, :2], axis=0), axis=-1).sum()),
        "root_horizontal_speed_p95_m_s": float(np.quantile(speed, .95)),
        "root_translation_z_span_m": float(np.ptp(trans[:, 2])),
    }
    gates = {
        "moving_window": .2 <= statistics["root_max_horizontal_displacement_m"] <= 2.,
        "gentle_root_speed": statistics["root_horizontal_speed_p95_m_s"] <= 1.5,
        "limited_root_translation_z_span": statistics["root_translation_z_span_m"] <= .2,
    }
    return {"source_start_frame": 0, "source_stop_frame_exclusive": frames, "source_hz": 120.,
            "source_window_duration_s": 4., "planned_target_hz": 50., "planned_target_frames": 201,
            "num_betas": 16, "full_pose_fields_agree_in_window": True,
            "source_window_arrays_finite": True, "root_statistics": statistics,
            "candidate_gates": gates, "raw_window_candidate": all(gates.values())}


def candidates(index_path, seed):
    excluded = ("run", "hop", "jump", "leap", "box", "stair", "backward", "crouch")
    rows = []
    with index_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            path = row.get("relative_path", "").lower()
            if (row.get("schema_valid") and row.get("kind") == "smplx_motion"
                    and row["source_hz"] == 120. and row["headers"]["betas"]["shape"] == [16]
                    and 481 <= row["source_frames"] <= 2500 and row["bytes"] <= 16 * 1024 ** 2
                    and "walk" in path and not any(term in path for term in excluded)):
                rows.append(row)
    return sorted(rows, key=lambda row: hashlib.sha256(
        (str(seed) + ":" + row["relative_path"]).encode()).hexdigest())


def run(args):
    if not 1 <= args.count <= 32 or not args.count <= args.maximum_probes <= 128:
        raise ValueError("Use at most 32 candidates and 128 bounded numeric probes")
    root = args.root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "data/supermimic_for_q1"):
        raise ValueError("Use a new independent Q1 output directory")
    cgroup = Path("/sys/fs/cgroup/memory")
    observed = []

    def resource_check():
        current = int((cgroup / "memory.usage_in_bytes").read_text())
        oom = int(dict(line.split() for line in (cgroup / "memory.oom_control").read_text().splitlines())["oom_kill"])
        if current >= int(28.5 * 1024 ** 3) or oom != args.expected_oom_kill:
            raise RuntimeError("AMASS source selection stopped at its memory reserve or new OOM")
        stat = os.statvfs(output.parent)
        if stat.f_bavail * stat.f_frsize < 100 * 1024 ** 3:
            raise RuntimeError("AMASS source selection disk reserve reached")
        observed.append(current)
        return {"memory_bytes": current, "oom_kill": oom}

    baseline = resource_check()
    index_id = identity(args.index.resolve())
    eligible = candidates(args.index, args.seed)
    resource_check()
    output.mkdir()
    started, accepted, checked, hashes = time.monotonic(), [], [], set()
    try:
        for row in eligible:
            if len(accepted) >= args.count or len(checked) >= args.maximum_probes:
                break
            resource_check()
            relative = PurePosixPath(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("AMASS index contains an unsafe relative path")
            path = (source_root / relative).resolve(strict=True)
            if not path.is_relative_to(source_root):
                raise ValueError("AMASS source leaves the read-only source tree")
            before = identity(path)
            details = check_window(path, row)
            if identity(path) != before:
                raise ValueError("AMASS source changed during numeric loading")
            with path.open("rb") as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            record = {"relative_path": row["relative_path"], "source": before, "collection": row["collection"],
                      "gender": row["gender"], "source_group": str(relative.with_suffix("")),
                      "split": "UNASSIGNED_DIAGNOSTIC_ONLY", "qualified_for_q1_training": False, **details}
            if before["sha256"] in hashes:
                record["raw_window_candidate"] = False
                record["duplicate_source_bytes"] = True
            hashes.add(before["sha256"])
            checked.append(record)
            if record["raw_window_candidate"]:
                accepted.append(record)
            resource_check()
        if identity(args.index.resolve()) != index_id:
            raise ValueError("AMASS header index changed during selection")
        after = resource_check()
        record = {
            "status": "AMASS_BOUNDED_RAW_MOVING_CANDIDATES_SELECTED",
            "captured_utc": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root),
            "index": index_id, "seed": args.seed, "eligible_name_header_candidates": len(eligible),
            "requested_candidates": args.count, "maximum_probes": args.maximum_probes,
            "probed_count": len(checked), "selected_count": len(accepted),
            "selected_collection_counts": dict(collections.Counter(row["collection"] for row in accepted)),
            "selected_gender_counts": dict(collections.Counter(row["gender"] for row in accepted)),
            "selected_source_window_duration_s": sum(row["source_window_duration_s"] for row in accepted),
            "selected": accepted, "checked": checked, "source": identity(Path(__file__).resolve()),
            "sources": [identity(Path(__file__).resolve()), identity(Path(__file__).with_name("inventory_amass.py").resolve())],
            "health_before": baseline, "health_after": after, "max_sampled_shared_bytes": max(observed),
            "elapsed_s": time.monotonic() - started, "new_q1_references": 0, "new_physics_steps": 0,
            "new_optimizer_updates": 0, "qualified_for_q1_training": False, "storage_status": "LOCAL_ONLY",
            "limits": "Name hints and finite first-four-second raw translations only. No SMPL-X FK, Q1 retargeting or geometry checks. Splits intentionally unassigned; HumanML/PHUMA aliases and source subjects must be reconciled before mixing or evaluation.",
        }
        with (output / "selection.json").open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({key: value for key, value in record.items() if key not in ("selected", "checked")}), flush=True)
    except BaseException as exc:
        with (output / "failure.json").open("x") as stream:
            json.dump({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                       "checked_count": len(checked), "new_q1_references": 0,
                       "qualified_for_q1_training": False}, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--maximum-probes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    run(parser.parse_args())
