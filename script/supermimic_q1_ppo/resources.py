"""Index existing Q1 resources and remove only inventoried, unused AMASS motion files."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def project_path(root, path):
    for old in ("/opt/data/private/YuanQi_Q1", "/opt/data/private/yuanqi_ops"):
        if str(path).startswith(old + "/"):
            return root / str(path)[len(old) + 1:]
    return Path(path)


def index_resources(root, output):
    sources = {
        "SOMA_raw": ("data/references/bones_seed/inbox/soma_uniform.tar.gz", "raw_human_BVH", False),
        "SOMA_canonical": ("data/references/bones_seed/inbox/soma_base_skel_minimal.bvh", "skeleton", False),
        "SOMA_metadata": ("data/references/bones_seed/metadata/seed_metadata_v004.parquet", "metadata", False),
        "SOMA_AP02_Q1": ("data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json", "Q1_reference", True),
        "PHUMA_Q1": ("data/supermimic_for_q1/PHUMA_P08_expanded_library_20261025/library/references/manifest.json", "Q1_reference_historical_filters", True),
        "PHUMA_inventory": ("data/manifests/supermimic_q1_phuma_inventory_20261024.json", "G1_H1_raw_inventory", False),
        "BONES_inventory": ("data/manifests/supermimic_q1_bones_metadata_intake_20261024.json", "other_robot_raw_inventory", False),
        "AMASS_inventory": ("data/manifests/supermimic_q1_amass_inventory_20261024.json", "SMPLX_raw_inventory", False),
        "RF11_results": ("data/manifests/supermimic_q1_RF11_results_20261027.json", "physical_trial_index", False),
    }
    records = []
    for name, (relative, kind, q1_ready) in sources.items():
        path = root / relative
        record = dict(name=name, kind=kind, q1_ready=q1_ready, path=str(path), exists=path.exists())
        if path.exists():
            record["bytes"] = path.stat().st_size
            if path.stat().st_size < 32 * 1024**2:
                record["identity"] = identity(path)
        records.append(record)
    lib_path = root / sources["SOMA_AP02_Q1"][0]
    library = json.loads(lib_path.read_text())
    clips = [dict(row, split=split, source_dataset="SOMA", source_fps=120,
                  reference_fps=30, control_fps=50, root_quaternion="xyzw",
                  coordinates="Q1 z-up, root position in metres, joints radians")
             for split, rows in library["splits"].items() for row in rows]
    identities = [(row["name"], row["actor_uid"]) for row in clips]
    digests = Counter(row["sha256"] for row in clips)
    actors = {split: {r["actor_uid"] for r in clips if r["split"] == split}
              for split in library["splits"]}
    duplicate_hashes = [key for key, count in digests.items() if count > 1]
    if len(identities) != len(set(identities)) or actors["train"] & actors["validation"]:
        raise ValueError("AP02 identity/actor split contract failed")
    save(output, {"status": "Q1_RESOURCE_INDEX_CREATED", "captured_utc": stamp(),
                  "resources": records, "AP02_clips": clips,
                  "AP02_duplicate_content_sha256": duplicate_hashes,
                  "AP02_actor_overlap": [], "storage": "LOCAL_ONLY",
                  "cross_dataset_deduplication": "SOMA identities/hashes checked. PHUMA, BONES and AMASS are inventoried but not mixed; cross-dataset motion equivalence is not established.",
                  "models_checkpoints_trajectories": "Existing trial receipts and manifests retain exact paths/identities; raw data are not automatically Q1-ready. No large-file rehash."})


def cleanup(root, output, execute):
    raw = root / "data/datasets/amass_bridge/amass_raw/AMASS_smplxg"
    if raw.is_symlink() or not raw.resolve(strict=True).is_relative_to(root.resolve()):
        raise ValueError("AMASS raw root must be a real subtree of ZHY")
    old_receipt = json.loads((root / "data/manifests/supermimic_q1_amass_inventory_20261024.json").read_text())
    index = project_path(root, old_receipt["index"]["path"])
    if identity(index)["sha256"] != old_receipt["index"]["sha256"]:
        raise ValueError("Historical AMASS inventory changed")
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    planned, preserved = [], []
    for row in rows:
        path = raw / row["relative_path"]
        if not path.resolve(strict=True).is_relative_to(raw.resolve()) or path.is_symlink():
            raise ValueError("Unexpected symlink or path outside AMASS raw")
        info = path.stat()
        if row["kind"] != "smplx_motion":
            preserved.append({"path": str(path), "kind": row["kind"], "bytes": info.st_size})
            continue
        # The verified container migration rounded subsecond mtimes; require
        # identical size and whole seconds rather than silently ignoring time.
        if info.st_size != row["bytes"] or info.st_mtime_ns // 10**9 != row["mtime_ns"] // 10**9:
            raise ValueError("AMASS motion changed since historical inventory: " + str(path))
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Only regular inventoried files may be removed")
        planned.append({"path": str(path), "bytes": info.st_size, "inode": info.st_ino,
                        "device": info.st_dev, "mtime_ns": info.st_mtime_ns, "links": info.st_nlink,
                        "historical_mtime_ns": row["mtime_ns"]})
    open_uses = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            links = list((process / "fd").iterdir()) + [process / "cwd", process / "exe"]
            for link in links:
                target = Path(os.readlink(link))
                if target.is_absolute() and target.is_relative_to(raw):
                    open_uses.append({"pid": int(process.name), "link": str(link), "target": str(target)})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    dependent_links = []
    # Inspect the active reference/model/trial trees, not the entire shared disk.
    for subtree in (root / "script", root / "data/references", root / "data/supermimic_for_q1"):
        for directory, dirs, files in os.walk(subtree, followlinks=False):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "source_bvh")]
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink() and path.resolve().is_relative_to(raw):
                    dependent_links.append({"path": str(path), "target": str(path.resolve())})
    library = root / "data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json"
    active = json.loads(library.read_text())
    for values in active["splits"].values():
        for row in values:
            if Path(row["path"]).resolve().is_relative_to(raw):
                raise ValueError("Active AP02 references depend on AMASS raw")
    receipt = {"status": "DELETION_MANIFEST_READY", "captured_utc": stamp(), "raw_root": str(raw),
               "historical_inventory": identity(index), "active_AP02": identity(library),
               "planned_files": len(planned), "planned_logical_bytes": sum(r["bytes"] for r in planned),
               "preserved_static_or_unclassified_files": len(preserved), "open_uses": open_uses,
               "dependent_symlinks": dependent_links, "not_a_backup": True,
               "preserved": preserved, "deletion_manifest": planned,
               "limits": "Delete only inventoried motion NPZs. Static shapes/unclassified files, metadata, all converted references/models and historical evidence remain. Hardlinks may reduce physical reclaimed bytes."}
    if open_uses or dependent_links:
        receipt["status"] = "DELETION_BLOCKED_BY_LIVE_DEPENDENCY"
    if output.exists():
        raise FileExistsError(output)
    save(output, receipt)
    if not execute:
        return receipt
    if open_uses or dependent_links:
        raise RuntimeError("AMASS raw still has active use or dependent symlinks")
    before = os.statvfs(root)
    deleted_bytes = 0
    for record in planned:
        path = Path(record["path"])
        info = path.stat()
        if any(record[key] != value for key, value in (
            ("bytes", info.st_size), ("inode", info.st_ino), ("device", info.st_dev), ("mtime_ns", info.st_mtime_ns))):
            raise RuntimeError("Deletion source identity changed: " + str(path))
        path.unlink()
        deleted_bytes += record["bytes"]
    after = os.statvfs(root)
    receipt.update(status="UNUSED_AMASS_RAW_MOTIONS_REMOVED", completed_utc=stamp(),
                   deleted_files=len(planned), deleted_logical_bytes=deleted_bytes,
                   filesystem_available_bytes_before=before.f_bavail * before.f_frsize,
                   filesystem_available_bytes_after=after.f_bavail * after.f_frsize,
                   preserved_models_and_converted_references=True)
    save(output.with_name("cleanup_result.json"), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("index", "amass-cleanup", "receipt"))
    parser.add_argument("--root", type=Path, default=Path("/opt/data/private/ZHY"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "index":
        index_resources(args.root, args.output)
    elif args.mode == "amass-cleanup":
        value = cleanup(args.root, args.output, args.execute)
        print(json.dumps({k: v for k, v in value.items() if k not in ("preserved", "deletion_manifest")}))
    else:
        prep = args.root / "data/supermimic_for_q1/RF12_PREPARATION_20261027"
        result = json.loads((prep / "cleanup_result.json").read_text())
        save(args.output, {**{k: v for k, v in result.items() if k not in ("preserved", "deletion_manifest")},
                           "full_deletion_manifest": identity(prep / "amass_deletion_manifest.json"),
                           "full_cleanup_result": identity(prep / "cleanup_result.json"),
                           "resource_index": identity(prep / "resource_index.json"),
                           "first_dryrun": "Strict nanosecond mtime check stopped before deletion. Migration truncation to whole seconds verified; size and whole seconds subsequently required for every motion.",
                           "new_optimizer_updates": 0})


if __name__ == "__main__":
    main()
