"""Import a user-transferred, release-hash-verified A3 PT without network access."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch

from script.a3.fullchain_support import ROOT, VENDOR, digest, write_json


def run(source):
    manifest = json.loads((VENDOR / "a3_hf_manifest.json").read_text())
    release = next(row for row in manifest["files"] if row["remote"] == "035_step200000/model_step_200000.pt")
    source = source.resolve(strict=True)
    if source.stat().st_size != release["size"] or digest(source) != release["sha256"]:
        raise ValueError("Transferred PT differs from the fixed official release")
    target = ROOT / "data/models/a3_official_035" / release["destination"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != release["size"] or digest(target) != release["sha256"]:
            raise FileExistsError("Do not overwrite an unverified existing model")
    else:
        partial = target.with_suffix(".pt.importing")
        shutil.copy2(source, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        if partial.stat().st_size != release["size"] or digest(partial) != release["sha256"]:
            raise ValueError("Copied PT size or SHA mismatch; partial file retained")
        partial.replace(target)
        from script.a3.checkpoint_store import fsync_directory
        fsync_directory(target.parent)
    if target.stat().st_size != release["size"] or digest(target) != release["sha256"]:
        raise ValueError("Imported PT size or SHA mismatch")
    torch.set_num_threads(1)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    networks = {}
    for name in ("policy_state_dict", "value_state_dict"):
        values = payload[name]
        tensors = [value for value in values.values() if isinstance(value, torch.Tensor)]
        if not tensors:
            raise ValueError("Official network contains no tensors")
        finite = all(not value.is_floating_point() or bool(torch.isfinite(value).all()) for value in tensors)
        networks[name] = {"tensors": len(tensors), "finite": finite,
                          "parameters": sum(value.numel() for value in tensors)}
        if not finite:
            raise ValueError("Nonfinite official network")
    optimizer = payload.get("optimizer_state_dict")
    optimizer_tensors = [] if optimizer is None else [value for state in optimizer["state"].values()
                                                     for value in state.values() if isinstance(value, torch.Tensor)]
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in optimizer_tensors):
        raise ValueError("Nonfinite optimizer tensor")
    configuration = []
    for row in manifest["files"]:
        if row["group"] != "pt" or row["remote"].endswith(".pt"):
            continue
        path = ROOT / "data/models/a3_official_035" / row["destination"]
        checked = path.is_file() and path.stat().st_size == row["size"] and digest(path) == row["sha256"]
        configuration.append({"name": Path(row["remote"]).name, "path": str(path),
                              "bytes": row["size"], "sha256": row["sha256"],
                              "status": "verified" if checked else "missing_or_unverified"})
    complete = all(row["status"] == "verified" for row in configuration)
    result = {"imported_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "checkpoint": str(target), "bytes": target.stat().st_size, "sha256": digest(target),
              "release_repo": manifest["repo_id"], "release_revision": manifest["revision"],
              "cpu_reload": "passed", "networks": networks,
              "source_global_step": int(payload["state"].global_step) if "state" in payload else None,
              "optimizer_tensor_count": len(optimizer_tensors), "optimizer_finite": True,
              "configuration": configuration, "bundle_integrity_verified": complete, "backup_status": "LOCAL_ONLY",
              "inference_tested": False, "training_started": False,
              "scope": "RELEASE_BUNDLE_INTEGRITY_AND_CPU_RELOAD_NOT_POLICY_ACCEPTANCE" if complete
                       else "PT_INTEGRITY_AND_CPU_RELOAD_ONLY_NOT_COMPLETE_MODEL_BUNDLE"}
    destination = ROOT / "data/manifests/a3_official_import_20261010.json"
    write_json(destination, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    run(parser.parse_args().source)
