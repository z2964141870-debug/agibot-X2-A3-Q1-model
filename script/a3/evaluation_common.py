"""Trusted artifact checks shared by explicit A3 evaluation entry points."""

import json
from pathlib import Path

from script.a3.autoresume import ROOT, VENDOR_COMMIT
from script.a3.checkpoint_store import sha256
from script.a3.host_health import command, sample


def checkpoint_metadata(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(ROOT / "data"):
        raise ValueError("Checkpoint must stay in project data")
    if not checkpoint.with_name("config.yaml").is_file():
        raise ValueError("Missing matching checkpoint config.yaml")
    actual = sha256(checkpoint)
    sidecar = checkpoint.with_suffix(".json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if actual != metadata["sha256"] or checkpoint.stat().st_size != metadata["bytes"]:
            raise ValueError("Checkpoint integrity mismatch")
        return metadata
    manifest = json.loads((ROOT / "script/vendor/sonic_for_a3/a3_hf_manifest.json").read_text())
    candidates = [row for row in manifest["files"] if row["remote"].endswith(".pt")]
    for row in candidates:
        if actual == row["sha256"] and checkpoint.stat().st_size == row["size"]:
            configuration = next(item for item in manifest["files"] if item["remote"] == "035_step200000/config.yaml")
            if sha256(checkpoint.with_name("config.yaml")) != configuration["sha256"]:
                raise ValueError("Official configuration checksum mismatch")
            return {"sha256": actual, "bytes": row["size"], "source": "official_035",
                    "step": 200000, "backup_status": "LOCAL_ONLY"}
    raise ValueError("Unregistered checkpoint: no trusted sidecar or release SHA match")


def preflight():
    vendor = ROOT / "script/vendor/sonic_for_a3"
    if command(["git", "-C", str(vendor), "rev-parse", "HEAD"])["stdout"] != VENDOR_COMMIT:
        raise ValueError("Vendor version changed")
    gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if gpu["returncode"] != 0 or gpu["stdout"]:
        raise RuntimeError("GPU unreadable or occupied")
    health = sample()
    cpu = health["temperatures_c"].get("x86_pkg_temp")
    if (cpu is None or cpu >= 90 or health["gpu_query"]["returncode"] != 0 or not health["gpus"] or
            any(g.get("temperature.gpu") is None or g["temperature.gpu"] >= 85 for g in health["gpus"])):
        raise RuntimeError("Unreadable or high temperatures before launch")
    return health
