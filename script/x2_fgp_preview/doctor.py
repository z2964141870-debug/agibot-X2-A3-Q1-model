"""Verify private runtime assets before any optional BLE connection."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--install-incoming", action="store_true")
    parser.add_argument("--load-models", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.root/"data/runtime/x2-fgp-viewer/assets_manifest.json").read_text())
    checked=[]
    for row in manifest["files"]:
        target=args.root/row["path"]
        if not target.exists() and args.install_incoming and row.get("incoming_name"):
            incoming=args.root/"data/runtime/x2-fgp-viewer/incoming"/row["incoming_name"]
            if not incoming.is_file() or incoming.stat().st_size!=row["bytes"] or digest(incoming)!=row["sha256"]:
                raise ValueError(f"incoming asset missing/mismatched: {incoming}")
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(incoming,target)
        if not target.is_file() or target.stat().st_size!=row["bytes"] or digest(target)!=row["sha256"]:
            raise ValueError(f"asset missing/mismatched: {target}")
        checked.append(row["path"])
    packages={name:importlib.metadata.version(name) for name in ("torch","numpy","scipy","numpy-quaternion","bleak","dbus-fast")}
    profile = None
    if args.load_models:
        from .engine import LatestFgp
        profile = LatestFgp(args.root/"data/runtime/clotho_fgp_latest/CLOTHO-mocap-main").model_profile
    print(json.dumps(dict(status="FGP_ASSETS_AND_IMPORTS_PASSED",files=len(checked),packages=packages,
                         python=sys.version,machine=platform.machine(),models_loaded=args.load_models,
                         model_profile=profile,
                         ble_connections=0,hardware_commands=0,feedback_subscriptions=0,publish_allowed=False)))


if __name__=="__main__":
    main()
