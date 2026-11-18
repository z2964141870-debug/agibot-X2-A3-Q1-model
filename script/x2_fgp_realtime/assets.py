"""Validate the complete author ZIP before optionally installing its assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


INVENTORY_SHA = "d838e8c07df5cfa9656be68b174c8cf0bdce5ac37a52226aaa5ebefa5df86d2f"
BUNDLES = ("onnx_s1", "onnx_cano_s_ep120_fast")


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return digest(stream)


def audit(archive, inventory):
    rows = []
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        names = [m.filename for m in members]
        if len(set(names)) != len(names) or set(names) != set(inventory):
            raise ValueError("ZIP must contain exactly the author inventory, without duplicates")
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            mode = stat.S_IFMT(member.external_attr >> 16)
            if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
                    or mode not in (0, stat.S_IFREG) or member.flag_bits & 1):
                raise ValueError("Unsafe ZIP member: " + name)
            expected = inventory[name]
            if member.file_size != expected["bytes"]:
                raise ValueError("Asset size mismatch: " + name)
            with package.open(member) as stream:
                actual = digest(stream)
            if actual != expected["sha256"]:
                raise ValueError("Asset SHA mismatch: " + name)
            rows.append(dict(path=name, bytes=member.file_size, sha256=actual))
        bundles = {}
        for name in BUNDLES:
            manifest = json.loads(package.read("models/" + name + "/manifest.json"))
            gp = json.loads(package.read("models/" + name + "/gp_manifest.json"))
            bundles[name] = dict(model_size=manifest["model_size"],
                window_size=manifest["window_size"], cano_output=manifest.get("cano_output", "full"),
                checkpoint_hashes={k: v["sha256"] for k, v in manifest["checkpoints"].items()},
                gp_metadata=gp["model_metadata"],
                upstream_mismatches=[k for k in ("cano", "lip")
                    if manifest["checkpoints"][k]["sha256"] != gp["model_metadata"].get(k + "_checkpoint_sha256")])
    return dict(status="AUTHOR_REALTIME_ASSETS_PASSED", files=rows, bundles=bundles,
                total_unpacked_bytes=sum(r["bytes"] for r in rows),
                archive_bytes=Path(archive).stat().st_size, archive_sha256=file_digest(archive))


def install(archive, root, rows):
    root = Path(root).resolve()
    targets = []
    for row in rows:
        target = root / row["path"]
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError("Unsafe installation target: " + row["path"])
        if target.exists() and (not target.is_file() or file_digest(target) != row["sha256"]):
            raise FileExistsError("Different existing asset: " + row["path"])
        targets.append((target, row))
    with zipfile.ZipFile(archive) as package:
        for target, row in targets:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".incoming")
                with temporary.open("xb") as dest, package.open(row["path"]) as source:
                    shutil.copyfileobj(source, dest)
                if file_digest(temporary) != row["sha256"]:
                    raise ValueError("Extracted asset mismatch: " + row["path"])
                temporary.rename(target)
            if target.stat().st_size != row["bytes"] or file_digest(target) != row["sha256"]:
                raise ValueError("Installed asset mismatch: " + row["path"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; inspect the previous result")
    source = args.source_root / "deployment_assets.json"
    if file_digest(source) != INVENTORY_SHA:
        raise ValueError("Pinned author deployment inventory changed")
    result = audit(args.archive, json.loads(source.read_text())["files"])
    if args.install:
        install(args.archive, args.source_root, result["files"])
    result.update(installed=args.install, models_loaded=0, hardware_commands=0,
                  ble_connections=0, publish_allowed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "bundles")}))


if __name__ == "__main__":
    main()
