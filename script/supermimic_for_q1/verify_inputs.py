#!/usr/bin/env python3
"""Verify transferred ZIP identities and compile Q1 assets without stepping physics."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import zipfile

import mujoco


def identity(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text())
    directory = args.root / "data/references/q1/archives"
    archives = []
    for item in expected["archives"]:
        found = identity(directory / item["filename"])
        if any(found[key] != item[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Transferred archive identity mismatch: {item['filename']}")
        archives.append(found)
    model_dir = args.root / "data/references/q1/model_20261023_P01"
    model_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / "q1_v3.zip") as archive:
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)
            if member.is_absolute() or ".." in member.parts or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe model ZIP member")
            destination = model_dir.joinpath(*member.parts)
            if entry.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                content = archive.read(entry)
                if destination.exists():
                    if identity(destination)["sha256"] != hashlib.sha256(content).hexdigest():
                        raise ValueError(f"Existing extracted model changed: {entry.filename}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream:
                        stream.write(content)
    models = []
    for path in sorted((model_dir / "q1_v3/xml").glob("*.xml")):
        model = mujoco.MjModel.from_xml_path(str(path))
        models.append({"input": identity(path), "nq": model.nq, "nv": model.nv,
                       "nu": model.nu, "njnt": model.njnt, "nbody": model.nbody,
                       "ngeoms": model.ngeom, "physics_steps": 0})
        expected_nu = 22 if path.name == "q1_v3_mc.xml" else 0
        if (model.nq, model.nv, model.nu) != (29, 28, expected_nu):
            raise ValueError("Unexpected Q1 dimensions")
    if len(models) != 3:
        raise ValueError("Expected three Q1 MJCF models")
    receipt = {"stage": "SM-Q1-P01", "status": "INPUTS_AND_MODEL_COMPILE_PASSED",
               "scope": "No physics steps, policy inference, optimizer or hardware control",
               "platform": platform.platform(), "mujoco_version": mujoco.__version__,
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "archives": archives, "models": models,
               "mesh_files": [identity(path) for path in sorted((model_dir / "q1_v3/meshes").glob("*.STL"))],
               "source": identity(__file__), "input_manifest": identity(args.manifest),
               "storage_status": "LOCAL_ONLY"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "archives_verified": len(archives),
                      "models_compiled": len(models), "meshes": len(receipt["mesh_files"]),
                      "physics_steps": 0, "output": identity(args.output)}, indent=2))


if __name__ == "__main__":
    main()
