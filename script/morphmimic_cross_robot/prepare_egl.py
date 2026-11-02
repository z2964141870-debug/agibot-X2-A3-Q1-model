"""Extract matching NVIDIA graphics libraries into an isolated project folder."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

VERSION = "595.112.07"
BASE = "https://developer.download.nvidia.com/compute/nvidia-driver/redist/"


def is_x86_64(data):
    return (len(data) >= 20 and data[:4] == b"\x7fELF" and data[4] == 2
            and data[5] == 1 and int.from_bytes(data[18:20], "little") == 62)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = urllib.request.urlopen(BASE + "redistrib_" + VERSION + ".json", timeout=30).read()
    (args.output / "redistrib.json").write_bytes(metadata)
    row = json.loads(metadata)["nvidia_driver"]["linux-x86_64"]
    archive = args.archive or (args.output / "driver_archive.tar.xz")
    digest = hashlib.sha256()
    size = 0
    if args.archive is not None:
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    else:
        with urllib.request.urlopen(BASE + row["relative_path"], timeout=60) as response:
            with archive.open("xb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    if size > int(row["size"]):
                        raise ValueError("Driver archive exceeds pinned size")
                    stream.write(block)
    if size != int(row["size"]) or digest.hexdigest() != row["sha256"]:
        raise ValueError("Driver archive size/SHA differs from NVIDIA metadata")
    libraries = args.output / "lib"
    libraries.mkdir()
    prefixes = ("libEGL_nvidia.", "libGLESv1_CM_nvidia.", "libGLESv2_nvidia.",
                "libGLX_nvidia.", "libnvidia-glcore.", "libnvidia-eglcore.",
                "libnvidia-glsi.", "libnvidia-tls.", "libnvidia-glvkspirv.",
                "libnvidia-allocator.", "libnvidia-gpucomp.", "libnvidia-egl-gbm.")
    files = []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or not name.startswith(prefixes):
                continue
            data = tar.extractfile(member).read()
            if not is_x86_64(data):
                continue
            path = libraries / name
            path.write_bytes(data)
            files.append(dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            if ".so." in name:
                soname = name.split(".so.")[0] + ".so.1"
                link = libraries / soname
                if link != path and not link.exists():
                    link.symlink_to(path.name)
    egl = libraries / ("libEGL_nvidia.so." + VERSION)
    if not egl.is_file():
        raise RuntimeError("Archive has no matching EGL vendor library")
    (args.output / "10_nvidia.json").write_text(json.dumps(dict(file_format_version="1.0.0",
        ICD=dict(library_path=str(egl)))) + "\n")
    receipt = dict(status="ARCHIVE_VERIFIED_LIBRARIES_EXTRACTED", version=VERSION,
        archive=dict(path=str(archive), bytes=size, sha256=digest.hexdigest()), files=files,
        driver_install_or_system_configuration_changed=False, graphics_runtime_tested=False)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
