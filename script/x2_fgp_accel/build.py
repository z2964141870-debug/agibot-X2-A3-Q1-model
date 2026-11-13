"""Bounded TensorRT CANO build in a fresh output, with own-child cleanup."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from script.x2_fgp_realtime.assets import file_digest


PRECISION_FLAGS = {
    "fp32": (),
    "fp16": ("--fp16",),
    "mixed": ("--fp16", "--precisionConstraints=obey",
              "--layerPrecisions=*LayerNormalization:fp32,*Softmax:fp32,/ReduceSum*:fp32",
              "--layerOutputTypes=*LayerNormalization:fp32,*Softmax:fp32,/ReduceSum*:fp32"),
}


def resources():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    temperatures = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            temperatures[(zone / "type").read_text().strip()] = int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
    if not temperatures:
        raise RuntimeError("No temperature telemetry")
    return dict(available_bytes=available, max_temperature_c=max(temperatures.values()),
                temperatures_c=temperatures)


def require_resources(snapshot):
    if snapshot["available_bytes"] < 4 * 1024**3:
        raise RuntimeError("Less than 4 GiB available memory")
    if snapshot["max_temperature_c"] >= 80:
        raise RuntimeError("Temperature reached 80 C")


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--precision", choices=tuple(PRECISION_FLAGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    if shutil.disk_usage(args.output.parent).free < 10 * 1024**3:
        parser.error("Less than 10 GiB free disk")
    before = resources()
    require_resources(before)
    onnx_hash = file_digest(args.onnx)
    inventory = json.loads((args.onnx.parent / "manifest.json").read_text())
    if onnx_hash != inventory["files"][args.onnx.name]:
        parser.error("ONNX does not match its author manifest")
    lock_path = Path("data/locks/x2_fgp_accel.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        engine = args.output / "cano.plan"
        command = ["/usr/src/tensorrt/bin/trtexec", "--onnx=" + str(args.onnx.resolve()),
                   "--saveEngine=" + str(engine.resolve()), "--memPoolSize=workspace:256M",
                   "--builderOptimizationLevel=2", "--maxAuxStreams=0", "--avgTiming=2",
                   "--noTF32", "--skipInference"]
        command.extend(PRECISION_FLAGS[args.precision])
        child = None
        started = time.monotonic()
        samples = [before]
        status, error = "BUILD_FAILED", None
        try:
            with (args.output / "build.log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                while child.poll() is None:
                    snapshot = resources()
                    samples.append(snapshot)
                    require_resources(snapshot)
                    if time.monotonic() - started > 600:
                        raise TimeoutError("600 second build limit")
                    time.sleep(1)
                if child.returncode != 0:
                    raise RuntimeError("trtexec exited " + str(child.returncode))
            if not engine.is_file() or not engine.stat().st_size:
                raise RuntimeError("No serialized engine")
            if file_digest(args.onnx) != onnx_hash:
                raise RuntimeError("Original ONNX changed")
            status = "BUILD_PASSED"
        except BaseException as caught:
            error = repr(caught)
        finally:
            stop_child(child)
            receipt = dict(status=status, error=error, precision=args.precision,
                wall_s=time.monotonic()-started, command=command,
                onnx_sha256=onnx_hash, engine_sha256=file_digest(engine) if engine.is_file() else None,
                engine_bytes=engine.stat().st_size if engine.is_file() else 0,
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                min_available_bytes=min(r["available_bytes"] for r in samples),
                max_temperature_c=max(r["max_temperature_c"] for r in samples),
                samples=len(samples), own_child_returncode=child.returncode if child else None,
                limits=dict(wall_s=600, workspace_mib=256, min_available_gib=4, temperature_c=80),
                workspace_is_total_memory_limit=False, hardware_commands=0, training_updates=0,
                publish_allowed=False, lock_scope="This project's acceleration checks only; not a global GPU lock")
            (args.output / "build_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            print(json.dumps(receipt))
        if status != "BUILD_PASSED":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
