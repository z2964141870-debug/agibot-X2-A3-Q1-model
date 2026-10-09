"""Run the pinned official Isaac evaluation with the existing thermal guard."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.autoresume import ROOT, VENDOR_COMMIT, monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.host_health import command, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    checkpoint, output, logs = args.checkpoint.resolve(), args.output.resolve(), args.logs.resolve()
    if (not checkpoint.is_relative_to(ROOT / "data/training") or
            not output.is_relative_to(ROOT / "data/evaluation") or
            not logs.is_relative_to(ROOT / "logs")):
        parser.error("Paths must stay inside the approved project directories")
    if output.exists() or logs.exists():
        raise FileExistsError("Use a new evaluation ID")
    vendor = ROOT / "script/vendor/sonic_for_a3"
    if command(["git", "-C", str(vendor), "rev-parse", "HEAD"])["stdout"] != VENDOR_COMMIT:
        raise ValueError("Vendor version changed")
    gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if gpu["returncode"] != 0 or gpu["stdout"]:
        raise RuntimeError("GPU unavailable or occupied")
    health = sample()
    if (health["temperatures_c"].get("x86_pkg_temp", 90) >= 90 or
            health["gpu_query"]["returncode"] != 0 or not health["gpus"] or
            any((g.get("temperature.gpu") or 85) >= 85 for g in health["gpus"])):
        raise RuntimeError("Cannot verify safe starting temperatures")
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    if sha256(checkpoint) != metadata["sha256"] or checkpoint.stat().st_size != metadata["bytes"]:
        raise ValueError("Checkpoint integrity mismatch")
    logs.mkdir(parents=True)
    argv = [str(ROOT / "data/environments/a3-sonic/bin/python"), "-m", "gear_sonic.evaluation", "run",
            "--config", str(ROOT / "script/a3/isaac_eval.yaml"), "--checkpoint", str(checkpoint),
            "--dataset", str(ROOT / "data/training/a3_20261009/motionlib/agibot_a3"),
            "--output", str(output)]
    record = {"checkpoint": str(checkpoint), "checkpoint_sha256": metadata["sha256"],
              "vendor_commit": VENDOR_COMMIT, "argv": argv, "started_at": time.time()}
    atomic_json(logs / "execution.json", record)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONPATH=str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
                       WANDB_MODE="disabled", HYDRA_FULL_ERROR="1")
    guard = {"max_cpu_c": 90, "max_gpu_c": 85, "sample_seconds": 5, "expires_at": time.time() + 900}
    with (logs / "console.log").open("w") as stream:
        child = subprocess.Popen(argv, cwd=vendor, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        code, reason = monitor_child(child, logs, guard)
    # Isaac can exit zero after a callback exception; verify the actual outputs.
    if code == 0 and reason is None:
        validation_argv = [argv[0], "-m", "gear_sonic.evaluation", "validate", "--output", str(output)]
        try:
            with (logs / "validation.log").open("w") as stream:
                validation = subprocess.run(validation_argv, cwd=vendor, env=environment,
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            record["validation_returncode"] = validation.returncode
            if validation.returncode != 0:
                reason = "evaluation_output_invalid"
        except subprocess.TimeoutExpired:
            reason = "evaluation_validation_timeout"
    record.update(returncode=code, reason=reason, finished_at=time.time())
    atomic_json(logs / "execution.json", record)
    print(json.dumps(record), flush=True)
    return 0 if code == 0 and reason is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
