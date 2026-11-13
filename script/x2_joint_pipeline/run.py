"""Keep the original observe controller and select independent pipeline workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from script.x2_fgp_realtime.assets import file_digest


def validate_parity(directory, fixture):
    receipt = json.loads((directory/"parity.json").read_text())
    guard = json.loads((directory/"guard.json").read_text())
    if (receipt["status"] != "PIPELINE_PREFIX_EQUIVALENCE_PASSED"
            or guard["status"] != receipt["status"] or guard["child_returncode"] != 0
            or receipt["outputs"] != 64 or receipt["fixture_sha256"] != file_digest(fixture)
            or receipt["engine_sha256"] != file_digest(Path("data/analysis/X2_FGP_TRT_ACCEL_20261113/build_gemm_r3/cano.plan"))):
        raise ValueError("Require the completed, same-input prefix gate and clean worker exit")
    reference = Path(receipt["reference"])
    for name, key in (("fgp.jsonl", "reference_timing_sha256"), ("fgp_outputs.npz", "reference_pose_sha256")):
        if file_digest(reference/name) != receipt[key]:
            raise ValueError("Saved serial reference changed")
    return receipt


def select_worker(command):
    changed = list(command)
    if "script.x2_joint_latency.worker" not in changed:
        return changed
    i = changed.index("script.x2_joint_latency.worker")
    if i == 0 or changed[i-1] != "-m" or changed[i+1] not in ("fgp", "policy"):
        raise ValueError("Unexpected owned worker")
    changed[i] = "script.x2_joint_pipeline.worker" if changed[i+1] == "fgp" else "script.x2_joint_perf.worker"
    return changed


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parity", type=Path, required=True)
    a = p.parse_args()
    receipt = validate_parity(a.parity, a.fixture)
    if a.output.exists():
        p.error("Preserve existing outputs")
    original = subprocess.Popen
    def spawn(command, *args, **kwargs):
        changed = select_worker(command)
        if changed != list(command):
            env = dict(kwargs.get("env") or os.environ)
            env["X2_JOINT_PERF_CONFIG"] = json.dumps(dict(variant="all_sleep1_fk", threads=1, spinning=False, batched_fk=True))
            kwargs["env"] = env
        return original(changed, *args, **kwargs)
    subprocess.Popen = spawn
    from script.x2_joint_latency.run import main as joint_main
    sys.argv = [sys.argv[0], "--output", str(a.output), "--fixture", str(a.fixture), "--scheduling", "latest"]
    try:
        joint_main()
    finally:
        subprocess.Popen = original
        if a.output.is_dir():
            (a.output/"pipeline_config.json").write_text(json.dumps(dict(
                variant="pipeline_r1", in_flight_limit=2, ordered_cpu=True, ordered_sender=True,
                drops_after_denoising=0, parity_receipt=receipt, preview_span_ms=180), indent=2)+"\n")


if __name__ == "__main__":
    main()
