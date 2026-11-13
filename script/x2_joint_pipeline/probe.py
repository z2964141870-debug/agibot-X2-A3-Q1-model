"""Resource-guarded 64-output pipeline comparison with the saved serial prefix."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from script.x2_fgp_accel.build import resources, require_resources, stop_child


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve prior outputs")
    require_resources(resources())
    if shutil.disk_usage('.').free < 10*1024**3:
        raise RuntimeError("Less than 10GiB disk")
    child, samples = None, []
    status, error = "PIPELINE_PREFIX_FAILED", None
    with Path("data/locks/x2_fgp_accel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        a.output.mkdir(parents=True)
        began = time.monotonic()
        try:
            with (a.output/"worker.log").open("x") as f:
                child = subprocess.Popen([sys.executable,"-B","-m","script.x2_joint_pipeline.worker","fgp",
                    "--endpoint","parity_no_network", "--output",str(a.output),"--fixture",str(a.fixture),
                    "--source-root","data/runtime/clotho_fgp_realtime_20261113",
                    "--build-root","data/analysis/X2_FGP_TRT_ACCEL_20261113/build_gemm_r3",
                    "--parity-reference",str(a.reference)], stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                while child.poll() is None:
                    snapshot=resources();samples.append(snapshot);require_resources(snapshot)
                    if time.monotonic()-began > 100:
                        raise TimeoutError("100s prefix gate limit")
                    time.sleep(.5)
            if child.returncode or json.loads((a.output/"parity.json").read_text())["status"] != "PIPELINE_PREFIX_EQUIVALENCE_PASSED":
                raise RuntimeError("Prefix gate failed; preserve partial evidence")
            status="PIPELINE_PREFIX_EQUIVALENCE_PASSED"
        except BaseException as exc:
            error=repr(exc)
        finally:
            stop_child(child)
            receipt=dict(status=status,error=error,wall_s=time.monotonic()-began,
                child_pid=child.pid if child else None,child_returncode=child.returncode if child else None,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),resources=samples,
                gpu_frame_bound=64,wall_bound_s=100,hardware_commands=0,training_updates=0)
            (a.output/"guard.json").write_text(json.dumps(receipt,indent=2)+"\n")
            print(json.dumps({k:v for k,v in receipt.items() if k!="resources"}))
    if status != "PIPELINE_PREFIX_EQUIVALENCE_PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
