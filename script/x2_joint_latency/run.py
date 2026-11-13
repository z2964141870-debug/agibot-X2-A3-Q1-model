"""Bounded same-SoC FGP replay + unchanged native observe backend. Never arms."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from script.x2_fgp_accel.build import resources, require_resources
from script.x2_fgp_realtime.assets import file_digest


def read_ready(child, timeout=40):
    if not select.select([child.stdout], [], [], timeout)[0]:
        raise TimeoutError("Owned process initialization timeout")
    line = child.stdout.readline()
    if not line:
        raise RuntimeError("Owned process exited during initialization")
    return json.loads(line)


def finish(child):
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--scheduling", choices=("fifo", "latest"), default="fifo")
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve previous outputs")
    require_resources(resources())
    if shutil.disk_usage('.').free < 10*1024**3:
        raise RuntimeError("Less than 10GiB free disk")
    root = Path.cwd()
    binary = root / "data/analysis/X2_MORPH_NATIVE_BUILD_R6_20261113/x2_morph_native"
    profile = root / "data/analysis/X2_MORPH_NATIVE_SENSOR_V2_20261113.json"
    if not binary.is_file() or not profile.is_file():
        raise ValueError("Existing observe binary/profile unavailable")
    from script.x2_morph_backend.sensor_profile import validate
    validate(json.loads(profile.read_text()))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        endpoint = "tcp://127.0.0.1:"+str(available.getsockname()[1])
    lock_path = root / "data/locks/x2_fgp_accel.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    children, logs, snapshots = [], [], []
    status, error = "JOINT_TEST_FAILED", None
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        began = time.monotonic()
        def spawn(command, name, env=None):
            log = (a.output / (name+".stderr.log")).open("x")
            logs.append(log)
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=log, text=True, env=env)
            children.append(child)
            return child
        try:
            policy = spawn([str(root/"data/environments/orin-py310/bin/python"), "-B", "-m",
                "script.x2_joint_latency.worker", "policy", "--endpoint", endpoint,
                "--output", str(a.output)], "policy")
            if read_ready(policy).get("status") != "ADVISORY_ONLY":
                raise RuntimeError("Unexpected policy initialization")
            fgp = spawn([sys.executable, "-B", "-m", "script.x2_joint_latency.worker", "fgp",
                "--endpoint", endpoint, "--output", str(a.output), "--fixture", str(a.fixture),
                "--source-root", str(root/"data/runtime/clotho_fgp_realtime_20261113"),
                "--build-root", str(root/"data/analysis/X2_FGP_TRT_ACCEL_20261113/build_gemm_r3"),
                "--scheduling", a.scheduling], "fgp")
            startup = read_ready(fgp)
            if startup.get("status") != "FGP_READY":
                raise RuntimeError("Unexpected FGP initialization")
            native_env = dict(os.environ)
            native_env["LD_LIBRARY_PATH"] = "/home/agi/aimdk_msgs_pc2/lib:/opt/ros/humble/lib:/agibot/software/common/lib:"+os.environ.get("LD_LIBRARY_PATH", "")
            native_env["AMENT_PREFIX_PATH"] = "/home/agi/aimdk_msgs_pc2:/opt/ros/humble:/agibot/software/common"
            native = spawn([str(binary), "--profile", str(profile), "--output", str(a.output/"native"),
                "--endpoint", endpoint, "--mode", "observe", "--run-id", "j"+uuid.uuid4().hex[:10],
                "--seconds", "55"], "native", native_env)
            fgp.stdin.write(str(time.monotonic_ns()+200_000_000)+"\n")
            fgp.stdin.flush()
            while fgp.poll() is None:
                if policy.poll() is not None or native.poll() is not None:
                    raise RuntimeError("Policy or observe backend exited during replay")
                snapshot = dict(monotonic_ns=time.monotonic_ns(), **resources())
                snapshots.append(snapshot)
                require_resources(snapshot)
                if time.monotonic()-began > 100:
                    raise TimeoutError("100s joint test wall limit")
                time.sleep(.5)
            if fgp.returncode:
                raise RuntimeError("FGP worker failed; preserve stderr and partial tapes")
            time.sleep(.1)
            finish(native)
            final = json.loads((a.output/"native/status.json").read_text())
            if final["hardware_commands"] or final["sdk_publish_calls"]:
                raise RuntimeError("Observe invariant violated")
            status = "JOINT_RECORDING_COMPLETED"
        except BaseException as exc:
            error = repr(exc)
        finally:
            for child in reversed(children):
                finish(child)
                if child.stdin:
                    child.stdin.close()
                if child.stdout:
                    child.stdout.close()
            for log in logs:
                log.close()
            receipt = dict(status=status, error=error, wall_s=time.monotonic()-began,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                native_binary_sha256=file_digest(binary), profile_sha256=file_digest(profile),
                fixture_sha256=file_digest(a.fixture), children=[dict(pid=c.pid, returncode=c.returncode) for c in children],
                resources=snapshots, hardware_commands=0, mc_mutations=0, new_ble_connections=0,
                robot_feedback="LIVE_ROS_READ_ONLY", human_input="CONTIGUOUS_RECORDED_IMU_AT_IDEAL_30HZ",
                physical_end_to_end_ms=None, training_updates=0,
                wall_limit_s=100, min_available_gib=4, max_temperature_c=80,
                publish_allowed=False)
            receipt["scheduling"] = a.scheduling
            (a.output/"run_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
            print(json.dumps({k:v for k,v in receipt.items() if k != "resources"}))
    if status != "JOINT_RECORDING_COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
