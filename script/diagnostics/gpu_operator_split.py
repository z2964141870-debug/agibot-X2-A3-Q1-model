"""E26: one-shot, durable boundaries around the interrupted E24 GPU operation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from script.a3.host_health import append_record, command, sample
from script.a3.idle_reboot_watch import save_json, utc
from script.a3.phase_trace import PhaseTrace, summarize_files
from script.diagnostics.host_loadcheck import FATAL, protection, stop

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/experiments/gpu_operator_split_20261011_E26"
LOGS = ROOT / "logs/gpu_operator_split_20261011_E26"
PYTHON = ROOT / "data/environments/a3-sonic/bin/python"
SOURCES = ["script/diagnostics/gpu_operator_split.py", "script/a3/phase_trace.py",
           "script/a3/host_health.py", "script/a3/idle_reboot_watch.py",
           "script/diagnostics/host_loadcheck.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((OUTPUT / name).read_text())


def allowed(state, job, boot):
    return state.get("status") == "READY" and job["boot_id"] == boot


def preflight():
    health = sample()
    reason, hot = protection(health, 0)
    if reason or hot:
        raise RuntimeError(reason or "INITIAL_HIGH_TEMPERATURE")
    apps = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if apps.get("returncode") != 0 or apps.get("stdout", "").strip():
        raise RuntimeError("GPU_NOT_IDLE_OR_UNREADABLE")
    kernel = command(["journalctl", "-k", "-b", "-n", "20", "--no-pager"])
    if kernel.get("returncode") != 0:
        raise RuntimeError("KERNEL_TELEMETRY_UNREADABLE")
    return health


def register():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "job.json").exists():
            raise FileExistsError("E26 already registered: never reset or repeat")
        health = preflight()
        job = dict(experiment="GPU_OPERATOR_SPLIT_20261011_E26", boot_id=health["boot_id"],
                   registered_at_utc=utc(time.time()), matrix_size=1024, seed=0, dtype="float32",
                   tf32=False, iterations=1, cuda_launch_blocking="1", idle_observation_seconds=60,
                   worker_timeout_seconds=120, sample_seconds=5, max_cpu_c=90, max_gpu_c=85,
                   consecutive_hot_samples=2, memory_reserve_gib=4, python=str(PYTHON),
                   source_sha256={name:sha(ROOT / name) for name in SOURCES},
                   auto_resume=False, stop_on_first_failure=True, a3_or_isaac=False,
                   backup_status="LOCAL_ONLY", interpretation="BLOCKING_AND_EXTRA_SYNCS_CHANGE_TIMING")
        save_json(OUTPUT / "job.json", job)
        save_json(OUTPUT / "state.json", dict(status="READY", boot_id=health["boot_id"]))
        append_record(LOGS / "health.jsonl", dict(phase="registration", health=health))
        metadata = {name:command(args) for name,args in {
            "gpu": ["nvidia-smi", "-q"], "driver": ["cat", "/proc/driver/nvidia/version"],
            "kernel": ["uname", "-a"], "kernel_before": ["journalctl", "-k", "-b", "--no-pager"]}.items()}
        save_json(LOGS / "metadata.json", metadata)
        save_json(ROOT / "data/manifests/gpu_operator_split_registration_20261011_E26.json", job)
        print(json.dumps(job), flush=True)


def worker():
    job, state = read("job.json"), read("state.json")
    if state["status"] != "RUNNING" or sample()["boot_id"] != job["boot_id"]:
        raise RuntimeError("WORKER_NOT_AUTHORIZED")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") != "1":
        raise RuntimeError("BLOCKING_ENV_MISSING")
    trace = PhaseTrace(LOGS / "phases", context=lambda:dict(experiment=job["experiment"], iteration=0))
    def call(name, function):
        # Automatic sync is disabled: each explicit synchronization is its own span.
        with trace.span(name, cuda_sync=False):
            return function()
    try:
        def import_torch():
            import torch
            return torch
        torch = call("TORCH_IMPORT", import_torch)
        trace.emit("METADATA", "RUNTIME", torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   cuda_launch_blocking=os.environ["CUDA_LAUNCH_BLOCKING"])
        call("CPU_SEED_AND_THREADS", lambda:(torch.set_num_threads(1), torch.manual_seed(job["seed"])))
        torch.backends.cuda.matmul.allow_tf32 = False
        call("CUDA_INIT", torch.cuda.init)
        trace.emit("METADATA", "GPU", name=torch.cuda.get_device_name(0))
        n = job["matrix_size"]
        a = call("RANDN_ALLOCATION", lambda:torch.randn(n,n,device="cuda",dtype=torch.float32))
        b = call("IDENTITY_ALLOCATION", lambda:torch.eye(n,device="cuda",dtype=torch.float32))
        call("ALLOCATION_SYNCHRONIZE", torch.cuda.synchronize)
        value = call("MATMUL", lambda:a @ b)
        call("MATMUL_SYNCHRONIZE", torch.cuda.synchronize)
        equal = call("ALLCLOSE", lambda:torch.allclose(a,value,atol=1e-5,rtol=1e-5))
        trace.emit("VALUE", "ALLCLOSE_RESULT", value=equal)
        if not equal:
            raise ValueError("IDENTITY_MATRIX_MISMATCH")
        call("ALLCLOSE_SYNCHRONIZE", torch.cuda.synchronize)
        finite = call("ISFINITE", lambda:torch.isfinite(value))
        call("ISFINITE_SYNCHRONIZE", torch.cuda.synchronize)
        reduced = call("ALL_REDUCTION", finite.all)
        call("ALL_REDUCTION_SYNCHRONIZE", torch.cuda.synchronize)
        scalar = call("SCALAR_READBACK", reduced.item)
        trace.emit("VALUE", "FINITE_RESULT", value=scalar)
        if not scalar:
            raise ValueError("NONFINITE_MATRIX")
        call("FINAL_SYNCHRONIZE", torch.cuda.synchronize)
        result = dict(status="PASSED", at_utc=utc(time.time()), boot_id=job["boot_id"],
                      matrix_size=n, iterations=1, allclose=equal, all_finite=scalar,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      cuda_launch_blocking="1")
        call("RESULT_SAVE", lambda:save_json(OUTPUT / "worker_result.json", result))
        trace.emit("COMPLETE", "WORKER_COMPLETE")
    except BaseException as error:
        trace.emit("ERROR", "WORKER_FAILED", error_type=type(error).__name__, error=str(error)[:1000])
        raise


def run():
    with (OUTPUT / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read("job.json"), read("state.json")
        health = preflight()
        if not allowed(state, job, health["boot_id"]):
            raise RuntimeError("E26 may only launch once on its registered boot")
        for name, digest in job["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError("SOURCE_CHANGED: " + name)
        start_epoch = time.time()
        state.update(status="RUNNING", started_at_utc=utc(start_epoch))
        save_json(OUTPUT / "state.json", state)  # Consume permission before any GPU import.
        env = os.environ.copy()
        env.update(CUDA_LAUNCH_BLOCKING="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        child = None
        high, reason, done_at = 0, None, None
        try:
            with (LOGS / "worker.console.log").open("w") as console:
                child = subprocess.Popen([job["python"], "-u", "-m", "script.diagnostics.gpu_operator_split", "worker"],
                                         cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                state["worker_pid"] = child.pid
                save_json(OUTPUT / "state.json", state)
                while True:
                    health = sample()
                    kernel = command(["journalctl", "-k", "-b", "--since", "@"+str(int(start_epoch)),
                                      "--no-pager", "-o", "short-iso-precise"])
                    critical = [line for line in kernel.get("stdout", "").splitlines() if FATAL.search(line)]
                    append_record(LOGS / "kernel.jsonl", dict(at_utc=utc(time.time()), query=kernel))
                    append_record(LOGS / "health.jsonl", dict(phase=state["status"], health=health, critical_kernel_lines=critical))
                    os.fsync(console.fileno())
                    reason, high = protection(health, high)
                    if health["boot_id"] != job["boot_id"]:
                        reason = "BOOT_CHANGED"
                    if critical:
                        reason = "CRITICAL_KERNEL_EVENT"
                    if kernel.get("returncode") != 0:
                        reason = "KERNEL_TELEMETRY_UNREADABLE"
                    if reason:
                        break
                    if child.poll() is not None:
                        if child.returncode != 0:
                            reason = "WORKER_FAILED_"+str(child.returncode)
                            break
                        if done_at is None:
                            result = read("worker_result.json")
                            if result["status"] != "PASSED" or result["boot_id"] != job["boot_id"]:
                                reason = "INVALID_WORKER_RESULT"
                                break
                            done_at = time.monotonic()
                            state.update(status="OBSERVING", worker_completed_at_utc=utc(time.time()))
                            save_json(OUTPUT / "state.json", state)
                        if time.monotonic() - done_at >= job["idle_observation_seconds"]:
                            break
                    elif time.time() - start_epoch >= job["worker_timeout_seconds"]:
                        reason = "WORKER_TIMEOUT"
                        break
                    time.sleep(job["sample_seconds"])
        except BaseException as error:
            reason = "SUPERVISOR_ERROR:"+type(error).__name__+":"+str(error)[:500]
        finally:
            if child is not None:
                stop(child)
            state.update(status="STOPPED" if reason else "COMPLETED", reason=reason,
                         ended_at_utc=utc(time.time()), worker_exit=child.returncode if child else None)
            save_json(OUTPUT / "state.json", state)
        summarize()
        if reason:
            raise RuntimeError(reason)


def summarize():
    job, state = read("job.json"), read("state.json")
    health = sample()  # Read-only telemetry; never imports CUDA.
    if health["boot_id"] != job["boot_id"]:
        state.update(status="STOPPED_REBOOT", reason="CONFIRMED_NEW_BOOT_NO_RESUME",
                     confirmed_at_utc=health["at_utc"], observed_boot=health["boot_id"])
        save_json(OUTPUT / "state.json", state)
    records = [json.loads(line) for line in (LOGS / "health.jsonl").read_text().splitlines()]
    cpu = [r["health"].get("temperatures_c",{}).get("x86_pkg_temp") for r in records]
    gpu = [g["temperature.gpu"] for r in records for g in r["health"].get("gpus",[])]
    result_path = OUTPUT / "worker_result.json"
    summary = dict(experiment=job["experiment"], at_utc=utc(time.time()), state=state,
                   trace=summarize_files(LOGS / "phases"), current_health=health,
                   samples=len(records), first_health=records[0], last_health=records[-1],
                   max_cpu_c=max(x for x in cpu if x is not None), max_gpu_c=max(gpu),
                   min_available_memory_bytes=min(r["health"]["memory_bytes"]["MemAvailable"] for r in records),
                   worker_result=read("worker_result.json") if result_path.exists() else None,
                   worker_result_sha256=sha(result_path) if result_path.exists() else None,
                   source_sha256=job["source_sha256"], backup_status="LOCAL_ONLY",
                   scope="ONE_BLOCKING_ITERATION_WITH_EXTRA_SYNCS_NOT_ROOT_CAUSE_PROOF")
    save_json(ROOT / "data/manifests/gpu_operator_split_result_20261011_E26.json", summary)
    print(json.dumps(summary), flush=True)


def watch():
    """Read-only stream for an external, fsync-on-every-line collector."""
    job = read("job.json")
    sent, previous = {}, None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        paths = list((LOGS / "phases").glob("*.jsonl")) + [LOGS / "health.jsonl", LOGS / "kernel.jsonl"]
        for path in paths:
            if not path.exists():
                continue
            lines = path.read_text().splitlines()
            for i,line in enumerate(lines[sent.get(str(path),0):],sent.get(str(path),0)):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # A concurrently appended final line can be incomplete; retry it.
                print(json.dumps(dict(file=str(path.relative_to(ROOT)), line_number=i+1, record=record)), flush=True)
                sent[str(path)] = i+1
        state = read("state.json")
        if state != previous:
            print(json.dumps(dict(file="state.json", record=state)), flush=True)
            previous = state
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot != job["boot_id"] or state["status"] in ("STOPPED", "STOPPED_REBOOT", "COMPLETED"):
            return
        time.sleep(1)
    print(json.dumps(dict(watch_status="TIMEOUT_UNKNOWN")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register","run","worker","summarize","watch"])
    globals()[parser.parse_args().action]()
