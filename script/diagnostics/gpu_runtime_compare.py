"""E30/E31: existing CUDA 11.7 runtime, exact E28 worker, live kernel evidence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import threading
from script.diagnostics import gpu_operator_compare as engine

from script.a3.host_health import append_record, command, sample
from script.a3.idle_reboot_watch import save_json, utc
from script.a3.phase_trace import PhaseTrace, summarize_files
from script.diagnostics.host_loadcheck import FATAL, protection, stop

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/yu/miniconda3/envs/twist/bin/python")
SOURCES = ["script/diagnostics/gpu_runtime_compare.py", "script/diagnostics/gpu_operator_compare.py", "script/a3/phase_trace.py",
           "script/a3/host_health.py", "script/a3/idle_reboot_watch.py",
           "script/diagnostics/host_loadcheck.py"]
CASES = {
    "E30": dict(duration_seconds=0, cuda_launch_blocking="0", extra_syncs=True,
                predecessor=None, variable="EXISTING_RUNTIME_BUNDLE_VERSUS_E27"),
    "E31": dict(duration_seconds=30, cuda_launch_blocking="0", extra_syncs=True,
                predecessor="gpu_runtime_compare_result_20261011_E30.json", variable="REPEAT_DURATION_ONLY_VERSUS_E30_RUNTIME_VERSUS_E28"),
}


def configure(case):
    global CASE, OUTPUT, LOGS
    if case not in CASES:
        raise ValueError("Unknown fixed diagnostic case")
    CASE = case
    OUTPUT = ROOT / ("data/experiments/gpu_runtime_compare_20261011_"+case)
    LOGS = ROOT / ("logs/gpu_runtime_compare_20261011_"+case)
    engine.OUTPUT, engine.LOGS = OUTPUT, LOGS


configure("E30")


def predecessor(case, boot):
    name = CASES[case]["predecessor"]
    if name is None:
        return  # Fresh human authorization after E28; no old job is resumed.
    previous = json.loads((ROOT / "data/manifests" / name).read_text())
    if not predecessor_allowed(previous,boot):
        raise RuntimeError("PREDECESSOR_FAILED_OR_BOOT_CHANGED_NO_ESCALATION")


class KernelFollow:
    def __init__(self, directory, start_epoch):
        self.directory, self.start_epoch = directory, start_epoch
        self.ready = threading.Event()
        self.critical = []
        self.error = None
        self.child = None

    def consume(self):
        try:
            for line in self.child.stdout:
                item = json.loads(line)
                message = item.get("MESSAGE", "")
                timestamp = int(item.get("__REALTIME_TIMESTAMP", "0"))/1000000
                critical = isinstance(message,str) and timestamp >= self.start_epoch and bool(FATAL.search(message))
                append_record(self.directory / "kernel_live.jsonl",dict(at_utc=utc(time.time()),journal=item,critical=critical))
                if critical:
                    self.critical.append(message)
                self.ready.set()
        except BaseException as error:
            self.error = type(error).__name__+":"+str(error)

    def start(self):
        self.stderr = (self.directory / "kernel_live.stderr.log").open("w")
        self.child = subprocess.Popen(["journalctl","-k","-b","-f","-n","3","-o","json","--no-pager"],
                                      stdout=subprocess.PIPE,stderr=self.stderr,text=True,start_new_session=True)
        self.thread = threading.Thread(target=self.consume,daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=8) or self.error or self.child.poll() is not None:
            self.close()
            raise RuntimeError("LIVE_KERNEL_NOT_READY_NO_GPU_LAUNCH")

    def close(self):
        if self.child is not None:
            stop(self.child)
            self.thread.join(timeout=2)
            self.stderr.flush()
            os.fsync(self.stderr.fileno())
            self.stderr.close()


def predecessor_allowed(previous, boot):
    return (previous.get("state",{}).get("status") == "COMPLETED"
            and previous["state"].get("boot_id") == boot
            and not previous.get("trace",{}).get("errors")
            and not previous.get("trace",{}).get("incomplete"))


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
            raise FileExistsError(CASE+" already registered: never reset or repeat")
        health = preflight()
        case = CASES[CASE]
        predecessor(CASE,health["boot_id"])
        predecessor_path = ROOT / "data/manifests" / (case["predecessor"] or "gpu_operator_compare_stop_20261011_E28.json")
        package_paths = list((PYTHON.parent.parent / "lib").glob("python*/site-packages/torch-*.dist-info/METADATA"))
        if len(package_paths)!=1:
            raise RuntimeError("TORCH_METADATA_MISSING_OR_AMBIGUOUS")
        fields=dict(line.split(": ",1) for line in package_paths[0].read_text().splitlines() if line.startswith(("Name: ","Version: ")))
        if fields.get("Version")!="1.13.1+cu117":
            raise RuntimeError("EXISTING_RUNTIME_VERSION_CHANGED")
        job = dict(experiment="GPU_RUNTIME_COMPARE_20261011_"+CASE, case=CASE, boot_id=health["boot_id"],
                   registered_at_utc=utc(time.time()), matrix_size=1024, seed=0, dtype="float32",
                   tf32=False, iterations=1 if case["duration_seconds"]==0 else None,
                   duration_seconds=case["duration_seconds"], duty_target=0.25, minimum_sleep_seconds=0.01,
                   extra_syncs=case["extra_syncs"], cuda_launch_blocking=case["cuda_launch_blocking"],
                   predecessor_sha256=sha(predecessor_path), primary_variable=case["variable"],
                   authorization="USER_CONTINUE_TEST_AFTER_E28_NO_OLD_JOB_RESUME", expected_torch_version="1.13.1+cu117",
                   expected_cuda_version="11.7", torch_metadata_path=str(package_paths[0]),torch_metadata_sha256=sha(package_paths[0]),
                   live_kernel_follow=True, external_mirror_seconds=0.1,
                   idle_observation_seconds=60,
                   worker_timeout_seconds=120, sample_seconds=5, max_cpu_c=90, max_gpu_c=85,
                   consecutive_hot_samples=2, memory_reserve_gib=4, python=str(PYTHON),
                   source_sha256={name:sha(ROOT / name) for name in SOURCES},
                   auto_resume=False, stop_on_first_failure=True, a3_or_isaac=False,
                   backup_status="LOCAL_ONLY", interpretation="DURABLE_LOGGING_PERTURBS_TIMING")
        save_json(OUTPUT / "job.json", job)
        save_json(OUTPUT / "state.json", dict(status="READY", boot_id=health["boot_id"]))
        append_record(LOGS / "health.jsonl", dict(phase="registration", health=health))
        metadata = {name:command(args) for name,args in {
            "gpu": ["nvidia-smi", "-q"], "driver": ["cat", "/proc/driver/nvidia/version"],
            "kernel": ["uname", "-a"], "kernel_before": ["journalctl", "-k", "-b", "--no-pager"]}.items()}
        save_json(LOGS / "metadata.json", metadata)
        save_json(ROOT / ("data/manifests/gpu_runtime_compare_registration_20261011_"+CASE+".json"), job)
        print(json.dumps(job), flush=True)


def worker():
    engine.worker()  # Exact previously registered E28 computation, new output paths only.


def run():
    with (OUTPUT / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read("job.json"), read("state.json")
        health = preflight()
        if not allowed(state, job, health["boot_id"]):
            raise RuntimeError(CASE+" may only launch once on its registered boot")
        predecessor(CASE,health["boot_id"])
        if sha(Path(job["torch_metadata_path"])) != job["torch_metadata_sha256"]:
            raise RuntimeError("TORCH_METADATA_CHANGED")
        for name, digest in job["source_sha256"].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError("SOURCE_CHANGED: " + name)
        start_epoch = time.time()
        state.update(status="RUNNING", started_at_utc=utc(start_epoch))
        save_json(OUTPUT / "state.json", state)  # Consume permission before any GPU import.
        env = os.environ.copy()
        env.update(CUDA_LAUNCH_BLOCKING=job["cuda_launch_blocking"], OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        child, follower = None, None
        high, reason, done_at = 0, None, None
        try:
            follower = KernelFollow(LOGS,start_epoch)
            follower.start()  # Prove live read access before importing CUDA.
            with (LOGS / "worker.console.log").open("w") as console:
                child = subprocess.Popen([job["python"], "-u", "-m", "script.diagnostics.gpu_runtime_compare", CASE, "worker"],
                                         cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
                state["worker_pid"] = child.pid
                save_json(OUTPUT / "state.json", state)
                while True:
                    health = sample()
                    kernel = command(["journalctl", "-k", "-b", "--since", "@"+str(int(start_epoch)),
                                      "--no-pager", "-o", "short-iso-precise"])
                    critical = [line for line in kernel.get("stdout", "").splitlines() if FATAL.search(line)] + list(follower.critical)
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
                    if follower.error or follower.child.poll() is not None:
                        reason = "LIVE_KERNEL_MONITOR_FAILED"
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
                            if result["torch_version"] != job["expected_torch_version"] or result["cuda_version"] != job["expected_cuda_version"]:
                                reason = "RUNTIME_VERSION_MISMATCH"
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
            if follower is not None and hasattr(follower,"stderr") and not follower.stderr.closed:
                follower.close()
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
                   scope="FIXED_COMPARISON_WITH_DURABLE_BOUNDARIES_NOT_ROOT_CAUSE_PROOF")
    save_json(ROOT / ("data/manifests/gpu_runtime_compare_result_20261011_"+CASE+".json"), summary)
    print(json.dumps(summary), flush=True)


def watch():
    """Read-only stream for an external, fsync-on-every-line collector."""
    job = read("job.json")
    offsets, previous = {}, None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        paths = [LOGS / "kernel_live.jsonl"] + list((LOGS / "phases").glob("*.jsonl")) + [LOGS / "health.jsonl", LOGS / "kernel.jsonl"]
        for path in paths:
            if not path.exists():
                continue
            with path.open("rb") as stream:
                stream.seek(offsets.get(str(path),0))
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break  # Retry partial concurrent appends without advancing offset.
                    record = json.loads(line)
                    print(json.dumps(dict(file=str(path.relative_to(ROOT)), byte_offset=offsets.get(str(path),0), record=record)), flush=True)
                    offsets[str(path)] = stream.tell()
        state = read("state.json")
        if state != previous:
            print(json.dumps(dict(file="state.json", record=state)), flush=True)
            previous = state
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot != job["boot_id"] or state["status"] in ("STOPPED", "STOPPED_REBOOT", "COMPLETED"):
            return
        time.sleep(0.1)
    print(json.dumps(dict(watch_status="TIMEOUT_UNKNOWN")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case",choices=list(CASES))
    parser.add_argument("action", choices=["register","run","worker","summarize","watch"])
    args = parser.parse_args()
    configure(args.case)
    globals()[args.action]()
