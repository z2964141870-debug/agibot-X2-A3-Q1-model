"""One-shot E23 ordinary CUDA control; no Isaac, A3 weights or restart."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from script.a3.host_health import append_record, command, sample
from script.a3.idle_reboot_watch import save_json, utc
from script.diagnostics.host_loadcheck import FATAL, protection, stop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/experiments/a3_isolation_20261011_E23"
LOG = ROOT / "logs/a3_isolation_20261011_E23"
PYTHON = ROOT / "data/environments/a3-sonic/bin/python"
SOURCES = ("script/diagnostics/gpu_isolation.py", "script/diagnostics/host_loadcheck.py",
           "script/a3/host_health.py", "script/a3/idle_reboot_watch.py")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed(state, job, boot):
    return state["status"] == "READY" and job["boot_id"] == boot


def register():
    if (OUT / "job.json").exists() or (OUT / "state.json").exists():
        raise FileExistsError("Existing experiment cannot be reset")
    health = sample()
    reason, hot = protection(health, 0)
    if reason or hot:
        raise RuntimeError(reason or "HOT_PREFLIGHT")
    gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    kernel = command(["journalctl", "-k", "-b", "-n", "1", "--no-pager"])
    if gpu["returncode"] != 0 or gpu.get("stdout") or kernel["returncode"] != 0:
        raise RuntimeError("Compute occupied or telemetry unreadable")
    stages = [dict(kind="matrix", size=256, duty=0.1),
              dict(kind="matrix", size=1024, duty=0.25)]
    stages += [dict(kind=k, size=32, duty=0.25) for k in ("forward", "backward", "optimizer")]
    job = dict(experiment="A3_ISOLATION_20261011_E23", boot_id=health["boot_id"],
               registered_at_utc=utc(time.time()), stages=stages, durations=[30, 120, 600],
               cooldown_seconds=300, sample_seconds=5, seed=0, dtype="float32",
               max_cpu_c=90, max_gpu_c=85, consecutive_hot_samples=2, memory_reserve_gib=4,
               auto_resume=False, stop_on_first_fault=True, repeat_allowed=False,
               isaac_import=False, a3_weights=False, backup_status="LOCAL_ONLY",
               sources_sha256={p:sha(ROOT/p) for p in SOURCES},
               python=str(PYTHON), python_resolved=str(PYTHON.resolve()))
    save_json(OUT / "job.json", job)
    save_json(OUT / "state.json", dict(status="READY", results=[]))
    append_record(LOG / "health.jsonl", dict(stage="registration", health=health))
    save_json(ROOT / "data/manifests/a3_isolation_registration_20261011_E23.json", job)
    print(json.dumps(job), flush=True)


def marker(name, phase, **details):
    append_record(LOG / (name + ".phases.jsonl"),
                  dict(at_utc=utc(time.time()), phase=phase, **details))


def worker(index, seconds):
    job = read(OUT / "job.json")
    spec = job["stages"][index]
    name = f"{index}_{spec['kind']}_{seconds}s"
    state = read(OUT / "state.json")
    if (state["status"] != "RUNNING" or state.get("current_stage") != name
            or seconds not in job["durations"]
            or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != job["boot_id"]):
        raise RuntimeError("Worker requires live registered supervisor on original boot")
    marker(name, "TORCH_IMPORT_BEGIN")
    import torch
    marker(name, "TORCH_IMPORT_END", version=torch.__version__, cuda_version=torch.version.cuda)
    torch.set_num_threads(1)
    torch.manual_seed(job["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    marker(name, "CUDA_INIT_BEGIN")
    torch.cuda.init()
    torch.cuda.synchronize()
    marker(name, "CUDA_INIT_END", device=torch.cuda.get_device_name(0))
    marker(name, "ALLOCATION_BEGIN")
    if spec["kind"] == "matrix":
        n = spec["size"]
        a = torch.randn(n, n, device="cuda")
        b = torch.eye(n, device="cuda")
        def operation():
            value = a @ b
            if not torch.allclose(a, value, atol=1e-5, rtol=1e-5):
                raise ValueError("Matrix identity check failed")
            return value
    else:
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                    torch.nn.Linear(16, 4)).cuda()
        x = torch.randn(8, 3, 32, 32, device="cuda")
        target = torch.zeros(8, 4, device="cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) if spec["kind"] == "optimizer" else None
        initial = [p.detach().clone() for p in model.parameters()]
        def operation():
            model.zero_grad(set_to_none=True)
            if spec["kind"] == "forward":
                with torch.no_grad():
                    value = model(x)
            else:
                value = model(x)
                loss = (value - target).square().mean()
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Nonfinite gradient")
                if optimizer:
                    optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite parameter")
            return value
    torch.cuda.synchronize()
    marker(name, "ALLOCATION_END")
    start = time.monotonic()
    count = 0
    active = 0.0
    last_marker = start
    while time.monotonic() - start < seconds:
        before = time.monotonic()
        if count == 0:
            marker(name, spec["kind"].upper() + "_FIRST_OPERATION_BEGIN")
        value = operation()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite output")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - before
        active += elapsed
        count += 1
        if count == 1 or time.monotonic() - last_marker >= 5:
            marker(name, "SYNCHRONIZED_OPERATION_END", iterations=count)
            last_marker = time.monotonic()
        time.sleep(max(0.01, elapsed * (1/spec["duty"] - 1)))
    if spec["kind"] != "matrix":
        changed = any(not torch.equal(p, old) for p, old in zip(model.parameters(), initial))
        if changed != (spec["kind"] == "optimizer"):
            raise ValueError("Unexpected parameter update behavior")
    result = dict(status="PASSED", iterations=count, elapsed_seconds=time.monotonic()-start,
                  active_seconds=active, optimizer_updates=count if spec["kind"] == "optimizer" else 0,
                  spec=spec, at_utc=utc(time.time()))
    save_json(OUT / (name + ".result.json"), result)
    marker(name, "WORKER_COMPLETE", **result)


def check(job, stage, started, hot):
    health = sample()
    kernel = command(["journalctl", "-k", "-b", "--since", f"@{int(started)}", "--no-pager", "-o", "cat"])
    critical = [x for x in kernel.get("stdout", "").splitlines() if FATAL.search(x)]
    append_record(LOG / "health.jsonl", dict(stage=stage, health=health,
                  kernel_readable=kernel["returncode"] == 0, critical_kernel_lines=critical))
    reason, hot = protection(health, hot)
    if health["boot_id"] != job["boot_id"]:
        reason = "CONFIRMED_NEW_BOOT"
    elif kernel["returncode"] != 0:
        reason = "KERNEL_TELEMETRY_UNREADABLE"
    elif critical:
        reason = "CRITICAL_KERNEL_EVENT"
    if reason:
        raise RuntimeError(reason)
    return hot


def run():
    with (OUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job, state = read(OUT / "job.json"), read(OUT / "state.json")
        if not allowed(state, job, sample()["boot_id"]):
            raise RuntimeError("No repeat or reboot recovery")
        for p, digest in job["sources_sha256"].items():
            if sha(ROOT / p) != digest:
                raise RuntimeError("Registered source changed: " + p)
        state.update(status="RUNNING", started_at_utc=utc(time.time()))
        save_json(OUT / "state.json", state)
        def interrupted(signum, frame):
            raise RuntimeError("OPERATOR_OR_SERVICE_STOP")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        child = None
        try:
            for index, spec in enumerate(job["stages"]):
                for seconds in job["durations"]:
                    name = f"{index}_{spec['kind']}_{seconds}s"
                    state.update(current_stage=name)
                    save_json(OUT / "state.json", state)
                    gpu = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
                    if gpu["returncode"] != 0 or gpu.get("stdout"):
                        raise RuntimeError("GPU_OCCUPIED_OR_UNREADABLE")
                    started, hot = time.time(), 0
                    check(job, name, started, hot)
                    health = sample()
                    reason, hot = protection(health, 0)
                    if reason or hot:
                        raise RuntimeError(reason or "HOT_PREFLIGHT")
                    with (LOG / (name + ".console.log")).open("w") as stream:
                        env = os.environ.copy()
                        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                        child = subprocess.Popen([str(PYTHON), "-m", "script.diagnostics.gpu_isolation",
                                                  "worker", "--index", str(index), "--seconds", str(seconds)],
                                                 cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        while True:
                            hot = check(job, name, started, hot)
                            if child.poll() is not None:
                                break
                            if time.time() - started > seconds + 90:
                                raise RuntimeError("WORKER_TIMEOUT")
                            time.sleep(5)
                        if child.returncode != 0:
                            raise RuntimeError("WORKER_FAILED_" + str(child.returncode))
                    result_path = OUT / (name + ".result.json")
                    result = read(result_path)
                    if result["status"] != "PASSED" or result["elapsed_seconds"] < seconds:
                        raise RuntimeError("MISSING_OR_INVALID_RESULT")
                    state["results"].append(dict(stage=name, **result, result_sha256=sha(result_path)))
                    save_json(OUT / "state.json", state)
                    child = None
                if index < len(job["stages"])-1:
                    state.update(status="COOLDOWN", current_stage=f"cooldown_after_{index}")
                    save_json(OUT / "state.json", state)
                    until = time.monotonic() + job["cooldown_seconds"]
                    started, hot = time.time(), 0
                    while time.monotonic() < until:
                        hot = check(job, state["current_stage"], started, hot)
                        time.sleep(min(5, max(0, until-time.monotonic())))
                    state["status"] = "RUNNING"
            state.update(status="COMPLETED")
        except BaseException as error:
            state.update(status="STOPPED", reason=repr(error))
            raise
        finally:
            if child is not None:
                stop(child)
            state["ended_at_utc"] = utc(time.time())
            save_json(OUT / "state.json", state)


def summarize():
    job, state = read(OUT / "job.json"), read(OUT / "state.json")
    health = sample()
    if health["boot_id"] != job["boot_id"]:
        state.update(status="STOPPED_REBOOT", reason="CONFIRMED_NEW_BOOT", observed_boot=health["boot_id"])
        save_json(OUT / "state.json", state)
    rows = [json.loads(x) for x in (LOG / "health.jsonl").read_text().splitlines()]
    cpu = [x["health"]["temperatures_c"].get("x86_pkg_temp") for x in rows]
    gpu = [g.get("temperature.gpu") for x in rows for g in x["health"]["gpus"]]
    result = dict(experiment=job["experiment"], state=state, observed_at_utc=utc(time.time()),
                  initial_boot=job["boot_id"], observed_boot=health["boot_id"], samples=len(rows),
                  cpu_max_c=max(x for x in cpu if x is not None),
                  gpu_max_c=max(x for x in gpu if x is not None), backup_status="LOCAL_ONLY",
                  job_sha256=sha(OUT / "job.json"), health_sha256=sha(LOG / "health.jsonl"))
    save_json(ROOT / "data/manifests/a3_isolation_result_20261011_E23.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "worker", "summarize"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--seconds", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.index, args.seconds)
    else:
        dict(register=register, run=run, summarize=summarize)[args.action]()
