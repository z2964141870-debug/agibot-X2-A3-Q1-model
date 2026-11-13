"""Bounded real-policy Terminal 2/3 IPC and native C++ adapter check.

Human input and robot feedback are explicitly synthetic. No ROS or HAL is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from script.x2_morph_human.__main__ import adapter, demo_stream
from script.x2_morph_human.artifacts import home, sha
from script.x2_morph_human.client import Client
from .contract import Contract, GROUPS


def read_line(process, timeout=30.):
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError("owned probe process did not reply")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"owned probe process exited ({process.poll()})")
    return json.loads(line)


def finish(process):
    if process is None:
        return
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=60)
    args = p.parse_args()
    if not 30 <= args.frames <= 150:
        p.error("probe is bounded to 30..150 synthetic frames")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = home()
    contract_path = root/"assets/morphmimic_policy/x2.json"
    c = Contract.load(contract_path)
    packets = list(demo_stream(adapter(root), args.frames + 20))
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("C++17 compiler is required")
    binary = out/"native_probe"
    build = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
             "-I", str(root/"cpp_deps"), str(Path(__file__).parent/"cpp/native_probe.cpp"),
             "-o", str(binary)]
    subprocess.run(build, check=True)
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{available.getsockname()[1]}"
    env = dict(os.environ, X2_MORPH_HOME=str(root), PYTHONDONTWRITEBYTECODE="1")
    service = backend = human = None
    results, initial = [], None
    started = time.perf_counter()
    with (out/"policy.stderr.log").open("x") as service_error, (out/"backend.stderr.log").open("x") as backend_error:
        try:
            service = subprocess.Popen([sys.executable, "-B", "-m", "script.x2_morph_human", "serve",
                                        "--endpoint", endpoint], stdout=subprocess.PIPE, stderr=service_error,
                                       text=True, env=env)
            initial = read_line(service)
            if initial.get("status") != "ADVISORY_ONLY":
                raise ValueError("unexpected policy service startup")
            backend = subprocess.Popen([sys.executable, "-B", "-m", "script.x2_morph_backend", "pipe",
                                        "--contract", str(contract_path), "--endpoint", endpoint,
                                        "--output", str(out/"backend.targets.jsonl"), "--rpc-timeout-ms", "1000"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=backend_error,
                                       text=True, env=env)
            if read_line(backend).get("status") != "NO_PUBLISH_BACKEND_READY":
                raise ValueError("unexpected backend startup")
            human = Client(endpoint)
            sequence = 0

            def request_state():
                nonlocal sequence
                state = dict(kind="x2_measurement", version=1, sequence=sequence,
                             joint_names=list(c.names), q_rad=list(c.default), dq_rad_s=[0.]*31,
                             imu_quat_wxyz=[1., 0., 0., 0.], imu_gyro_rad_s=[0., 0., 0.],
                             imu_body="pelvis", body_from_sensor_wxyz=[1., 0., 0., 0.],
                             source_receive_ages_s={k:.001 for k in (*GROUPS, "imu")})
                backend.stdin.write(json.dumps(state) + "\n")
                backend.stdin.flush()
                reply = read_line(backend, 5.)
                if reply["state_sequence"] != sequence or reply["publish_allowed"] is not False:
                    raise ValueError("backend sequence/advisory contract mismatch")
                sequence += 1
                results.append(reply)
                return reply

            def send_human(index):
                packet = dict(packets[index], capture_ns=time.monotonic_ns(), capture_clock="synthetic_probe_capture")
                response = human.request(packet)
                if response.get("accepted_sequence") != index:
                    raise ValueError("human producer acknowledgement mismatch")

            for i in range(args.frames):
                tick = time.monotonic()
                send_human(i)
                request_state()
                time.sleep(max(0., .02 - (time.monotonic() - tick)))
            before_dropout = len(results)
            time.sleep(.18)
            dropout = request_state()
            if dropout["ready"] or "q_target_rad" in dropout:
                raise ValueError("stopped human input must not produce a target")
            for i in range(args.frames, args.frames + 20):
                tick = time.monotonic()
                send_human(i)
                request_state()
                time.sleep(max(0., .02 - (time.monotonic() - tick)))
            ready = [r for r in results if r["ready"]]
            if not ready or not any(r["ready"] for r in results[before_dropout+1:]):
                raise ValueError("no native targets or no recovery after human stream resumed")
            rows = [dict(reply=r["source_advisory"], sequence=r["state_sequence"],
                         joint_names=list(c.names), kp=list(c.kp), kd=list(c.kd)) for r in ready]
            payload = "".join(json.dumps(r) + "\n" for r in rows)
            (out/"native.inputs.jsonl").write_text(payload)
            cpp = subprocess.run([str(binary)], input=payload, text=True, capture_output=True, timeout=10)
            (out/"native.stdout.jsonl").write_text(cpp.stdout)
            (out/"native.stderr.log").write_text(cpp.stderr)
            if cpp.returncode:
                raise RuntimeError(cpp.stderr)
            decoded = [json.loads(line) for line in cpp.stdout.splitlines()]
            if len(decoded) != len(ready):
                raise ValueError("C++ candidate count mismatch")
            for source, target in zip(ready, decoded):
                flat = sum((source["groups"][g]["position_rad"] for g in GROUPS), [])
                if (target["q_target_rad"] != source["q_target_rad"] or flat != source["q_target_rad"]
                        or target["model_kp"] != list(c.kp) or target["model_kd"] != list(c.kd)):
                    raise ValueError("native adapter changed positions or model gain metadata")
            # Only the service created above is terminated. A failed RPC must latch.
            human.close(); human = None
            finish(service)
            failed = request_state()
            latched = request_state()
            if failed.get("reason") != "policy_transport_failed" or latched["ready"]:
                raise ValueError("backend did not latch loss of its policy process")
            backend.stdin.close()
            backend.wait(timeout=5)
            if backend.returncode:
                raise RuntimeError("backend did not exit cleanly")
            times = sorted(r["local_rpc_ms"] for r in results if "local_rpc_ms" in r)
            summary = dict(status="MORPH_TERMINAL23_NO_PUBLISH_PASSED", wall_s=time.perf_counter()-started,
                           fixture="SYNTHETIC_SMPL_FIXED_FEEDBACK", robot_feedback_used=False,
                           requests=len(results), native_candidates=len(ready), native_cpp_exact_match=len(decoded),
                           groups={g:b-a for g,(a,b) in GROUPS.items()}, target_transform_max_difference=0.,
                           human_dropout_rejected=True, human_resume_ready=True, policy_loss_latched=True,
                           reasons=dict(Counter(r.get("reason", "ready") for r in results)),
                           rpc_ms=dict(median=times[len(times)//2], p95=times[min(len(times)-1,int(len(times)*.95))],
                                       maximum=max(times), over_20ms=sum(t>20 for t in times)),
                           robot_hardware_end_to_end_ms=None, model_sha256=ready[0]["policy_sha256"],
                           policy_startup=initial, native_binary_sha256=sha(binary),
                           hardware_commands=0, mc_operations=0, full_sdk_build=False,
                           rpc_timeout_ms_for_probe=1000, candidate_rpc_budget_ms=20.)
            (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
            print(json.dumps(summary, indent=2))
        finally:
            if human is not None:
                human.close()
            finish(backend)
            finish(service)
            for process in (backend, service):
                if process is not None:
                    if process.stdin is not None and not process.stdin.closed:
                        process.stdin.close()
                    if process.stdout is not None:
                        process.stdout.close()


if __name__ == "__main__":
    main()
