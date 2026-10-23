#!/usr/bin/env python3
"""One finite MC SIGSTOP/SIGCONT diagnostic; no ROS or HAL publication."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def emit(event, **values):
    print(json.dumps({
        "event": event,
        "utc": datetime.now(timezone.utc).isoformat(),
        "mono_s": time.monotonic(),
        **values,
    }), flush=True)


def snapshot(pid):
    root = Path(f"/proc/{pid}")
    fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
    argv = (root / "cmdline").read_bytes()
    return {
        "pid": pid,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "start_ticks": int(fields[19]),
        "uid": root.stat().st_uid,
        "argv_sha256": hashlib.sha256(argv).hexdigest(),
        "argv0": argv.split(b"\0", 1)[0].decode(),
        "state": fields[0],
    }


class Process:
    def __init__(self, expected):
        self.expected = expected
        self.fd = os.pidfd_open(expected["pid"])
        try:
            self.state()
        except BaseException:
            os.close(self.fd)
            raise

    def state(self):
        current = snapshot(self.expected["pid"])
        for field in ("pid", "boot_id", "start_ticks", "uid", "argv_sha256"):
            if current[field] != self.expected[field]:
                raise RuntimeError(f"process identity changed: {field}")
        return current["state"]

    def send(self, sig):
        signal.pidfd_send_signal(self.fd, sig)

    def close(self):
        os.close(self.fd)


def bounded_pause(target, seconds, *, clock=time.monotonic,
                  sleep=time.sleep, log=emit, cancelled=lambda: False):
    if not math.isfinite(seconds) or not 0 < seconds <= 120.0:
        raise ValueError("pause duration must be finite and in (0, 120] seconds")
    initial = target.state()
    if initial in ("T", "t", "Z", "X"):
        raise RuntimeError(f"target is not initially running: {initial}")
    log("validated", state=initial, planned_seconds=seconds)
    if cancelled():
        raise RuntimeError("cancelled before STOP; no signal sent")

    # The same local process owns STOP and CONT; no separately armed early timer.
    started = clock()
    try:
        target.send(signal.SIGSTOP)
        deadline = min(started + seconds, started + 0.5)
        while target.state() not in ("T", "t"):
            if cancelled() or clock() >= deadline:
                raise RuntimeError("STOP was not acknowledged within deadline")
            sleep(0.005)
        log("stop_confirmed", elapsed_s=clock() - started)
        while not cancelled() and clock() < started + seconds:
            sleep(min(0.01, started + seconds - clock()))
    finally:
        target.send(signal.SIGCONT)
        continued = clock()
        deadline = continued + 1.0
        state = target.state()
        while state in ("T", "t") and clock() < deadline:
            sleep(0.005)
            state = target.state()
        if state in ("T", "t", "Z", "X"):
            raise RuntimeError(f"CONT did not restore a running process: {state}")
        log("resume_confirmed", state=state, elapsed_s=clock() - started,
            stop_to_cont_s=continued - started, cancelled=cancelled())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--boot-id")
    parser.add_argument("--start-ticks", type=int)
    parser.add_argument("--argv-sha256")
    parser.add_argument("--execute", choices=("SUSPENDED_MC_PAUSE_2S",))
    parser.add_argument("--pause-seconds", type=float, default=2.0,
                        help="bounded pause length (default 2.0s; auto-CONT always applies)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        parser.error("Linux pidfd APIs are required; no PID-only fallback")
    if args.self_test and any((args.pid, args.execute, args.boot_id,
                              args.start_ticks, args.argv_sha256)):
        parser.error("self-test cannot be combined with target or execute options")

    child = None
    if args.self_test:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 start_new_session=True)
        expected = snapshot(child.pid)
        seconds = 0.1
        emit("self_test_child", identity=expected)
    else:
        if not all((args.pid, args.boot_id, args.start_ticks, args.argv_sha256)):
            parser.error("exact PID, boot ID, start ticks, and argv hash required")
        expected = snapshot(args.pid)
        # Firmware >= interface v1.0.0.6 (2026-10-22 upgrade) renames the MC
        # binary to aima-mc-app-main; accept both vendor names.
        if expected["argv0"].rsplit("/", 1)[-1] not in ("mc_app_main", "aima-mc-app-main"):
            parser.error("execute target must have exact mc_app_main/aima-mc-app-main argv0")
        if expected["uid"] != os.getuid():
            parser.error("target must be owned by the executing user")
        if (expected["boot_id"], expected["start_ticks"], expected["argv_sha256"]) != (
                args.boot_id, args.start_ticks, args.argv_sha256):
            parser.error("target identity differs from supplied preflight")
        emit("identity", identity=expected, execute=bool(args.execute))
        if not args.execute:
            return 0
        seconds = float(args.pause_seconds)
        if not (0.05 <= seconds <= 120.0):
            parser.error("--pause-seconds must be within 0.05-120")

    cancellation = {"requested": False}

    def request_cancel(signum, frame):
        cancellation["requested"] = True

    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_cancel)
    target = None
    try:
        target = Process(expected)
        target.send(0)
        bounded_pause(target, seconds,
                      cancelled=lambda: cancellation["requested"])
        emit("end", success=True, self_test=args.self_test)
        return 0
    finally:
        if target is not None:
            target.close()
        if child is not None:
            child.terminate()
            child.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
