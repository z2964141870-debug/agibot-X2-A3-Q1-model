"""Resume only the failed startup, using the completed unchanged R2 prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_ready(line):
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Readiness must be one JSON object")
    return value


def select_worker(command, output):
    command = list(command)
    module = "script.x2_joint_alloc.worker"
    if module in command:
        index = command.index(module)
        if index == 0 or command[index-1] != "-m" or command[index+1] not in ("fgp", "policy"):
            raise ValueError("Unexpected owned worker command")
        command[index] = "script.x2_joint_startup.worker"
    elif Path(command[0]).name == "taskset":
        if (command[1:3] != ["--cpu-list", "0,1"] or len(command) < 4
                or Path(command[3]).name != "x2_morph_native"
                or "--mode" not in command or command[command.index("--mode")+1] != "observe"):
            raise ValueError("Unexpected taskset/observe command")
        command = command[:3]+[sys.executable,"-B","-m","script.x2_joint_startup.native_exec",
            "--receipt",str(output/"native_kernel_binding.json"),"--",*command[3:]]
    return command


def read_ready(child, tape, timeout=40):
    if not select.select([child.stdout], [], [], timeout)[0]:
        raise TimeoutError("Owned process initialization timeout")
    line = child.stdout.readline(16385)
    tape.write(json.dumps(dict(pid=child.pid,monotonic_ns=time.monotonic_ns(),raw=line))+"\n")
    tape.flush()
    if not line or len(line) > 16384 or not line.endswith("\n"):
        raise ValueError("Missing, overlong, or incomplete readiness record")
    return decode_ready(line)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--failed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-log", type=Path, required=True)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    if len(args.release) != 40 or any(c not in "0123456789abcdef" for c in args.release):
        parser.error("Require the published 40-character release")
    if args.output.exists() or args.protocol_log.exists():
        parser.error("Preserve previous outputs and logs")
    failure = json.loads((args.failed/"run_receipt.json").read_text())
    if (failure["status"] != "JOINT_TEST_FAILED" or "JSONDecodeError" not in failure["error"]
            or (args.failed/"fgp_outputs.npz").exists() or (args.failed/"native/status.json").exists()):
        raise ValueError("Require the preserved startup-only failure, not a replay retry")
    from script.x2_joint_alloc.allocation import validate_prefix
    validate_prefix(args.parity)
    from script.x2_joint_latency import run as coordinator
    from script.x2_joint_alloc import run as allocation
    source = {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}
    original_spawn, original_ready = subprocess.Popen, coordinator.read_ready
    def spawn(command, *a, **kw):
        return original_spawn(select_worker(command,args.output),*a,**kw)
    with args.protocol_log.open("x") as tape:
        subprocess.Popen = spawn
        coordinator.read_ready = lambda child,timeout=40: read_ready(child,tape,timeout)
        sys.argv = [sys.argv[0],"joint","--fixture",str(args.fixture),"--parity",str(args.parity),
                    "--output",str(args.output)]
        try:
            allocation.main()
        finally:
            subprocess.Popen, coordinator.read_ready = original_spawn, original_ready
            if args.output.is_dir():
                current = {p.name:digest(p) for p in Path(__file__).parent.glob("*.py")}
                if current != source:
                    raise RuntimeError("Startup wrapper source changed during execution")
                (args.output/"startup_config.json").write_text(json.dumps(dict(
                    variant="r2_native_diagnostic_channel",release=args.release,source_sha256=source,
                    reused_prefix=str(args.parity),prefix_receipt_sha256=digest(args.parity/"parity.json"),
                    preserved_failure=str(args.failed),failure_receipt_sha256=digest(args.failed/"run_receipt.json"),
                    protocol_log=str(args.protocol_log),protocol_log_sha256=digest(args.protocol_log),
                    gpu_frame_bound=900,graph_initialization_enqueues_bound=2,hardware_commands=0,
                    limitation="Transport-only change; reused R2 arithmetic prefix. No historical failure overwritten"),indent=2)+"\n")


if __name__ == "__main__":
    main()
