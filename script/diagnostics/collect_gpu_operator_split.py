"""Mac-side read-only SSH mirror; each received JSONL line is fsynced."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs/gpu_operator_split_20261011_E26"
LOGS.mkdir(parents=True, exist_ok=True)
remote = "cd /media/yu/FAFF-E9771/YUANQI_A3 && /usr/bin/python3 -u -m script.diagnostics.gpu_operator_split watch"
with (LOGS / "external_mirror.jsonl").open("xb", buffering=0) as stream, (LOGS / "external_mirror.stderr.log").open("wb") as err:
    child = subprocess.Popen(["ssh","-T","-o","ConnectTimeout=10","-o","ServerAliveInterval=10",
                              "-o","ServerAliveCountMax=2","hp3090",remote], stdout=subprocess.PIPE, stderr=err)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    pending, deadline = b"", time.monotonic()+270
    try:
        while time.monotonic()<deadline:
            events = selector.select(timeout=1)
            if not events and child.poll() is not None:
                break
            for key,_ in events:
                block = os.read(key.fileobj.fileno(),65536)
                if not block:
                    selector.unregister(child.stdout)
                    break
                pending += block
                while b"\n" in pending:
                    line,pending = pending.split(b"\n",1)
                    stream.write(line+b"\n")
                    os.fsync(stream.fileno())
                    record = json.loads(line)
                    item = record.get("record",{})
                    if "event" in item or record.get("file")=="state.json":
                        print(json.dumps(record),flush=True)
            if not selector.get_map():
                break
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        if pending:
            stream.write(pending)
            os.fsync(stream.fileno())
        print(json.dumps(dict(ssh_exit=child.returncode, bytes_received=stream.tell(),
                              meaning="TRANSPORT_END_NOT_REBOOT_PROOF")),flush=True)
