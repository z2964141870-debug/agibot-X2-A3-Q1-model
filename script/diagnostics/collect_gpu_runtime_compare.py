"""Mac-side read-only SSH mirror; each received JSONL line is fsynced."""
import json
import argparse
import os
from pathlib import Path
import selectors
import subprocess
import time
from collections import deque

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("case",choices=["E30","E31"])
case = parser.parse_args().case
LOGS = ROOT / ("logs/gpu_runtime_compare_20261011_"+case)
LOGS.mkdir(parents=True, exist_ok=True)
remote = "cd /media/yu/FAFF-E9771/YUANQI_A3 && /usr/bin/python3 -u -m script.diagnostics.gpu_runtime_compare "+case+" watch"
with (LOGS / "external_mirror.jsonl").open("xb", buffering=0) as stream, (LOGS / "external_mirror.stderr.log").open("wb") as err:
    child = subprocess.Popen(["ssh","-T","-o","ConnectTimeout=10","-o","ServerAliveInterval=10",
                              "-o","ServerAliveCountMax=2","hp3090",remote], stdout=subprocess.PIPE, stderr=err)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    pending, deadline = b"", time.monotonic()+270
    tail, received_bytes, received_records, prefix_full = deque(maxlen=500), 0, 0, False
    def save_tail():
        path = LOGS / "external_tail.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as target:
            target.write(b"".join(tail))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary,path)
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
                    received_bytes += len(line)+1
                    received_records += 1
                    tail.append(line+b"\n")
                    if not prefix_full and stream.tell()+len(line)+1 <= 1024**2:
                        stream.write(line+b"\n")
                        os.fsync(stream.fileno())
                    else:
                        prefix_full=True
                    record = json.loads(line)
                    item = record.get("record",{})
                    if ("event" in item and item.get("iteration",0)==0) or item.get("event") in ("ERROR","COMPLETE") or record.get("file")=="state.json":
                        print(json.dumps(record),flush=True)
                save_tail()  # Fsync each received block; bounded last 500 records.
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
            with (LOGS / "external_partial.bin").open("wb") as partial:
                partial.write(pending)
                partial.flush()
                os.fsync(partial.fileno())
        save_tail()
        print(json.dumps(dict(ssh_exit=child.returncode, bytes_received=stream.tell(),
                              total_received_bytes=received_bytes, total_received_records=received_records,
                              prefix_cap_bytes=1024**2, tail_records=len(tail),
                              meaning="TRANSPORT_END_NOT_REBOOT_PROOF")),flush=True)
