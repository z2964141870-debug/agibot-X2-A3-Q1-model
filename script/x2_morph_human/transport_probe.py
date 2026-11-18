"""Bounded local C++ -> ZeroMQ -> real human ONNX IPC check, no robot connection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

import numpy as np

from .__main__ import adapter, demo_stream, new_output, write_json
from .artifacts import home, sha
from .core import DT_NS
from .runtime import Runtime, synthetic_state


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();out=new_output(a.output)
    with socket.socket() as available:
        available.bind(("127.0.0.1",0));port=available.getsockname()[1]
    endpoint=f"tcp://127.0.0.1:{port}"
    r=Runtime();requests=[]
    for i,packet in enumerate(demo_stream(adapter(home()),40)):
        requests.append(packet)
        state=synthetic_state(r,i,i*DT_NS)
        state["expected_model_joint_names"]=list(r.robot.native_joint_names)
        requests.append(state)
    payload="".join(json.dumps(x)+"\n" for x in requests)
    (out/"requests.jsonl").write_text(payload)
    t=time.perf_counter();server=None
    with open(out/"server.stderr.log","w") as error:
        try:
            server=subprocess.Popen([sys.executable,"-m","script.x2_morph_human","serve","--endpoint",endpoint],
                                    stdout=subprocess.PIPE,stderr=error,text=True)
            readable,_,_=select.select([server.stdout],[],[],20)
            if not readable:raise RuntimeError("local service startup timeout")
            startup=json.loads(server.stdout.readline())
            if startup.get("status")!="ADVISORY_ONLY":raise RuntimeError("wrong service identity")
            result=subprocess.run([str(a.binary),endpoint],input=payload,text=True,capture_output=True,timeout=20)
            (out/"cpp.stdout.jsonl").write_text(result.stdout)
            (out/"cpp.stderr.log").write_text(result.stderr)
            if result.returncode:raise RuntimeError("C++ client failed: "+result.stderr)
            replies=[json.loads(x) for x in result.stdout.splitlines()]
            if len(replies)!=80:raise ValueError("request/reply count mismatch")
            states=replies[1::2];ready=[x for x in states if x["ready"]]
            assert len(ready)==31 and all(not x["ready"] for x in states[:9])
            assert all(not x["publish_allowed"] for x in replies)
            assert all(np.isfinite(x["q_target_rad"]).all() and len(x["q_target_rad"])==31 for x in ready)
            assert [x["state_sequence"] for x in ready]==list(range(9,40))
            summary=dict(status="CPP_LOCAL_IPC_PASSED",requests=80,real_policy_calls=31,
                         startup_wait_states=9,wall_s=time.perf_counter()-t,
                         client_binary_sha256=sha(a.binary),hal_commands=0,
                         cpp_decode_and_identity_passed=True,source="SYNTHETIC_SMPL_FIXED_ROBOT_STATE")
            write_json(out/"summary.json",summary)
            print(json.dumps(summary,indent=2))
        finally:
            if server is not None:
                server.send_signal(signal.SIGINT)
                try:server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.terminate();server.wait(timeout=5)
                server.stdout.close()


if __name__=="__main__":main()
