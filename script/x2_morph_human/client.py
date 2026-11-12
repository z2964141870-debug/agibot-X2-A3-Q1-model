"""Advisory service client for a garment adapter or no-publish backend callback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import zmq


class Client:
    def __init__(self, endpoint="tcp://127.0.0.1:5765", timeout_ms=1000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(endpoint)

    def request(self, message):
        self.socket.send_json(message)
        return self.socket.recv_json()

    def close(self):
        self.socket.close();self.context.term()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--endpoint",default="tcp://127.0.0.1:5765")
    p.add_argument("--input",type=Path,required=True,help="HumanFrame or X2Measurement JSONL")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--realtime",action="store_true",help="pace human frames by source capture timestamps")
    a=p.parse_args()
    client=Client(a.endpoint)
    try:
        start=time.monotonic_ns();first=None
        with open(a.input) as source,open(a.output,"x") as output:
            for line in source:
                request=json.loads(line)
                if a.realtime:
                    t=request["capture_ns"]
                    if first is None:first=t
                    wait=(start+t-first-time.monotonic_ns())/1e9
                    if wait>0:time.sleep(wait)
                t0=time.perf_counter_ns();reply=client.request(request)
                reply["local_rpc_ms"]=(time.perf_counter_ns()-t0)/1e6
                output.write(json.dumps(reply,allow_nan=False)+"\n")
    finally:client.close()


if __name__=="__main__":main()
