"""Bounded tensor-level PC2 policy round trip using saved ONNX inputs."""

import argparse
import json
from pathlib import Path
import socket
import struct
import time

import numpy as np


DYNAMIC = ("obs_history", "goal_human", "human_heading",
           "human_point_mask", "human_frame_mask", "human_heading_mask")
COUNT = 164
WARMUP = 4
HEADER = struct.Struct("!Id")


def exact(sock, size):
    payload = bytearray()
    while len(payload) < size:
        chunk = sock.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("connection closed during benchmark")
        payload.extend(chunk)
    return bytes(payload)


class Inputs:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as source:
            self.dynamic = {k: source[k] for k in DYNAMIC}
            self.fixed = {}
            for key in set(source.files) - set(DYNAMIC) - {"expected"}:
                values = source[key]
                if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                    raise ValueError(f"{key} was expected to be fixed")
                self.fixed[key] = values[0]
            self.expected = source["expected"]
        self.size = sum(v[0].nbytes for v in self.dynamic.values())

    def pack(self, row):
        return b"".join(v[row].tobytes() for v in self.dynamic.values())

    def unpack(self, raw):
        feed = dict(self.fixed)
        offset = 0
        for key, values in self.dynamic.items():
            template = values[0]
            feed[key] = np.frombuffer(raw, template.dtype, count=template.size, offset=offset).reshape(template.shape)
            offset += template.nbytes
        return feed


def serve(args, inputs):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(args.directory / "morph_human.onnx"), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    with socket.socket() as listener:
        listener.bind((args.host, 0))
        listener.listen(1)
        listener.settimeout(30)
        print(f"PC2_TEST_PORT {listener.getsockname()[1]}", flush=True)
        peer, address = listener.accept()
        with peer:
            if address[0] != args.allow:
                raise ValueError("unexpected test peer")
            peer.settimeout(3)
            peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for i in range(COUNT):
                raw = exact(peer, 4 + inputs.size)
                sequence, = struct.unpack_from("!I", raw)
                if sequence != i:
                    raise ValueError("out-of-order policy request")
                feed = inputs.unpack(raw[4:])
                start = time.perf_counter_ns()
                action = session.run(["action_human"], feed)[0]
                elapsed = (time.perf_counter_ns() - start) / 1e6
                peer.sendall(HEADER.pack(i, elapsed) + action.tobytes())


def client(args, inputs):
    scores, inference, errors = [], [], []
    with socket.create_connection((args.host, args.port), timeout=3) as transport:
        transport.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        epoch = time.monotonic_ns() + 100_000_000
        for i in range(COUNT):
            sample = i % len(inputs.expected)
            expected = inputs.expected[sample]
            wire = struct.pack("!I", i) + inputs.pack(sample)
            due = epoch + i * 20_000_000
            while (remaining := due - time.monotonic_ns()) > 0:
                time.sleep(remaining / 1e9)
            start = time.monotonic_ns()
            transport.sendall(wire)
            response = exact(transport, HEADER.size + expected.nbytes)
            stop = time.monotonic_ns()
            received, forward_ms = HEADER.unpack_from(response)
            if received != i:
                raise ValueError("policy reply mismatch")
            answer = np.frombuffer(response, expected.dtype, offset=HEADER.size).reshape(expected.shape)
            errors.append(float(np.max(np.abs(expected - answer))))
            if i >= WARMUP:
                scores.append(((stop-start)/1e6, (stop-due)/1e6))
                inference.append(forward_ms)
    if max(errors) > 2e-5:
        raise ValueError("cross-board policy output differs")
    summaries = {}
    for name, data in zip(("rpc", "scheduled_to_reply", "pc2_forward"), (*np.array(scores).T, inference)):
        summaries[name] = dict(p50_ms=float(np.median(data)), p95_ms=float(np.percentile(data, 95)),
                               max_ms=float(max(data)), above_20_ms=int(np.count_nonzero(np.asarray(data) > 20)))
    print(json.dumps(dict(measured_requests=len(scores), input_bytes=inputs.size + 4,
                          timings=summaries, max_abs_action_error=max(errors),
                          scope="saved precomputed tensors only; no control or live feedback", hardware_commands=0),
                     indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("role", choices=("server", "client"))
    cli.add_argument("directory", type=Path)
    cli.add_argument("host")
    cli.add_argument("--allow")
    cli.add_argument("--port", type=int)
    args = cli.parse_args()
    if args.role == "server" and not args.allow:
        cli.error("server requires the allowed peer IP")
    if args.role == "client" and not args.port:
        cli.error("client requires the ephemeral server port")
    inputs = Inputs(args.directory / "morph_inputs.npz")
    (serve if args.role == "server" else client)(args, inputs)


if __name__ == "__main__":
    main()
