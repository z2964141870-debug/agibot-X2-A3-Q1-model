"""Measure HumanFrame and state service contention through local ZMQ REP sockets."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue
import time

import numpy as np
import zmq

from .artifacts import sha
from .bench_human_ingress import CLOCK, TICKS, SharedHumanBuffer
from .core import DT_NS
from .replay_window_pacing import distribution
from .review_saved_trace import delta, rows
from .runtime import Runtime, synthetic_state


def socket(context, kind):
    connection = context.socket(kind)
    connection.setsockopt(zmq.LINGER, 0)
    connection.setsockopt(zmq.RCVTIMEO, 2000)
    connection.setsockopt(zmq.SNDTIMEO, 2000)
    connection.setsockopt(zmq.MAXMSGSIZE, 256 * 1024)
    return connection


def serve(context, endpoints, runtime, count, expected):
    with socket(context, zmq.REP) as reply:
        port = reply.bind_to_random_port("tcp://127.0.0.1")
        endpoints.put(f"tcp://127.0.0.1:{port}")
        for _ in range(count):
            data = reply.recv()
            received = time.monotonic_ns()
            request = json.loads(data)
            kind = request["kind"]
            if expected is not None and kind != expected:
                raise ValueError("request sent to wrong test endpoint")
            if kind == "morph_human_frame":
                stamp = runtime.buffer.push(request)
                result = dict(accepted_sequence=request["sequence"], accepted_ns=stamp)
            elif kind == "x2_measurement":
                cutoff = time.monotonic_ns()
                result = runtime.handle(request, cutoff, CLOCK)
                if result.get("reset") or result.get("publish_allowed"):
                    raise ValueError("state reply exceeded advisory-only contract")
                if result["ready"]:
                    result["cutoff_ns"] = cutoff
            else:
                raise ValueError("unexpected test request")
            reply.send_json(dict(received_ns=received, result=result))


def request_frames(context, endpoint, events, anchor):
    result = []
    with socket(context, zmq.REQ) as client:
        client.connect(endpoint)
        for offset, wire in events:
            deadline = anchor + offset
            while (remaining := deadline - time.monotonic_ns()) > 0:
                time.sleep(remaining / 1e9)
            sent = time.monotonic_ns()
            client.send(wire)
            response = client.recv()
            finished = time.monotonic_ns()
            reply = json.loads(response)
            result.append(dict(due_ns=deadline, sent_ns=sent, finished_ns=finished, **reply))
    return result


def exercise(root, packets, split):
    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    runtime.buffer = SharedHumanBuffer(runtime.buffer)
    for packet in packets[:12]:
        runtime.push(packet)
    now = time.monotonic_ns()
    if not runtime.handle(synthetic_state(runtime, 0, now), now, CLOCK)["ready"]:
        raise ValueError("warmup did not produce a target")
    runtime.reset()
    origin = packets[0]["arrival_ns"]
    people = [(p["arrival_ns"] - origin, json.dumps(p).encode()) for p in packets
              if p["arrival_ns"] - origin < TICKS * DT_NS]
    states = [(tick * DT_NS, json.dumps(synthetic_state(runtime, tick, 0)).encode()) for tick in range(TICKS)]
    with zmq.Context() as context, ThreadPoolExecutor(max_workers=4) as workers:
        endpoints = Queue()
        tasks = [workers.submit(serve, context, endpoints, runtime,
                                len(states) if split else len(states) + len(people),
                                "x2_measurement" if split else None)]
        state_url = endpoints.get(timeout=2)
        if split:
            tasks.append(workers.submit(serve, context, endpoints, runtime, len(people), "morph_human_frame"))
            human_url = endpoints.get(timeout=2)
        else:
            human_url = state_url
        anchor = time.monotonic_ns() + 100_000_000
        state_future = workers.submit(request_frames, context, state_url, states, anchor)
        human_future = workers.submit(request_frames, context, human_url, people, anchor)
        state_results, human_results = state_future.result(timeout=15), human_future.result(timeout=15)
        for task in tasks:
            task.result(timeout=2)

    humans = [row["result"] for row in human_results]
    accepted_ids = tuple(row["accepted_sequence"] for row in humans)
    decoded = [json.loads(wire) for _, wire in people]
    if accepted_ids != tuple(p["sequence"] for p in decoded):
        raise ValueError("human inputs were lost or reordered")
    arrivals = {p["capture_ns"]: row["accepted_ns"] for p, row in zip(decoded, humans, strict=True)}
    targets, supports = {}, {}
    for seq, row in enumerate(state_results):
        result = row["result"]
        if result["ready"]:
            timing = result["timing"]
            right = timing["raw_support_end_ns"]
            if result["state_sequence"] != seq or len(right) != 10:
                raise ValueError("unexpected state identity or reference width")
            if any(arrivals[stamp] > result["cutoff_ns"] for stamp in right):
                raise ValueError("reference used a packet after the state cutoff")
            targets[seq] = result["q_target_rad"]
            supports[seq] = (timing["slot_capture_ns"], right)
    steps = [float(np.rad2deg(delta(targets[seq-1], current))) for seq, current in targets.items()
             if seq - 1 in targets]

    def spans(replies):
        return {name: distribution([(row[end]-row[start])/1e6 for row in replies]) for name, end, start in (
            ("send_to_server_recv_ms", "received_ns", "sent_ns"),
            ("send_to_reply_ms", "finished_ns", "sent_ns"),
            ("due_to_reply_ms", "finished_ns", "due_ns"))}

    return dict(split=split, ready_states=len(targets), human=spans(human_results), state=spans(state_results),
                state_not_ready_reasons=dict(Counter(
                    row["result"]["reason"] for row in state_results if not row["result"]["ready"])),
                state_over_20_ms=sum(r["finished_ns"] - r["sent_ns"] > DT_NS for r in state_results),
                target_step_deg=distribution(steps)), accepted_ids, supports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    args = parser.parse_args()
    packets = list(rows(args.capture))
    runs = [exercise(args.policy_root, packets, separate) for separate in (False, True, True, False)]
    if any(run[1] != runs[0][1] for run in runs):
        raise ValueError("input populations differ")
    pairs = []
    for left, right in ((0, 1), (3, 2)):
        a, b = runs[left][2], runs[right][2]
        shared = a.keys() & b.keys()
        pairs.append(dict(runs=[left, right], ready_in_both=len(shared),
                          same_support=sum(a[seq] == b[seq] for seq in shared)))
    print(json.dumps(dict(capture_sha256=sha(args.capture), rounds=[item[0] for item in runs],
                          policy_state_ticks=TICKS, human_packets=len(runs[0][1]), paired_supports=pairs,
                          scope="isolated TCP loopback with fixed synthetic feedback and real recorded human packets",
                          hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
