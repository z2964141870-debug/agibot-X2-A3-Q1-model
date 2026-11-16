"""Measure HumanFrame and state service contention through local ZMQ REP sockets."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Queue
import time
from unittest.mock import patch

import numpy as np
import zmq

from .artifacts import activate, sha
from .bench_human_ingress import CLOCK, TICKS, SharedHumanBuffer
from .core import DT_NS, HumanBuffer
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


class ProfiledBuffer(SharedHumanBuffer):
    def window(self, now_ns, receiver_clock):
        start = time.monotonic_ns()
        with self.lock:
            acquired = time.monotonic_ns()
            self.lock_wait_ns = acquired - start
            try:
                return HumanBuffer.window(self, now_ns, receiver_clock)
            finally:
                self.window_ns = time.monotonic_ns() - acquired


class ProfiledSession:
    def __init__(self, session):
        self.session = session
        self.run_ns = 0
        self.cpu_ns = 0

    def run(self, names, feeds):
        start, cpu = time.monotonic_ns(), time.thread_time_ns()
        try:
            return self.session.run(names, feeds)
        finally:
            self.run_ns = time.monotonic_ns() - start
            self.cpu_ns = time.thread_time_ns() - cpu


def summarize_states(client, service):
    samples = []
    for seq, row in enumerate(client):
        s = service[seq]
        if row["received_ns"] != s["received_ns"]:
            raise ValueError("state request and service timestamps differ")
        received, cutoff, returned, send_begin = (s[k] for k in ("received_ns", "cutoff_ns", "returned_ns",
                                                                "send_begin_ns"))
        stages = dict(transport_in=received - row["sent_ns"], decode=cutoff - received,
                      window_lock=s["lock_ns"], window_build=s["window_ns"], onnx=s["onnx_ns"],
                      runtime_other=returned - cutoff - s["lock_ns"] - s["window_ns"] - s["onnx_ns"],
                      reply_encode=send_begin - returned, transport_out=row["finished_ns"] - send_begin)
        if min(stages.values()) < 0 or sum(stages.values()) != row["finished_ns"] - row["sent_ns"]:
            raise ValueError("state timing decomposition does not close")
        samples.append(dict(sequence=seq, ready=row["result"]["ready"], rpc_ms=sum(stages.values()) / 1e6,
                            stages_ms={k: value / 1e6 for k, value in stages.items()},
                            onnx_calling_thread_cpu_ms=s["onnx_cpu_ns"] / 1e6,
                            state_handler_thread_cpu_ms=s["handler_cpu_ns"] / 1e6))
    ready = [row for row in samples if row["ready"]]
    baseline = {key: float(np.median([row["stages_ms"][key] for row in ready])) for key in stages}
    slow = [row for row in ready if row["rpc_ms"] > 20.0]
    for row in slow:
        row["largest_increase_from_ready_median"] = max(
            baseline, key=lambda key: row["stages_ms"][key] - baseline[key])
    return dict(ready=len(ready), ready_ms={key: distribution([r["stages_ms"][key] for r in ready])
                                           for key in stages},
                ready_over_20_ms=len(slow),
                biggest_excess_counts=dict(Counter(r["largest_increase_from_ready_median"] for r in slow)),
                worst_ready=sorted(ready, key=lambda row: row["rpc_ms"], reverse=True)[:5])



def serve(context, endpoints, runtime, count, expected, profile):
    measurements = {}
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
                if profile:
                    runtime.buffer.lock_wait_ns = runtime.buffer.window_ns = 0
                    runtime.runner.session.run_ns = runtime.runner.session.cpu_ns = 0
                    handler_cpu = time.thread_time_ns()
                result = runtime.handle(request, cutoff, CLOCK)
                if profile:
                    returned = time.monotonic_ns()
                    handler_cpu = time.thread_time_ns() - handler_cpu
                if result.get("reset") or result.get("publish_allowed"):
                    raise ValueError("state reply exceeded advisory-only contract")
                if result["ready"]:
                    result["cutoff_ns"] = cutoff
            else:
                raise ValueError("unexpected test request")
            outgoing = dict(received_ns=received, result=result)
            if profile and kind == "x2_measurement":
                wire = json.dumps(outgoing).encode("utf-8")
                send_begin = time.monotonic_ns()
                reply.send(wire)
                measurements[request["sequence"]] = dict(
                    received_ns=received, cutoff_ns=cutoff, returned_ns=returned, send_begin_ns=send_begin,
                    lock_ns=runtime.buffer.lock_wait_ns, window_ns=runtime.buffer.window_ns,
                    onnx_ns=runtime.runner.session.run_ns, onnx_cpu_ns=runtime.runner.session.cpu_ns,
                    handler_cpu_ns=handler_cpu)
            else:
                reply.send_json(outgoing)
    return measurements


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


def fixture_runtime(root, threads):
    if threads is None:
        return Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    activate(root)
    from deploy.sim2sim import policy_node

    with patch.object(policy_node, "ORT_THREADS", threads):
        runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    if runtime.runner.session.get_session_options().intra_op_num_threads != threads:
        raise ValueError("ORT thread setting not applied")
    return runtime


def fixed_clock_parity(root, packets, thread_counts):
    # Preserve identical packet visibility and feedback for all configurations.
    packets = packets[:96]
    baseline = None
    evidence = {}
    for threads in thread_counts:
        runtime = fixture_runtime(root, threads)
        position = 0
        output = {}
        for seq, now in enumerate(range(packets[0]["arrival_ns"], packets[-1]["arrival_ns"] + 1, DT_NS)):
            while position < len(packets) and packets[position]["arrival_ns"] <= now:
                runtime.push(dict(packets[position], arrival_clock=CLOCK))
                position += 1
            reply = runtime.handle(synthetic_state(runtime, seq, now, CLOCK), now, CLOCK)
            if reply.get("reset"):
                raise ValueError("parity run reset unexpectedly")
            if reply["ready"]:
                output[seq] = (reply["timing"]["raw_support_end_ns"], reply["canonical_action"], reply["q_target_rad"])
        if not output:
            raise ValueError("thread parity found no policy samples")
        if baseline is None:
            baseline = output
        if output.keys() != baseline.keys() or any(output[k][0] != baseline[k][0] for k in output):
            raise ValueError("thread parity changed visible reference frames")
        action = max(delta(output[k][1], baseline[k][1]) for k in output)
        target = max(delta(output[k][2], baseline[k][2]) for k in output)
        if action > 2e-5 or target > 2e-5:
            raise ValueError("thread parity exceeded tolerance")
        evidence[str(threads)] = dict(ready=len(output), action_max_abs=action, target_max_abs_rad=target)
    return evidence


def exercise(root, packets, split, ticks, profile, threads=None):
    runtime = fixture_runtime(root, threads)
    actual_threads = runtime.runner.session.get_session_options().intra_op_num_threads
    buffer_type = ProfiledBuffer if profile else SharedHumanBuffer
    runtime.buffer = buffer_type(runtime.buffer)
    for packet in packets[:12]:
        runtime.push(packet)
    now = time.monotonic_ns()
    if not runtime.handle(synthetic_state(runtime, 0, now), now, CLOCK)["ready"]:
        raise ValueError("warmup did not produce a target")
    runtime.reset()
    if profile:
        runtime.runner.session = ProfiledSession(runtime.runner.session)
    origin = packets[0]["arrival_ns"]
    people = [(p["arrival_ns"] - origin, json.dumps(p).encode()) for p in packets
              if p["arrival_ns"] - origin < ticks * DT_NS]
    states = [(tick * DT_NS, json.dumps(synthetic_state(runtime, tick, 0)).encode()) for tick in range(ticks)]
    with zmq.Context() as context, ThreadPoolExecutor(max_workers=4) as workers:
        endpoints = Queue()
        tasks = [workers.submit(serve, context, endpoints, runtime,
                                len(states) if split else len(states) + len(people),
                                "x2_measurement" if split else None, profile)]
        state_url = endpoints.get(timeout=2)
        if split:
            tasks.append(workers.submit(serve, context, endpoints, runtime, len(people), "morph_human_frame", profile))
            human_url = endpoints.get(timeout=2)
        else:
            human_url = state_url
        anchor = time.monotonic_ns() + 100_000_000
        state_future = workers.submit(request_frames, context, state_url, states, anchor)
        human_future = workers.submit(request_frames, context, human_url, people, anchor)
        state_results, human_results = state_future.result(timeout=ticks / 50 + 8), human_future.result(timeout=8)
        measurements = tasks[0].result(timeout=2)
        for task in tasks[1:]:
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

    profile_result = summarize_states(state_results, measurements) if profile else None
    return dict(split=split, ort_intra_op_threads=actual_threads, ready_states=len(targets), human=spans(human_results), state=spans(state_results),
                state_not_ready_reasons=dict(Counter(
                    row["result"]["reason"] for row in state_results if not row["result"]["ready"])),
                state_over_20_ms=sum(r["finished_ns"] - r["sent_ns"] > DT_NS for r in state_results),
                target_step_deg=distribution(steps), state_profile=profile_result), accepted_ids, supports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--profile-state", action="store_true")
    parser.add_argument("--thread-sweep", action="store_true", help="compare 4, 2 and 1 ORT thread on shared REP")
    args = parser.parse_args()
    if not TICKS <= args.ticks <= 900:
        parser.error("--ticks must be between 300 and 900")
    packets = list(rows(args.capture))
    if len(packets) < 96:
        parser.error("expected at least 96 recorded packets")
    if args.thread_sweep:
        counts = (4, 2, 1, 1, 2, 4)
        parity = fixed_clock_parity(args.policy_root, packets, (4, 2, 1))
        runs = [exercise(args.policy_root, packets, False, args.ticks, args.profile_state, threads)
                for threads in counts]
        comparisons = ((0, 1), (0, 2), (5, 4), (5, 3))
    else:
        parity = None
        runs = [exercise(args.policy_root, packets, split, args.ticks, args.profile_state)
                for split in (False, True, True, False)]
        comparisons = ((0, 1), (3, 2))
    if any(run[1] != runs[0][1] for run in runs):
        raise ValueError("input populations differ")
    pairs = []
    for left, right in comparisons:
        a, b = runs[left][2], runs[right][2]
        shared = a.keys() & b.keys()
        pairs.append(dict(runs=[left, right], ready_in_both=len(shared),
                          same_support=sum(a[seq] == b[seq] for seq in shared)))
    print(json.dumps(dict(capture_sha256=sha(args.capture), rounds=[item[0] for item in runs],
                          policy_state_ticks=args.ticks, human_packets=len(runs[0][1]), paired_supports=pairs,
                          fixed_clock_thread_parity=parity,
                          scope="isolated TCP loopback with fixed synthetic feedback and real recorded human packets",
                          hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
