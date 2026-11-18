"""Compare serial and independent HumanFrame dispatch with the advisory runtime."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
import time

import numpy as np

from .artifacts import sha
from .core import DT_NS, HumanBuffer
from .replay_window_pacing import distribution
from .review_saved_trace import delta, rows
from .runtime import Runtime, synthetic_state


CLOCK = "offline_ingress_benchmark"
TICKS = 300


class SharedHumanBuffer(HumanBuffer):
    """Serialize only buffer mutation and window sampling across the two workers."""

    def __init__(self, original):
        super().__init__(original.skeleton_id, max_age_ns=original.max_age_ns,
                         max_interp_gap_ns=original.max_interp_gap_ns)
        self.lock = Lock()

    def push(self, packet):
        with self.lock:
            stamp = time.monotonic_ns()
            super().push(dict(packet, arrival_ns=stamp, arrival_clock=CLOCK))
        return stamp

    def window(self, now_ns, receiver_clock):
        with self.lock:
            return super().window(now_ns, receiver_clock)


def run(root, packets, split):
    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    runtime.buffer = SharedHumanBuffer(runtime.buffer)
    # Warm the model through the normal advisory interface, then clear its history.
    warmup = time.monotonic_ns()
    for packet in packets[:12]:
        runtime.push(packet)
    ready = runtime.handle(synthetic_state(runtime, 0, warmup), time.monotonic_ns(), CLOCK)
    if not ready.get("ready"):
        raise ValueError(f"warmup did not produce a policy output: {ready}")
    runtime.reset()

    first = packets[0]["arrival_ns"]
    events = [(packet["arrival_ns"] - first, "human", packet)
              for packet in packets if packet["arrival_ns"] - first < TICKS * DT_NS]
    events.extend((tick * DT_NS, "state", tick) for tick in range(TICKS))
    events.sort(key=lambda event: event[:2])

    def handle(kind, value, due, enqueued_ns):
        started = time.monotonic_ns()
        if kind == "human":
            arrival = runtime.buffer.push(value)
            result = dict(sequence=value["sequence"], capture_ns=value["capture_ns"], accepted_ns=arrival)
        else:
            packet = synthetic_state(runtime, value, started, CLOCK)
            result = runtime.handle(packet, started, CLOCK)
            if result.get("reset") or result.get("publish_allowed"):
                raise ValueError(f"state reply outside advisory benchmark contract: {result}")
        finished = time.monotonic_ns()
        return dict(kind=kind, due=due, enqueued=enqueued_ns, started=started, finished=finished, result=result)

    with ThreadPoolExecutor(max_workers=1) as state_pool, ThreadPoolExecutor(max_workers=1) as human_pool:
        anchor = time.monotonic_ns() + 100_000_000
        futures = []
        pending = set()
        for offset, kind, value in events:
            due = anchor + offset
            while (remaining := due - time.monotonic_ns()) > 0:
                time.sleep(remaining / 1e9)
            pool = human_pool if split and kind == "human" else state_pool
            # Limit outstanding requests to keep a saturated run finite and observable.
            pending = {future for future in pending if not future.done()}
            if len(pending) >= 8:
                raise RuntimeError("more than the bounded pending request budget")
            future = pool.submit(handle, kind, value, due, time.monotonic_ns())
            pending.add(future)
            futures.append(future)
        outcomes = [future.result(timeout=2) for future in futures]

    humans = [row for row in outcomes if row["kind"] == "human"]
    states = [row for row in outcomes if row["kind"] == "state"]
    arrivals = {row["result"]["capture_ns"]: row["result"]["accepted_ns"] for row in humans}
    if len(arrivals) != len(humans):
        raise ValueError("duplicate source timestamp")
    ids = [row["result"]["sequence"] for row in humans]
    if ids != sorted(set(ids)):
        raise ValueError("human dispatch lost ordering")
    outputs = {}
    support_records = {}
    for seq, row in enumerate(states):
        result = row["result"]
        if result["ready"]:
            support = result["timing"]["raw_support_end_ns"]
            if len(support) != 10 or any(arrivals[stamp] > row["started"] for stamp in support):
                raise ValueError("reference used a packet accepted after its state cutoff")
            outputs[seq] = result["q_target_rad"]
            support_records[seq] = (tuple(result["timing"]["slot_capture_ns"]), tuple(support))
    if not outputs:
        raise ValueError("no ready states")
    steps = [float(np.rad2deg(delta(outputs[seq - 1], value))) for seq, value in outputs.items()
             if seq - 1 in outputs]
    if not steps:
        raise ValueError("no consecutive output")

    def spans(items):
        report = {}
        for name, end, start in (("source_due_to_reply_ms", "finished", "due"),
                                 ("executor_wait_ms", "started", "enqueued"),
                                 ("handler_ms", "finished", "started")):
            report[name] = distribution([(row[end] - row[start]) / 1e6 for row in items])
        return report

    return dict(split=split, human=spans(humans), state=spans(states), human_packets=len(humans),
                state_requests=len(states), ready_states=len(outputs), rejections=dict(Counter(
                    row["result"].get("reason", "unknown") for row in states if not row["result"]["ready"])),
                consecutive_target_step_deg=distribution(steps),
                human_input_sequences=ids, window_supports=support_records)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--policy-root", type=Path, required=True)
    cli.add_argument("--capture", type=Path, required=True)
    args = cli.parse_args()
    packets = list(rows(args.capture))
    if len(packets) < 12:
        raise ValueError("not enough recorded input")
    runs = [run(args.policy_root, packets, split) for split in (False, True, True, False)]
    ids = runs[0]["human_input_sequences"]
    if any(run["human_input_sequences"] != ids for run in runs):
        raise ValueError("release populations differ")
    common = []
    for left, right in ((0, 1), (3, 2)):
        before, after = [runs[index]["window_supports"] for index in (left, right)]
        shared = before.keys() & after.keys()
        common.append(dict(runs=[left, right], ready_in_both=len(shared), same_support=sum(
            before[seq] == after[seq] for seq in shared)))
    for result in runs:
        del result["human_input_sequences"], result["window_supports"]
    print(json.dumps(dict(source_sha256=sha(args.capture), state_ticks=TICKS, runs=runs, paired_supports=common,
                          scope="local dispatch without ZMQ or BLE; fixed synthetic feedback; original policy",
                          hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
