"""Compare bounded reference pacing on independent FGP captures, offline."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .artifacts import sha
from .core import DT_NS, HumanBuffer, NotReady
from .replay_window_pacing import bounded_end, distribution
from .review_saved_trace import delta, rows
from .runtime import Runtime, synthetic_state


CLOCK = "replay_monotonic"
LIMITS_MS = (0, 20, 40)
DROPOUT_FRAMES = 8


class ReplayBuffer(HumanBuffer):
    """Evaluate pacing after the normal ready/freshness/interpolation checks."""

    def __init__(self, skeleton_id, *, lag_limit, **options):
        super().__init__(skeleton_id, **options)
        self.lag_limit = lag_limit
        self.previous = None
        self.cache = {}
        self.lag_ms = 0
        self.forced = False

    def window(self, now_ns, receiver_clock):
        try:
            window, timing = super().window(now_ns, receiver_clock)
        except NotReady:
            self.previous = None
            self.cache.clear()
            raise
        slots = timing["slot_capture_ns"]
        support = timing["raw_support_end_ns"]
        latest = slots[-1]
        for stamp, source, pose in zip(slots, support, window, strict=True):
            self.cache[stamp] = source, pose
        previous = latest if self.previous is None else self.previous
        end, self.forced = bounded_end(latest, previous, 1, self.lag_limit)
        self.previous = end
        self.lag_ms = (latest - end) / 1e6
        wanted = [end - i * DT_NS for i in range(len(slots) - 1, -1, -1)]
        candidates = [self.cache[stamp] for stamp in wanted]
        if any(source > support[-1] for source, _ in candidates):
            raise ValueError("unexpected future support in paced window")
        self.cache = {stamp: sample for stamp, sample in self.cache.items() if stamp >= wanted[0]}
        return np.stack([pose for _, pose in candidates]), dict(
            timing, slot_capture_ns=wanted, raw_support_end_ns=[source for source, _ in candidates])


def play(root, packets, lag_limit):
    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    buffer = runtime.buffer
    if lag_limit:
        buffer = ReplayBuffer(buffer.skeleton_id, lag_limit=lag_limit // 20,
                              max_age_ns=buffer.max_age_ns, max_interp_gap_ns=buffer.max_interp_gap_ns)
        runtime.buffer = buffer
    outputs, rejections = {}, {}
    position = 0
    start, end = packets[0]["arrival_ns"], packets[-1]["arrival_ns"]
    for seq, now in enumerate(range(start, end + 1, DT_NS)):
        while position < len(packets) and packets[position]["arrival_ns"] <= now:
            runtime.push(packets[position])
            position += 1
        reply = runtime.handle(synthetic_state(runtime, seq, now), now, CLOCK)
        if reply["ready"]:
            outputs[seq] = (np.asarray(reply["q_target_rad"]), buffer.lag_ms if lag_limit else 0,
                            buffer.forced if lag_limit else False, reply["timing"]["slot_capture_ns"][-1])
        else:
            rejections[seq] = reply.get("reason", reply.get("error"))
            if reply.get("reset") or rejections[seq] is None:
                raise ValueError(f"unexpected replay failure at {seq}: {reply}")
    return outputs, rejections, list(runtime.robot.native_joint_names)


def replay(root, path, dropout):
    full = list(rows(path))
    if len(full) < 100 or any(p["arrival_clock"] != CLOCK for p in full):
        raise ValueError("expected at least 100 recorded FGP packets on replay clock")
    middle = len(full) // 2
    dropped = full[middle:middle + DROPOUT_FRAMES] if dropout else []
    packets = full[:middle] + full[middle + len(dropped):]
    modes = {}
    for limit in LIMITS_MS:
        output, rejections, names = play(root, packets, limit)
        if limit == 0:
            baseline, baseline_rejects = output, rejections
        if output.keys() != baseline.keys() or rejections != baseline_rejects:
            raise ValueError("pacing changed the original readiness gates")
        def transition(previous, tick):
            a, _, _, start_slot = output[previous]
            b, _, _, end_slot = output[tick]
            changes = np.rad2deg(np.abs(b - a))
            joint = int(np.argmax(changes))
            return dict(state_ticks=[previous, tick], joint=names[joint], target_step_deg=float(changes[joint]),
                        window_advance_ms=(end_slot - start_slot) / 1e6)

        events = [transition(tick - 1, tick) for tick in output if tick - 1 in output]
        steps = [event["target_step_deg"] for event in events]
        if not steps:
            raise ValueError("no adjacent ready pairs")
        recovery = {}
        for previous, tick in zip(output, list(output)[1:]):
            if tick > previous + 1:
                recovery[tick] = dict(no_target_ticks=tick - previous - 1,
                                      first_window_lag_ms=output[tick][1], **transition(previous, tick))
        modes[str(limit)] = dict(step_deg=distribution(steps), steps_above_10_deg=sum(v > 10 for v in steps),
                                 peak_event=max(events, key=lambda event: event["target_step_deg"]),
                                 lag_ms=distribution([row[1] for row in output.values()]),
                                 forced_catchups=sum(row[2] for row in output.values()),
                                 deviation_from_baseline_deg=distribution([
                                     float(np.rad2deg(delta(row[0], baseline[tick][0])))
                                     for tick, row in output.items()]), recoveries=recovery)
    return dict(source=str(path), source_sha256=sha(path), packets=len(packets),
                dropped_sequences=[p["sequence"] for p in dropped], ready_cycles=len(baseline),
                rejections=dict(Counter(baseline_rejects.values())), results=modes)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--policy-root", type=Path, required=True)
    cli.add_argument("captures", type=Path, nargs="+")
    args = cli.parse_args()
    print(json.dumps(dict(scope="recorded human packets, fixed synthetic robot state, independent action history",
                          cases=[replay(args.policy_root, path, dropout) for path in args.captures
                                 for dropout in (False, True)], hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
