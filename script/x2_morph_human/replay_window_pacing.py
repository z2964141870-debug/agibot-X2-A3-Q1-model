"""Replay reference-window pacing on saved MorphMimic advisory inputs."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import json

import numpy as np

from .artifacts import activate
from .review_saved_trace import array, delta, rows, unique


TICK = 20_000_000
CAPS = (1, 2)


def distribution(values):
    return dict(count=len(values), p95=float(np.percentile(values, 95)), max=float(max(values)))


def bounded_end(latest, previous, cap, lag_limit):
    proposed = min(latest, previous + cap * TICK)
    if lag_limit is None:
        return proposed, False
    floor = latest - lag_limit * TICK
    return max(proposed, floor), proposed < floor


def grouped_baseline_steps(samples):
    by_advance, by_arrival = defaultdict(list), defaultdict(list)
    for seq, current in samples.items():
        previous = samples.get(seq - 1)
        if previous is None or not (current["accepted"] and previous["accepted"]):
            continue
        step = float(np.rad2deg(delta(previous["target"], current["target"])))
        advance = (current["end"] - previous["end"]) // 1_000_000
        by_advance[str(advance)].append(step)
        if current["support"] == previous["support"]:
            label = "same_right_support"
        else:
            gap = current["arrival_gap_ns"] / 1e6
            if gap < 0:
                raise ValueError("source arrival regressed")
            if gap <= 1000 / 30:
                label = "up_to_33.33_ms"
            elif gap <= 60:
                label = "33.33_to_60_ms"
            else:
                label = "over_60_ms"
        by_arrival[label].append(step)

    def summarize(groups):
        return {key: dict(**distribution(values), steps_above_10_deg=sum(v > 10 for v in values))
                for key, values in groups.items()}

    return dict(by_window_advance_ms=summarize(by_advance), by_right_support_arrival_gap=summarize(by_arrival))


def replay(root, trace, native, lag_limits=()):
    import onnx
    import onnxruntime as ort

    policy_dir = activate(root) / "assets/morphmimic_policy"
    from deploy.sim2sim.human import heading_features, human_features
    from deploy.sim2sim.obs import Robot
    from deploy.sim2sim.policy_node import pruned

    model = policy_dir / "policy.onnx"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(pruned(onnx.load(model), "action_human"), opts,
                                   providers=["CPUExecutionProvider"])
    robot = Robot.from_dict(json.loads((policy_dir / "x2.json").read_text())["robot"])
    accepted = unique((r for r in rows(native) if r["kind"] == "policy_sample"),
                      lambda r: r["state"]["sequence"])
    cache = {}
    modes = ({f"20_with_{limit * 20}_ms_max_lag": (1, limit) for limit in lag_limits}
             if lag_limits else {str(cap * 20): (cap, None) for cap in CAPS})
    history = {name: {} for name in modes}
    source_arrival_gap = {}
    last_source_arrival = {}
    ends = {}
    baseline = {}
    max_reproduction_error = 0.0
    max_feature_error = 0.0
    previous_seq = None
    epoch = None
    for event in rows(trace):
        if event.get("request_kind") == "morph_human_frame" and "accepted_sequence" in event["reply"]:
            packet = json.loads(event["request_wire"]["json_text"])
            stream = (packet["calibration_id"], packet["capture_clock"])
            key = (*stream, packet["capture_ns"])
            if key in source_arrival_gap:
                raise ValueError("non-unique source capture identity")
            arrived = packet["arrival_ns"]
            last = last_source_arrival.get(stream)
            source_arrival_gap[key] = None if last is None else arrived - last
            last_source_arrival[stream] = arrived
            continue
        if event.get("request_kind") != "x2_measurement" or not event["reply"].get("ready"):
            continue
        packet = json.loads(event["request_wire"]["json_text"])
        seq = packet["sequence"]
        row = accepted[seq]
        if (seq != event["request_sequence"] or row["state"] != packet or row["reply"] != event["reply"]
                or event["reply"]["policy_sha256"] != digest.hexdigest()
                or event["reply"]["joint_names"] != list(robot.native_joint_names)):
            raise ValueError("recorded reply, robot or model does not match")
        timing = event["timing"]
        current_epoch = (timing["calibration_id"], timing["arrival_clock"], event["heading_offset_rad"])
        if previous_seq is None or seq != previous_seq + 1 or current_epoch != epoch:
            cache.clear()
            ends.clear()
        previous_seq, epoch = seq, current_epoch
        slots = timing["slot_capture_ns"]
        support = timing["raw_support_end_ns"]
        newest = slots[-1]
        if len(slots) != 10 or any(b - a != TICK for a, b in zip(slots, slots[1:])):
            raise ValueError("unexpected saved reference grid")
        aligned = array(event["aligned_reference"])
        for slot, right, pose in zip(slots, support, aligned, strict=True):
            if slot in cache:
                old_right, old_pose = cache[slot]
                if right != old_right or not np.array_equal(old_pose, pose):
                    raise ValueError("saved interpolation support changed for the same slot")
            cache[slot] = right, pose
        feed = {name: array(value) for name, value in event["onnx_inputs"].items()}
        mask = feed["human_point_mask"][0]
        order = [packet["joint_names"].index(name) for name in robot.native_joint_names]
        q = np.asarray(packet["q_rad"])[order]
        torso = robot.anchor_quat(array(event["pelvis_quat_wxyz"]), q)

        def make_features(window):
            return (human_features(window, mask)[None].astype(np.float32),
                    heading_features(torso, window, mask)[None].astype(np.float32))

        goal, heading = make_features(aligned)
        feature_error = max(delta(feed["goal_human"], goal), delta(feed["human_heading"], heading))
        max_feature_error = max(max_feature_error, feature_error)
        output = session.run(["action_human"], feed)[0][0].astype(np.float64)
        target = robot.targets(output)
        error = delta(target, event["reply"]["q_target_rad"])
        max_reproduction_error = max(max_reproduction_error, error)
        if feature_error > 2e-5 or error > 2e-5:
            raise ValueError(f"failed baseline at state {seq}: {feature_error=}, {error=}")
        arrival_gap = source_arrival_gap[(timing["calibration_id"], timing["capture_clock"], support[-1])]
        baseline[seq] = dict(target=target, accepted=row["accepted"], end=newest, lag_ms=0.0,
                             support=support[-1], arrival_gap_ns=arrival_gap, forced=False)
        for name, (cap, lag_limit) in modes.items():
            end, forced = bounded_end(newest, ends.get(name, newest), cap, lag_limit)
            ends[name] = end
            paced = [cache[end - i * TICK] for i in range(9, -1, -1)]
            if any(right > support[-1] for right, _ in paced):
                raise ValueError("pacer used newer source support than baseline")
            if end == newest:
                candidate = target
            else:
                new_goal, new_heading = make_features(np.stack([pose for _, pose in paced]))
                candidate_feed = dict(feed, goal_human=new_goal, human_heading=new_heading)
                action = session.run(["action_human"], candidate_feed)[0][0].astype(np.float64)
                candidate = robot.targets(action)
            history[name][seq] = dict(target=candidate, accepted=row["accepted"], end=end,
                                      lag_ms=(newest - end) / 1e6, forced=forced)
        earliest_needed = min(ends.values()) - 9 * TICK
        cache = {slot: pair for slot, pair in cache.items() if slot >= earliest_needed}
    names = list(robot.native_joint_names)
    elbow = names.index("right_elbow_joint")
    spans = {}
    event_pair = [268, 269]
    for label, samples in [("baseline", baseline), *history.items()]:
        steps = []
        for seq, curr in samples.items():
            prev = samples.get(seq - 1)
            if prev is not None and curr["accepted"] and prev["accepted"]:
                degrees = np.rad2deg(np.abs(curr["target"] - prev["target"]))
                idx = int(degrees.argmax())
                steps.append((float(degrees[idx]), seq, names[idx]))
        peak, sequence, joint = max(steps)
        start, end = (samples[seq] for seq in event_pair)
        spans[label] = dict(accepted_adjacent_step_deg=distribution([row[0] for row in steps]),
                            accepted_adjacent_steps_above_10_deg=sum(row[0] > 10 for row in steps),
                            forced_catchups=sum(r["forced"] for r in samples.values()),
                            largest_forced_catchup_step_deg=max(
                                (step for step, seq, _ in steps if samples[seq]["forced"]), default=None),
                            peak_joint=joint, peak_sequences=[sequence-1, sequence],
                            added_reference_lag_ms=distribution([r["lag_ms"] for r in samples.values()]),
                            differing_window_cycles=sum(r["lag_ms"] > 0 for r in samples.values()),
                            deviation_from_baseline_target_deg=distribution([
                                 float(np.rad2deg(delta(r["target"], baseline[k]["target"])))
                                 for k, r in samples.items()]),
                            original_elbow_event=dict(
                                step_deg=float(np.rad2deg(abs(end["target"][elbow]-start["target"][elbow]))),
                                added_lag_ms=[start["lag_ms"], end["lag_ms"]]))
    return dict(policy_sha256=digest.hexdigest(), cycles=len(baseline), max_baseline_feature_error=max_feature_error,
                max_baseline_target_error_rad=max_reproduction_error,
                event_sequences=event_pair, cap_ms_per_ready_cycle=spans,
                baseline_groups=grouped_baseline_steps(baseline),
                arrival_scope="bridge arrival gap between the latest right bracket and its preceding input packet",
                scope="same recorded feedback and action history, real arrived slots only, no physical rollout",
                hardware_commands=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "trace", "native"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--lag-limits-ms", type=int, nargs="+", choices=(20, 40, 60),
                        help="compare a 20 ms nominal advance with each supplied backlog limit")
    args = parser.parse_args()
    limits = tuple(value // 20 for value in (args.lag_limits_ms or []))
    print(json.dumps(replay(args.policy_root, args.trace, args.native, limits), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
