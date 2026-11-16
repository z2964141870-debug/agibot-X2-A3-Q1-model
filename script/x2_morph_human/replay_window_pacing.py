"""Replay reference-window pacing on saved MorphMimic advisory inputs."""

import argparse
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


def replay(root, trace, native):
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
    history = {cap: {} for cap in CAPS}
    ends = {}
    baseline = {}
    max_reproduction_error = 0.0
    max_feature_error = 0.0
    previous_seq = None
    epoch = None
    for event in rows(trace):
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
        baseline[seq] = dict(target=target, accepted=row["accepted"], end=newest, lag_ms=0.0)
        for cap in CAPS:
            end = min(newest, ends.get(cap, newest) + cap * TICK)
            ends[cap] = end
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
            history[cap][seq] = dict(target=candidate, accepted=row["accepted"], end=end,
                                     lag_ms=(newest - end) / 1e6)
        earliest_needed = min(ends.values()) - 9 * TICK
        cache = {slot: pair for slot, pair in cache.items() if slot >= earliest_needed}
    names = list(robot.native_joint_names)
    elbow = names.index("right_elbow_joint")
    spans = {}
    event_pair = [268, 269]
    for label, samples in [("baseline", baseline), *((str(c * 20), history[c]) for c in CAPS)]:
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
                scope="same recorded feedback and action history, real arrived slots only, no physical rollout",
                hardware_commands=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "trace", "native"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.policy_root, args.trace, args.native), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
