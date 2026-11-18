"""Join saved FGP, advisory policy and native logs by frame and state sequence."""

import argparse
import base64
from bisect import bisect_left
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def unique(records, key):
    result = {}
    for row in records:
        value = key(row)
        if value in result:
            raise ValueError(f"duplicate sequence {value}")
        result[value] = row
    return result


def array(blob):
    value = np.frombuffer(base64.b64decode(blob["data_b64"], validate=True), dtype=blob["dtype"])
    value = value.reshape(blob["shape"])
    if not np.isfinite(value).all():
        raise ValueError("non-finite trace tensor")
    return value


def delta(a, b):
    return float(np.max(np.abs(np.asarray(b) - np.asarray(a))))


def angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cosine = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return float(np.rad2deg(2 * np.arccos(np.clip(np.abs(cosine), 0, 1))).max())


def target_step(previous, current):
    names = current["reply"]["joint_names"]
    if names != previous["reply"]["joint_names"]:
        raise ValueError("joint order changed")
    changes = np.rad2deg(np.abs(np.subtract(current["reply"]["q_target_rad"],
                                            previous["reply"]["q_target_rad"])))
    index = int(np.argmax(changes))
    return names[index], float(changes[index])


def incident(path):
    samples = [row for row in rows(path) if row["kind"] == "policy_sample"]
    samples = sorted(unique(samples, lambda row: row["state"]["sequence"]).items())
    pairs = [(target_step(a, b), ia, a, ib, b)
             for (ia, a), (ib, b) in zip(samples, samples[1:])
             if ib == ia + 1 and a["accepted"] and b["accepted"]]
    ready = [(seq, row) for seq, row in samples if row["reply"].get("ready")]
    accepted = [(seq, row) for seq, row in samples if row["accepted"]]
    (joint, step), ia, a, ib, b = max(pairs, key=lambda pair: pair[0][1])
    index = b["state"]["joint_names"].index(joint)
    return dict(requests=len(samples), ready=len(ready), accepted=len(accepted),
                accepted_with_position_exceedance=sum(bool(row["pd_targets_outside_qpos_ranges"])
                                                      for _, row in accepted),
                largest_consecutive_accepted=dict(sequences=[ia, ib], joint=joint, target_step_deg=step,
                    target_rad=[row["reply"]["q_target_rad"][index] for row in (a, b)],
                    feedback_rad=[row["state"]["q_rad"][index] for row in (a, b)],
                    rpc_ms=[row["rpc_ms"] for row in (a, b)]),
                field_coverage=dict(input_window=False, onnx_inputs=False, applied_writer_targets=False))


def observation(trace, bridge, native):
    bridge_by_seq = unique(rows(bridge), lambda row: row["sequence"])
    native_by_seq = unique((r for r in rows(native) if r["kind"] == "policy_sample"),
                           lambda row: row["state"]["sequence"])
    humans, capture_times, matched = {}, [], {}
    trace_count = 0
    trailer = None
    with trace.open() as stream:
        header = json.loads(next(stream))
        if header["kind"] != "trace_start" or header["publish_allowed"]:
            raise ValueError("expected advisory trace header")
        for line in stream:
            event = json.loads(line)
            if event["kind"] == "trace_end":
                trailer = event
                if stream.readline():
                    raise ValueError("entries after trace trailer")
                break
            trace_count += 1
            if event["request_index"] != trace_count:
                raise ValueError("trace index gap")
            packet = json.loads(event["request_wire"]["json_text"])
            seq = packet["sequence"]
            if seq != event["request_sequence"] or packet["kind"] != event["request_kind"]:
                raise ValueError("wire and trace identity differ")
            if packet["kind"] == "morph_human_frame":
                if event["reply"].get("accepted_sequence") != seq:
                    continue
                capture = packet["capture_ns"]
                row = bridge_by_seq[seq]
                if (row["received_ns"] != capture or row["arrival_ns"] != packet["arrival_ns"]
                        or row["reply"] != event["reply"] or row["hardware_commands"] != 0
                        or (capture_times and capture <= capture_times[-1])):
                    raise ValueError(f"unexpected bridge join at {seq}")
                capture_times.append(capture)
                humans[capture] = packet
            elif packet["kind"] == "x2_measurement":
                row = native_by_seq[seq]
                if row["state"] != packet or row["reply"] != event["reply"]:
                    raise ValueError(f"native wire contents differ at {seq}")
                if not event["reply"].get("ready"):
                    continue
                timing = event["timing"]
                support = timing["raw_support_end_ns"]
                slots = timing["slot_capture_ns"]
                pairs = []
                for slot, expected_right in zip(slots, support, strict=True):
                    right = bisect_left(capture_times, slot)
                    if right == len(capture_times) or capture_times[right] != expected_right:
                        raise ValueError(f"missing source support at state {seq}")
                    left = right if expected_right == slot else right - 1
                    if left < 0:
                        raise ValueError(f"missing preceding source support at state {seq}")
                    pairs.append([humans[capture_times[i]]["sequence"] for i in (left, right)])
                action = array(event["onnx_outputs"]["action_human"]).ravel()
                if not np.array_equal(action, event["reply"]["canonical_action"]):
                    raise ValueError(f"ONNX action and policy reply differ at {seq}")
                matched[seq] = dict(row=row, pairs=pairs, frame_support=[support[-1], slots[-1]],
                                    onnx_action=action, ref=array(event["reference_window"]).reshape(-1, 24, 7),
                                    goal=array(event["onnx_inputs"]["goal_human"]),
                                    obs=array(event["onnx_inputs"]["obs_history"]),
                                    heading=array(event["onnx_inputs"]["human_heading"]))
    if not trailer or not trailer["complete"] or trailer["written"] != trace_count:
        raise ValueError("incomplete advisory trace")
    adjacent = []
    for seq, current in matched.items():
        previous = matched.get(seq - 1)
        if previous is None:
            continue
        a, b = previous["row"], current["row"]
        joint, step = target_step(a, b)
        prev_source = humans[previous["frame_support"][0]]
        curr_source = humans[current["frame_support"][0]]
        adjacent.append(dict(sequences=[seq - 1, seq], joint=joint, target_step_deg=step,
                             both_native_accepted=a["accepted"] and b["accepted"],
                             source_sequence_span=[prev_source["sequence"], curr_source["sequence"]],
                             source_pose_max_deg=angle(prev_source["quaternions"], curr_source["quaternions"]),
                             source_root_delta_m=float(np.linalg.norm(
                                 np.subtract(prev_source["positions"][0], curr_source["positions"][0]))),
                             last_slot_step_ms=(current["frame_support"][1]-previous["frame_support"][1])/1e6,
                             latest_slot_source_pairs=[previous["pairs"][-1], current["pairs"][-1]],
                             window_pose_max_deg=angle(previous["ref"][..., 3:], current["ref"][..., 3:]),
                             window_position_component_max_m=delta(previous["ref"][..., :3], current["ref"][..., :3]),
                             feedback_q_max_deg=float(np.rad2deg(delta(a["state"]["q_rad"], b["state"]["q_rad"]))),
                             feedback_dq_max_rad_s=delta(a["state"]["dq_rad_s"], b["state"]["dq_rad_s"]),
                             feedback_imu_deg=angle(a["state"]["imu_quat_wxyz"], b["state"]["imu_quat_wxyz"]),
                             goal_max_component=delta(previous["goal"], current["goal"]),
                             heading_max_component=delta(previous["heading"], current["heading"]),
                             history_max_component=delta(previous["obs"], current["obs"]),
                             action_max_component=delta(previous["onnx_action"], current["onnx_action"]),
                             rpc_ms=[a["rpc_ms"], b["rpc_ms"]]))
    accepted = [r for r in adjacent if r["both_native_accepted"]]
    return dict(requests_in_complete_trace=trace_count, source_frames_linked=len(humans),
                ready_states_linked=len(matched), adjacent_ready_pairs=len(adjacent),
                ready_targets_identical_to_native_reply=True,
                onnx_actions_identical_to_reply=True,
                native_rejections=dict(Counter(row["rejection_reason"] for row in native_by_seq.values()
                                               if not row["accepted"])),
                max_adjacent_accepted=max(accepted, key=lambda row: row["target_step_deg"]),
                largest_adjacent_ready=sorted(adjacent, key=lambda row: row["target_step_deg"], reverse=True)[:5])


def replay_pair(trace, policy_root, pair):
    """Swap saved input groups for one advisory transition, with no controller."""
    import onnx
    import onnxruntime as ort

    from script.x2_morph_human.artifacts import activate
    policy_dir = activate(policy_root) / "assets/morphmimic_policy"
    from deploy.sim2sim.obs import Robot
    from deploy.sim2sim.policy_node import pruned

    a, b = pair["sequences"]
    selected = {}
    for event in rows(trace):
        if (event.get("request_kind") == "x2_measurement"
                and event["request_sequence"] in (a, b)):
            selected[event["request_sequence"]] = event
            if len(selected) == 2:
                break
    old, new = (selected[index] for index in (a, b))
    model = policy_dir / "policy.onnx"
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    model_sha = digest.hexdigest()
    robot = Robot.from_dict(json.loads((policy_dir / "x2.json").read_text())["robot"])
    names = list(robot.native_joint_names)
    if any(event["reply"]["policy_sha256"] != model_sha or event["reply"]["joint_names"] != names
           for event in (old, new)):
        raise ValueError("saved model or joint mapping identity does not match")
    index = names.index(pair["joint"])
    first, second = ({key: array(value) for key, value in event["onnx_inputs"].items()}
                     for event in (old, new))
    human_keys = ("goal_human", "human_heading", "human_point_mask",
                  "human_frame_mask", "human_heading_mask")
    changing = set(human_keys) | {"obs_history"}
    if first.keys() != second.keys() or any(
            not np.array_equal(first[key], second[key]) for key in first.keys() - changing):
        raise ValueError("a supposedly fixed input changed")
    old_obs, new_obs = (feed["obs_history"].reshape(10, 93) for feed in (first, second))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(pruned(onnx.load(model), "action_human"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    cases = []
    errors = []
    for use_human, use_feedback, use_action_history in product((False, True), repeat=3):
        human = second if use_human else first
        feedback = new_obs if use_feedback else old_obs
        history = new_obs if use_action_history else old_obs
        feed = dict(first)
        feed.update({key: human[key] for key in human_keys})
        feed["obs_history"] = np.concatenate((feedback[:, :64], history[:, 64:]), axis=1).reshape(1, 930)
        action = session.run(["action_human"], feed)[0].ravel()
        target_deg = float(np.rad2deg(robot.targets(action.astype(np.float64))[index]))
        if use_human == use_feedback == use_action_history:
            event = new if use_human else old
            error = delta(action, array(event["onnx_outputs"]["action_human"]).ravel())
            if error > 2e-5 or abs(target_deg - np.rad2deg(event["reply"]["q_target_rad"][index])) > 0.002:
                raise ValueError(f"saved endpoint failed to reproduce: action error {error}")
            errors.append(error)
        cases.append(dict(new_human_features=use_human, new_sensor_history=use_feedback,
                          new_previous_action_history=use_action_history, target_deg=target_deg))
    initial = cases[0]["target_deg"]
    for case in cases:
        case["signed_change_from_first_deg"] = case["target_deg"] - initial
    return dict(state_sequences=[a, b], joint=pair["joint"], policy_sha256=model_sha,
                endpoint_action_max_abs_error=max(errors),
                source_layout="10 x (29 q, 29 dq, 3 gyro, 3 gravity, 29 previous action)",
                scope="single-pair tensor swaps with saved features; no regenerated trajectory",
                cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "bridge", "native", "incident"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, help="optionally replay the largest accepted recorded pair on CPU")
    args = parser.parse_args()
    result = dict(observation=observation(args.trace, args.bridge, args.native),
                  incident=incident(args.incident), hardware_commands=0)
    if args.policy_root is not None:
        result["pair_replay"] = replay_pair(args.trace, args.policy_root,
                                            result["observation"]["max_adjacent_accepted"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
