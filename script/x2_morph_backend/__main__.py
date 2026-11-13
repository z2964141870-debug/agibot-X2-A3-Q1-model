"""Terminal 3: native Morph target staging; there is no motor publish mode."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

from script.x2_morph_human.artifacts import home
from script.x2_morph_human.client import Client
from script.x2_morph_human.shadow_reader import Feedback
from .contract import Backend, Contract, GROUPS
from .sensor_profile import validate as validate_sensor_profile


def make_backend(args, contract):
    if not args.endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("backend policy connection must use loopback")
    client = Client(args.endpoint, timeout_ms=args.rpc_timeout_ms)
    return Backend(contract, client, max_feedback_age_ms=args.max_feedback_age_ms,
                   max_rpc_ms=args.max_rpc_ms), client


def emit(output, value):
    line = json.dumps(value, allow_nan=False)
    output.write(line + "\n")
    output.flush()
    return line


def pipe(args, contract):
    backend, client = make_backend(args, contract)
    counts = Counter()
    try:
        with args.output.open("x") as output:
            print(json.dumps(dict(status="NO_PUBLISH_BACKEND_READY", publish_allowed=False)), flush=True)
            for line in sys.stdin:
                if not line.strip():
                    continue
                result = backend.step(json.loads(line))
                counts["ready" if result["ready"] else result["reason"]] += 1
                print(emit(output, result), flush=True)
    finally:
        client.close()
    print(json.dumps(dict(status="NO_PUBLISH_BACKEND_EXITED", counts=counts,
                          hardware_commands=0)), file=sys.stderr)


def ros(args, contract):
    cfg = json.loads(args.profile.read_text())
    sensor_basis = "INSPECTION_ONLY" if args.inspect else validate_sensor_profile(cfg)
    calibrated = sensor_basis == "MEASURED_PROFILE"
    if not 0 < args.seconds <= 600:
        raise ValueError("read-only ROS capture must be bounded to 0..600 seconds")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu

    feedback = Feedback(list(contract.names), cfg["imu_body"], cfg["body_from_sensor_wxyz"])
    counts, errors, frames = Counter(), Counter(), set()
    backend, client = (None, None) if args.inspect else make_backend(args, contract)
    rclpy.init()
    node = Node("x2_morph_backend_observer", enable_rosout=False, start_parameter_services=False)
    subscriptions = []

    def joints(msg, group):
        counts[group] += 1
        a, b = GROUPS[group]
        if sorted(j.name for j in msg.joints) != sorted(contract.names[a:b]):
            errors[group + ":joint_names"] += 1
            feedback.groups.pop(group, None)
            return
        try:
            feedback.joints(group, msg, time.monotonic_ns())
        except (ValueError, TypeError) as exc:
            feedback.groups.pop(group, None)
            errors[group + ":" + type(exc).__name__] += 1

    def imu(msg):
        counts["imu"] += 1
        frames.add(msg.header.frame_id)
        feedback.inertial(msg, time.monotonic_ns())

    try:
        for group in GROUPS:
            subscriptions.append(node.create_subscription(JointStateArray, cfg["joint_state_topics"][group],
                                 lambda msg, g=group: joints(msg, g), qos_profile_sensor_data))
        subscriptions.append(node.create_subscription(Imu, cfg["imu_topic"], imu, qos_profile_sensor_data))
        seq, started, next_tick = 0, time.monotonic(), time.monotonic() + .02
        with args.output.open("x") as output:
            print(json.dumps(dict(status="READ_ONLY_ROS_STARTED", inspect=args.inspect,
                                  measured_calibration=calibrated, sensor_basis=sensor_basis, hardware_commands=0)), flush=True)
            while rclpy.ok() and time.monotonic() - started < args.seconds:
                rclpy.spin_once(node, timeout_sec=max(0., next_tick - time.monotonic()))
                if time.monotonic() < next_tick:
                    continue
                next_tick = max(next_tick + .02, time.monotonic() + .001)
                state = feedback.snapshot(seq, time.monotonic_ns())
                if state is None:
                    continue
                if args.inspect:
                    try:
                        contract.measurement(state, args.max_feedback_age_ms)
                        valid, reason = True, None
                    except ValueError as exc:
                        valid, reason = False, str(exc)
                    emit(output, dict(state=state, feedback_valid=valid, reason=reason,
                                      policy_requests=0, hardware_commands=0))
                else:
                    emit(output, dict(state=state, candidate=backend.step(state)))
                seq += 1
    finally:
        if client is not None:
            client.close()
        node.destroy_node()
        rclpy.shutdown()
    summary = dict(status="READ_ONLY_ROS_FINISHED", received=dict(counts), ingest_errors=dict(errors),
                   complete_snapshots=seq, imu_frame_ids=sorted(frames), inspect=args.inspect,
                   measured_calibration=calibrated, sensor_basis=sensor_basis,
                   joint_schema=JointStateArray.get_fields_and_field_types(),
                   hardware_commands=0)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("pipe", "ros"))
    p.add_argument("--contract", type=Path, default=home()/"assets/morphmimic_policy/x2.json")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5765")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-feedback-age-ms", type=float, default=120.)
    p.add_argument("--max-rpc-ms", type=float, default=20.)
    p.add_argument("--rpc-timeout-ms", type=int, default=100)
    p.add_argument("--profile", type=Path)
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--inspect", action="store_true")
    args = p.parse_args()
    if args.mode == "ros" and args.profile is None:
        p.error("ROS mode requires --profile")
    if not 1 <= args.rpc_timeout_ms <= 1000:
        p.error("RPC timeout must be 1..1000 ms")
    contract = Contract.load(args.contract)
    (pipe if args.mode == "pipe" else ros)(args, contract)


if __name__ == "__main__":
    main()
