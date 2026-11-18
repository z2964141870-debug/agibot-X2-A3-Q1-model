"""Accept the terminal FGP datagram contract and test SONIC's native input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import signal
import socket
import sys
import time

import numpy as np

from .smpl_source import SmplFrame
from .sonic_preview import SonicIntentAdapter


def main():
    cli = argparse.ArgumentParser(__doc__)
    cli.add_argument("--forward-fd", type=int, required=True)
    cli.add_argument("--endpoint", required=True, choices=("tcp://127.0.0.1:15773",))
    cli.add_argument("--calibration", type=Path, required=True,
                     help="Parent contract; SONIC applies the old garment coordinate convention")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--trace", action="store_true",
                     help="Log at most 600 sent frames for the isolated receiver comparison")
    args = cli.parse_args()

    sys.path[:0] = ["/home/agi/projects/project_tools/x2_smpl_wholebody",
                    "/home/agi/projects/smartwear_v2"]
    import torch
    import zmq
    from general_motion_retargeting.SmpleXConverter import SMPLXConverter
    from garment_intent_adapter import joints_from_human_data, pack_pose_message, root_local_joints

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(1)
    converter = SMPLXConverter()
    adapter = SonicIntentAdapter(converter.convert_axis_angle_to_smpl_bundle_fast,
                                 joints_from_human_data, root_local_joints)
    counts = dict(sent=0, stale=0, old=0, invalid=0, superseded=0, would_block=0)
    last = -1
    with args.output.open("x") as audit, socket.socket(fileno=args.forward_fd) as source:
        context = zmq.Context()
        pub = context.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.setsockopt(zmq.SNDHWM, 2)
        try:
            pub.connect(args.endpoint)  # The native SONIC receiver binds.
            print("SONIC_INPUT_PROBE_READY engaged=0", flush=True)
            with torch.inference_mode():
                while not stopping:
                    if not select.select([source], [], [], .1)[0]:
                        continue
                    payload = source.recv(65536)
                    if not payload:
                        break
                    for _ in range(32):
                        if not select.select([source], [], [], 0)[0]:
                            break
                        payload = source.recv(65536)
                        counts["superseded"] += 1
                    if not payload:
                        break
                    try:
                        sample = json.loads(payload)
                        frame = SmplFrame.from_sample(sample)
                        if frame.sequence <= last:
                            counts["old"] += 1
                            continue
                        last = frame.sequence
                        if not 0 <= time.monotonic_ns() - frame.received_ns <= 150_000_000:
                            counts["stale"] += 1
                            continue
                        conversion_started_ns = time.monotonic_ns() if args.trace else None
                        joints, root = adapter.intent(frame)
                        conversion_ended_ns = time.monotonic_ns() if args.trace else None
                        if not 0 <= time.monotonic_ns() - frame.received_ns <= 150_000_000:
                            counts["stale"] += 1
                            continue
                        joint_payload = joints.reshape(72).astype(np.float32)
                        root_payload = root.astype(np.float32)
                        message = pack_pose_message({
                            "smpl_joints": joint_payload,
                            "human_quat": root_payload,
                            "engaged": np.zeros(1, np.float32),
                            "wrist_pr": np.zeros(4, np.float32),
                            "frame_index": np.array([frame.sequence], np.int64),
                        }, topic="pico_intent", version=4)
                    except (KeyError, TypeError, ValueError, OverflowError):
                        counts["invalid"] += 1
                        continue
                    try:
                        sent_ns = time.monotonic_ns()
                        pub.send(message, flags=zmq.DONTWAIT)
                        counts["sent"] += 1
                        if args.trace and counts["sent"] <= 600:
                            print(json.dumps({
                                "sequence": frame.sequence, "source_ns": frame.received_ns,
                                "output_ns": sample.get("output_ns"), "sent_ns": sent_ns,
                                "conversion_started_ns": conversion_started_ns,
                                "conversion_ended_ns": conversion_ended_ns,
                                "pose": frame.pose_mat, "translation": frame.root_translation_m,
                                "joints": joint_payload.tolist(), "quat": root_payload.tolist(),
                            }, separators=(",", ":"), allow_nan=False), file=audit)
                    except zmq.Again:
                        counts["would_block"] += 1
        finally:
            pub.close()
            context.term()
            print(json.dumps(dict(**counts, engaged=0, hardware_commands=0)), file=audit, flush=True)


if __name__ == "__main__":
    main()
