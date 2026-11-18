"""Forward fresh FGP frames to the dedicated SONIC shadow-test port."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .smpl_source import read_latest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seconds", type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("shadow input duration must be 1–60 seconds")

    sys.path[:0] = ["/home/agi/projects/project_tools/x2_smpl_wholebody",
                    "/home/agi/projects/smartwear_v2"]
    import torch
    import zmq
    from general_motion_retargeting.SmpleXConverter import SMPLXConverter
    from garment_intent_adapter import GarmentIntentAdapter

    torch.set_num_threads(1)
    converter = SMPLXConverter()
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.SNDHWM, 2)
    try:
        publisher.bind("tcp://127.0.0.1:15773")
        with tempfile.TemporaryDirectory(prefix="x2-sonic-shadow-") as temp:
            # Private marker consumed only by the old static engagement gate.
            marker = Path(temp) / "engage"
            marker.touch()
            adapter = GarmentIntentAdapter(engage_file=str(marker), stationary_s=2.)
            deadline = time.monotonic() + args.seconds
            last_sequence = -1
            last_frame_time = None
            frames = packets = rejected = 0
            with torch.inference_mode():
                while time.monotonic() < deadline:
                    frame = read_latest("http://127.0.0.1:8768", 2.)
                    if frame is None or frame.sequence <= last_sequence:
                        if last_frame_time is not None and time.monotonic() - last_frame_time > .25:
                            adapter = GarmentIntentAdapter(engage_file=str(marker), stationary_s=2.)
                            last_frame_time = None
                        time.sleep(.01)
                        continue
                    last_sequence = frame.sequence
                    matrices = np.asarray(frame.pose_mat, dtype=np.float64)
                    translation = np.asarray(frame.root_translation_m, dtype=np.float32)
                    if (matrices.shape != (24, 3, 3) or translation.shape != (3,)
                            or not np.isfinite(matrices).all() or not np.isfinite(translation).all()
                            or np.max(np.abs(matrices.transpose(0, 2, 1) @ matrices - np.eye(3))) > 1e-3
                            or np.max(np.abs(np.linalg.det(matrices) - 1)) > 1e-3):
                        rejected += 1
                        adapter = GarmentIntentAdapter(engage_file=str(marker), stationary_s=2.)
                        last_frame_time = None
                        continue
                    angles = Rotation.from_matrix(matrices).as_rotvec().astype(np.float32)
                    world = converter.convert_axis_angle_to_smpl_bundle_fast(angles, translation)["human_data"]
                    last_frame_time = time.monotonic()
                    message = adapter.publish(world, last_frame_time)
                    frames += 1
                    if message is not None:
                        publisher.send(message)
                        packets += 1
                        if packets == 1:
                            print("SHADOW_INTENT_ACTIVE", flush=True)
            print(json.dumps(dict(frames=frames, packets=packets, rejected=rejected,
                                  last_gate=adapter.last_reason, hardware_commands=0)), flush=True)
    finally:
        publisher.close()
        context.term()


if __name__ == "__main__":
    main()
