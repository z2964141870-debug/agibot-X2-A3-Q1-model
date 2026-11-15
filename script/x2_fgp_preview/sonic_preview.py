"""Run the installed X2 SONIC tokenizer on FGP frames in memory."""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .smpl_source import SmplFrame, read_latest


class SonicIntentAdapter:
    """Adapt FGP's matrices through the existing SONIC garment skeleton."""

    def __init__(self, convert, joints_from_human_data, root_local_joints):
        self.convert = convert
        self.joints_from_human_data = joints_from_human_data
        self.root_local_joints = root_local_joints

    def intent(self, frame):
        matrices = np.asarray(frame.pose_mat, dtype=np.float64)
        translation = np.asarray(frame.root_translation_m, dtype=np.float32)
        if matrices.shape != (24, 3, 3) or translation.shape != (3,):
            raise ValueError("FGP SMPL24 pose or root has the wrong shape")
        if not np.isfinite(matrices).all() or not np.isfinite(translation).all():
            raise ValueError("FGP SMPL24 pose and root must be finite")
        if (np.max(np.abs(matrices.transpose(0, 2, 1) @ matrices - np.eye(3))) > 1e-3
                or np.max(np.abs(np.linalg.det(matrices) - 1)) > 1e-3):
            raise ValueError("FGP pose contains invalid rotation matrices")
        angles = Rotation.from_matrix(matrices).as_rotvec().astype(np.float32)
        world = self.convert(angles, translation)["human_data"]
        positions, quaternions = self.joints_from_human_data(world)
        return self.root_local_joints(positions, quaternions[0])


def grab(preview_url, seconds, limit):
    last = -1
    deadline = time.monotonic() + seconds
    count = 0
    while time.monotonic() < deadline and count < limit:
        frame = read_latest(preview_url, timeout=2.)
        if frame is not None and frame.sequence > last:
            last, count = frame.sequence, count + 1
            yield frame
        time.sleep(.01)


def preview(frames, adapter, core, control_dt, unpack, decode):
    count = tokens = 0
    first_ns = last_ns = None
    next_tick = 1.
    longest_gap = max_root_error = max_token_norm = max_action = 0.
    for frame in frames:
        if last_ns is not None:
            if frame.received_ns <= last_ns:
                raise ValueError("source timestamps must increase")
            gap_ms = (frame.received_ns - last_ns) / 1e6
            if gap_ms > 500:
                raise ValueError("FGP frames crossed a source gap longer than 500 ms")
            longest_gap = max(longest_gap, gap_ms)
        if first_ns is None:
            first_ns = frame.received_ns
        capture = 1. + (frame.received_ns - first_ns) / 1e9
        last_ns = frame.received_ns
        joints, root = adapter.intent(frame)
        if joints.shape != (24, 3) or root.shape != (4,) or not np.isfinite(joints).all():
            raise ValueError("invalid SONIC joint or root output")
        if not np.isfinite(root).all() or abs(np.linalg.norm(root) - 1.) > 1e-4:
            raise ValueError("invalid SONIC root quaternion")
        max_root_error = max(max_root_error, float(np.linalg.norm(joints[0])))
        # Drive the official core with recorded receive spacing and mock robot
        # feedback. No sockets are constructed, and packed messages stay local.
        while next_tick < capture - 1e-9:
            core.feed_debug(next_tick, [1., 0., 0., 0.])
            result = core.tick(control_dt)
            if result.published:
                fields = unpack(result.msg, expected_topic="pose").fields
                token = np.asarray(fields["motion_token"]).ravel()
                if token.shape != (64,) or not np.isfinite(token).all():
                    raise ValueError("invalid encoded SONIC token")
                action = decode(token)
                if action.shape != (1, 31) or not np.isfinite(action).all():
                    raise ValueError("invalid SONIC decoder action")
                max_action = max(max_action, float(np.max(np.abs(action))))
                tokens += 1
                max_token_norm = max(max_token_norm, float(np.linalg.norm(token)))
            next_tick += control_dt
        core.feed_intent(capture, joints, root, engaged=1., wrist_pr=np.zeros(4))
        count += 1
    if tokens == 0:
        raise ValueError(f"no SONIC tokens after {count} usable SMPL frames")
    return dict(frames=count, tokens=tokens, max_source_gap_ms=longest_gap,
                max_pelvis_residual_m=max_root_error, max_token_norm=max_token_norm,
                decoder_actions=tokens, max_abs_decoder_action=max_action,
                token_feedback="synthetic-standing", decoder_proprioception="all-zero-fixture",
                hardware_commands=0, planner_messages=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recording", type=Path)
    source.add_argument("--preview-url")
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--garment-root", type=Path, default=Path("/home/agi/projects/smartwear_v2"))
    parser.add_argument("--sonic-tools", type=Path, default=Path("/home/agi/projects/project_tools/x2_smpl_wholebody"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/home/agi/gear-sonic/policies/x2_smpl_tokenizer_v11release.onnx"))
    parser.add_argument("--decoder", type=Path, default=Path("/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1_token.onnx"))
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 900 or not 1 <= args.seconds <= 30:
        parser.error("preview requires 1–900 frames and a 1–30 second read window")

    sys.path[:0] = [str(args.sonic_tools.resolve()), str(args.garment_root.resolve())]
    from general_motion_retargeting.SmpleXConverter import SMPLXConverter
    from garment_intent_adapter import joints_from_human_data, root_local_joints
    from token_service_core import CONTROL_DT, TokenServiceCore
    from wire_format import unpack_message
    import onnxruntime as ort
    import torch

    torch.set_num_threads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(args.tokenizer), sess_options=options, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    decoder = ort.InferenceSession(str(args.decoder), sess_options=options, providers=["CPUExecutionProvider"])
    if (decoder.get_inputs()[0].name != "obs" or decoder.get_inputs()[0].shape[-1] != 1670
            or decoder.get_modelmeta().custom_metadata_map.get("ori_mode") != meta["ori_mode"]):
        raise ValueError("SONIC decoder does not match the token observation contract")

    def decode(token):
        # OnnxActor concatenates the 680-D token slot (64 token + 616 zeros)
        # before 990-D proprioception. This preview uses a zero fixture.
        obs = np.zeros((1, 1670), np.float32)
        obs[0, :64] = token
        return decoder.run(["action"], {"obs": obs})[0]

    converter = SMPLXConverter()
    adapter = SonicIntentAdapter(converter.convert_axis_angle_to_smpl_bundle_fast,
                                 joints_from_human_data, root_local_joints)
    core = TokenServiceCore(lambda obs: sess.run(["motion_token"], {"smpl_obs": obs})[0],
                            ori_mode=meta["ori_mode"], now=1. - CONTROL_DT)
    with torch.inference_mode():
        if args.recording:
            with args.recording.open() as log:
                frames = (SmplFrame.from_sample(json.loads(line)) for line in islice(log, args.max_frames))
                result = preview(frames, adapter, core, CONTROL_DT, unpack_message, decode)
        else:
            result = preview(grab(args.preview_url, args.seconds, args.max_frames),
                             adapter, core, CONTROL_DT, unpack_message, decode)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
