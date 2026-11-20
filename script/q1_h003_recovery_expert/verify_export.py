"""Check frozen ONNX against its PyTorch checkpoint along three recorded falls."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from bench import BUNDLE, HERE, Q1V2bOnnxController, scene, quat_wxyz_to_mat
from leg_height import LegHeight
from recovery import sensor_frame
from sequence_model import SequenceExpert


@torch.inference_mode()
def verify(run):
    torch.set_num_threads(1)
    policy = SequenceExpert(HERE / "candidate_seq01")
    policy.load_state_dict(torch.load(HERE / "ppo_residual_01/actor_00200.pt", map_location="cpu"))
    policy.eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(run / "q1_recovery_expert.onnx"), options, providers=["CPUExecutionProvider"])
    tracker = Q1V2bOnnxController(BUNDLE, ())
    legs = LegHeight(BUNDLE / "robot_model/supermimic.xml")
    physics = scene(run / "export_xml")
    max_target_error = max_state_error = 0.0
    modes = set()
    calls = 0
    for trial in (0, 8, 16):
        saved = np.load(run / "e1" / f"trial_{trial:02}.npz")
        last_pose = saved["initial"]
        last_action = np.zeros(22, np.float32)
        history = None
        torch_state = torch.zeros(1, 26)
        ort_state = np.zeros((1, 26), np.float32)
        for pose in saved["qpos"]:
            velocity = np.empty(physics.nv)
            mujoco.mj_differentiatePos(physics, velocity, .02, last_pose, pose)
            q = pose[7:].astype(np.float32)
            up = quat_wxyz_to_mat(pose[3:7])[2]
            height, valid = legs.measure(up, q)
            obs = np.r_[q - tracker.default_q, velocity[6:], velocity[3:6], -up, last_action]
            x = sensor_frame(obs, height, valid)
            if history is None:
                history = np.repeat(x[None, None], 10, axis=1)
            else:
                history[0, :-1] = history[0, 1:]
                history[0, -1] = x
            target, torch_state = policy(torch.from_numpy(history), torch_state)
            out, ort_state = session.run(None, {"rec_obs_hist": history, "rec_state": ort_state})
            max_target_error = max(max_target_error, float(abs(target.numpy() - out).max()))
            max_state_error = max(max_state_error, float(abs(torch_state.numpy() - ort_state).max()))
            modes.add(int(ort_state[0, 1]))
            last_pose = pose
            last_action = (out[0] - tracker.default_q) / tracker.action_scale
            calls += 1
    report = dict(calls=calls, modes=sorted(modes), max_target_error_rad=max_target_error,
                  max_recurrent_state_difference=max_state_error, bound=2e-5,
                  previous_action_source="shared ONNX raw target for numerical replay",
                  velocity_source="20 ms recorded pose differences")
    print(json.dumps(report, indent=2))
    assert modes == {0, 1, 2} and max(max_target_error, max_state_error) < 2e-5
    (run / "export_parity.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run.resolve())

