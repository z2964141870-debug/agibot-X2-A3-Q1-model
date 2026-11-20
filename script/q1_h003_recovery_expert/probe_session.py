"""Compare exact frozen RobotPose outputs and cost across CPU ORT session settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
from check_base import verify


def run(bundle, clip, dest, cpu):
    os.sched_setaffinity(0, {cpu})
    verify(bundle)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dependencies"))
    from q1_cgfa_obs_np import build_robot_pose_590_np

    data = np.load(clip)
    position, rotation, joints = (data[key] for key in ("root_p", "root_R", "q"))
    features = [build_robot_pose_590_np(position, rotation, joints, i)[None] for i in range(min(180, len(joints)))]
    rng = np.random.default_rng(7042)
    histories = rng.normal(size=(len(features), 1, 10, 80)).astype(np.float32)
    results = {}
    cases = [("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, False),
             ("unpacked", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, True),
             ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, False),
             ("basic_unpacked", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, True)]
    for name, level, unpacked in cases:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.graph_optimization_level = level
        if unpacked:
            opts.add_session_config_entry("session.disable_prepacking", "1")
        tok, head = [ort.InferenceSession(str(bundle / "onnx" / f), opts, providers=["CPUExecutionProvider"])
                     for f in ("q1_tok_robot_pose.onnx", "q1_h002_token_head.onnx")]
        token_name = tok.get_inputs()[0].name
        names = [x.name for x in head.get_inputs()]
        times = []
        digest = hashlib.sha256()
        for i in range(1150):
            j = i % len(features)
            begin = time.perf_counter_ns()
            z = tok.run(None, {token_name: features[j]})[0]
            action = head.run(None, {names[0]: histories[j], names[1]: z})[0]
            t = (time.perf_counter_ns() - begin) / 1e6
            if i >= 150:
                digest.update(z.tobytes() + action.tobytes())
                times.append(t)
        results[name] = dict(sha256=digest.hexdigest(), p50=float(np.median(times)), p99=float(np.percentile(times, 99)))
        dest.write_text(json.dumps(results, indent=2))
        print(name, results[name], flush=True)
        del tok, head


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for key in ("bundle", "clip", "dest"):p.add_argument(key,type=Path)
    p.add_argument("--cpu", type=int, default=12)
    a=p.parse_args()
    run(a.bundle, a.clip, a.dest, a.cpu)

