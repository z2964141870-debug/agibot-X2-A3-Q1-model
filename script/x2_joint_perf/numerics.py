"""Bounded direct-graph thread equivalence; no replay, ROS, GPU or hardware output."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from script.x2_morph_human.artifacts import activate, sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve previous check")
    root = activate()
    import onnx
    import onnxruntime as ort
    from deploy.sim2sim.policy_node import pruned
    model_path = root/"assets/morphmimic_policy/policy.onnx"
    doc = json.loads((model_path.parent/"x2.json").read_text())
    model = pruned(onnx.load(model_path), "action_human")
    sessions = []
    for count in (4, 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = count
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        sessions.append(ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]))
    rng = np.random.default_rng(130711)
    maximum, rms, timings = 0., [], [[], []]
    for _ in range(32):
        feeds = {}
        for value in sessions[0].get_inputs():
            shape = [d if isinstance(d, int) else 1 for d in value.shape]
            if value.name in doc["inputs"]:
                feeds[value.name] = np.asarray(doc["inputs"][value.name], np.float32)
            elif value.type == "tensor(bool)":
                feeds[value.name] = np.ones(shape, dtype=bool)
            else:
                feeds[value.name] = rng.normal(0, .1, shape).astype(np.float32)
        outputs = []
        for i, session in enumerate(sessions):
            start = time.perf_counter_ns()
            out = session.run(["action_human"], feeds)[0]
            timings[i].append((time.perf_counter_ns()-start)/1e6)
            outputs.append(out)
        delta = outputs[1].astype(np.float64)-outputs[0]
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite output difference")
        maximum = max(maximum, float(np.max(np.abs(delta))))
        rms.append(float(np.sqrt(np.mean(delta**2))))
    receipt = dict(status="POLICY_THREAD_EQUIVALENCE_PASSED" if maximum <= 2e-5 else "FAILED",
        samples=32, cpu_graph_calls=64, gpu_calls=0, training_updates=0,
        model_sha256=sha(model_path), action_max_abs=maximum, rms_max=max(rms),
        tolerance=2e-5, timings_ms=timings, motor_commands=0,
        limitation="Synthetic finite feature inputs; numerical threading agreement, not human quality or timing certification")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k != "timings_ms"}))
    if receipt["status"] == "FAILED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
