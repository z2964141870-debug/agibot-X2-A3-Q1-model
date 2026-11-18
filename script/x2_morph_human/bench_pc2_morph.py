"""Prepare saved-human MorphMimic tensors on the Mac and benchmark them on PC2."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


WARMUP = 4
COUNT = 32
CLOCK = "replay_monotonic"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(root, capture, output):
    import onnx
    from script.x2_morph_human.artifacts import activate, MODEL_SHA
    from script.x2_morph_human.core import DT_NS
    from script.x2_morph_human.review_saved_trace import rows
    from script.x2_morph_human.runtime import Runtime, synthetic_state

    root = activate(root)
    from deploy.sim2sim.policy_node import pruned

    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    sessions = []
    wrapped = runtime.runner.session

    class Recorder:
        def run(self, names, feed):
            answer = wrapped.run(names, feed)
            sessions.append(({k: v.copy() for k, v in feed.items()}, answer[0].copy()))
            return answer

    runtime.runner.session = Recorder()
    packets = list(rows(capture))
    cursor = 0
    for seq, now in enumerate(range(packets[0]["arrival_ns"], packets[-1]["arrival_ns"], DT_NS)):
        while cursor < len(packets) and packets[cursor]["arrival_ns"] <= now:
            runtime.push(packets[cursor])
            cursor += 1
        runtime.handle(synthetic_state(runtime, seq, now, CLOCK), now, CLOCK)
        if len(sessions) >= WARMUP + COUNT:
            break
    if len(sessions) != WARMUP + COUNT:
        raise ValueError("insufficient recorded policy inputs")
    model = root / "assets/morphmimic_policy/policy.onnx"
    if sha(model) != MODEL_SHA:
        raise ValueError("unexpected model")
    model_path = output / "morph_human.onnx"
    model_path.write_bytes(pruned(onnx.load(model), "action_human"))
    fixture = output / "morph_inputs.npz"
    np.savez_compressed(fixture, expected=np.stack([reply for _, reply in sessions]), **{
        key: np.stack([feed[key] for feed, _ in sessions]) for key in sessions[0][0]})
    (output / "morph_sources.json").write_text(json.dumps(dict(
        source_model_sha256=MODEL_SHA, model_sha256=sha(model_path), fixture_sha256=sha(fixture),
        capture_sha256=sha(capture), sensor_feedback="fixed synthetic", cases=len(sessions))) + "\n")


def bench(root):
    import onnxruntime as ort

    expected = json.loads((root / "morph_sources.json").read_text())
    model, fixture = root / "morph_human.onnx", root / "morph_inputs.npz"
    if (sha(model), sha(fixture)) != (expected["model_sha256"], expected["fixture_sha256"]):
        raise ValueError("model or fixture differs")
    summaries = []
    with np.load(fixture, allow_pickle=False) as data:
        feed = {key: data[key] for key in data.files if key != "expected"}
        for threads in (1, 2):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
            times, errors = [], []
            for i, reference in enumerate(data["expected"]):
                inputs = {k: v[i] for k, v in feed.items()}
                start = time.perf_counter_ns()
                action = session.run(["action_human"], inputs)[0]
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if not np.isfinite(action).all():
                    raise ValueError("non-finite policy output")
                errors.append(float(np.max(np.abs(action - reference))))
                if i >= WARMUP:
                    times.append(elapsed)
            if len(times) != COUNT or max(errors) > 2e-5:
                raise ValueError("insufficient coverage or action parity mismatch")
            summaries.append(dict(threads=threads, calls=len(times), p50_ms=float(np.median(times)),
                                  p95_ms=float(np.percentile(times, 95)), max_ms=max(times),
                                  calls_above_20_ms=sum(v > 20 for v in times),
                                  max_abs_action_error=max(errors)))
    print(json.dumps(dict(**expected, ort_version=ort.__version__, rounds=summaries, hardware_commands=0), indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--policy-root", type=Path)
    cli.add_argument("--capture", type=Path)
    args = cli.parse_args()
    if (args.policy_root is None) != (args.capture is None):
        cli.error("--policy-root and --capture are required together")
    if args.policy_root is None:
        bench(args.directory)
    else:
        prepare(args.policy_root, args.capture, args.directory)


if __name__ == "__main__":
    main()
