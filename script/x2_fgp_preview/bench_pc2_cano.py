"""Bounded CPU CANO probe using saved tensor pairs and FP32 reference outputs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


MODEL_SHA = "adf4e93a607c45ee1a18d4e12a54648afbb709e5d5a769f9ed61743e5d581649"
WARMUP = 4
COUNT = 24


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe(model, frames, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    if [x.name for x in session.get_inputs()] != ["x_t", "x_cond"]:
        raise ValueError("CANO input contract differs")
    times = []
    error = 0.0
    for i in range(WARMUP + COUNT):
        inputs = {k: frames[k][i] for k in ("x_t", "x_cond")}
        started = time.perf_counter_ns()
        result = session.run(["residual"], inputs)[0]
        ms = (time.perf_counter_ns() - started) / 1e6
        if not np.isfinite(result).all() or result.shape != frames["cpu"][i].shape:
            raise ValueError("invalid CANO residual")
        error = max(error, float(np.max(np.abs(result - frames["cpu"][i]))))
        if i >= WARMUP:
            times.append(ms)
    if error > 2e-4:
        raise ValueError(f"CANO CPU residual diverges: {error}")
    return dict(threads=threads, calls=len(times), p50_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)), max_ms=max(times),
                calls_over_33_33_ms=sum(t > 1000 / 30 for t in times), max_abs_residual_error=error)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model", type=Path)
    cli.add_argument("fixture", type=Path)
    args = cli.parse_args()
    if digest(args.model) != MODEL_SHA:
        raise ValueError("unexpected original CANO checkpoint")
    with np.load(args.fixture, allow_pickle=False) as f:
        frames = {k: f[k] for k in ("x_t", "x_cond", "cpu")}
    if any(len(data) < WARMUP + COUNT for data in frames.values()):
        raise ValueError("missing CANO fixture frames")
    results = [probe(args.model, frames, threads) for threads in (1, 2)]
    print(json.dumps(dict(model_sha256=MODEL_SHA, fixture_sha256=digest(args.fixture),
                          ort_version=ort.__version__, runs=results, hardware_commands=0), indent=2))


if __name__ == "__main__":
    main()
