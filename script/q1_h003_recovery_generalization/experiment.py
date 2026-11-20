"""Fit a bounded feedback policy on three terrains, freeze it, then run held-out tests."""

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from protocol import TRAIN, TEST, utility
from model import template, save_weights
from run_suite import EXPERT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_job(job):
    split, condition, arm, kind = job
    dest = ROOT / "runs" / split / arm / condition / kind
    if (dest / "condition.json").is_file():
        return
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / ("_".join(job) + ".txt")).open("w") as f:
        subprocess.run(
            [sys.executable, str(ROOT / "trial.py"), *job],
            check=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def batch(jobs):
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(run_job, jobs))


def read(path):
    return json.loads(path.read_text())


def fit():
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    proto = models / "template.onnx"
    template(EXPERT, proto)
    params = {"published": np.zeros(6)}
    save_weights(proto, params["published"], models / "zero.onnx")
    batch([("parity", "nominal", "zero", "static")])
    archive = ROOT.parent / "static"
    for i in range(24):
        name = f"trial_{i:02}.npz"
        with (
            np.load(ROOT / "runs/parity/zero/nominal/static" / name) as a,
            np.load(archive / name) as b,
        ):
            assert all(np.array_equal(a[k], b[k]) for k in ("initial", "trace", "qpos"))
    rng = np.random.default_rng(1847)
    scores = {}

    def evaluate(names):
        for name in names:
            if name != "published":
                save_weights(proto, params[name], models / (name + ".onnx"))
        batch([("train", surface, name, "static") for name in names for surface in TRAIN])
        for name in names:
            rows = [read(ROOT / "runs/train" / name / s / "static/trials.json") for s in TRAIN]
            assert all(len(r) == 6 for r in rows)
            scores[name] = {
                "utility": utility(rows),
                "wins": [sum(x["success"] for x in r) for r in rows],
            }
        print({k: scores[k] for k in names}, flush=True)

    # Include constant pace-only anchors, then fit six feedback parameters by
    # seeded population search with a second generation around the best fits.
    for i in range(10):
        params[f"a{i}"] = rng.normal(0, 0.8, size=6)
    params["slow"] = np.array([-1.0, 0, 0, 0, 0, 0])
    params["fast"] = np.array([1.0, 0, 0, 0, 0, 0])
    evaluate(list(params))
    elite = sorted(params, key=lambda k: scores[k]["utility"], reverse=True)[:4]
    for i in range(8):
        params[f"b{i}"] = np.clip(
            params[elite[i % len(elite)]] + rng.normal(0, 0.4, size=6), -2.0, 2.0
        )
    evaluate([f"b{i}" for i in range(8)])
    # Require a state-dependent candidate for the comparison even when the
    # published reference beats all such candidates on the training subset.
    chosen = max(
        (k for k in params if k.startswith(("a", "b"))), key=lambda k: scores[k]["utility"]
    )
    candidate = models / "selected.onnx"
    shutil.copyfile(models / (chosen + ".onnx"), candidate)
    fixed = params[chosen].copy()
    fixed[[1, 2, 4, 5]] = 0.0
    save_weights(proto, fixed, models / "bias_only.onnx")
    decision = dict(
        selected=chosen,
        weights={k: v.tolist() for k, v in params.items()},
        scores=scores,
        sha256=digest(candidate),
        bias_only_sha256=digest(models / "bias_only.onnx"),
        protocol_sha256=digest(ROOT / "protocol.py"),
        parity_static=24,
    )
    (ROOT / "selection.json").write_text(json.dumps(decision, indent=2))
    print("Frozen", chosen, decision["sha256"], flush=True)


def confirm():
    frozen = read(ROOT / "selection.json")
    assert digest(ROOT / "models/selected.onnx") == frozen["sha256"]
    assert digest(ROOT / "models/bias_only.onnx") == frozen["bias_only_sha256"]
    assert digest(ROOT / "protocol.py") == frozen["protocol_sha256"]
    batch(
        [
            ("test", s, a, k)
            for s in TEST
            for a in ("published", "selected", "bias_only")
            for k in ("static", "relay")
        ]
    )


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit()
    elif sys.argv[1] == "test":
        confirm()
    else:
        raise ValueError("expected fit or test")
