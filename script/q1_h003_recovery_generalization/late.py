"""Fit late-only support under a nominal-performance constraint."""

import json
import shutil

import numpy as np
from experiment import ROOT, EXPERT, batch, read, digest, template, save_weights
from protocol import utility
from late_protocol import LATE_TEST, LATE_TRAIN

models = ROOT / "models"
proto = models / "late_template.onnx"
template(EXPERT, proto, late=True)
rng = np.random.default_rng(31847)
params = {"published": np.zeros(6)}
# Small corrections around the original controller, then wider gains.
for i in range(18):
    params[f"late{i}"] = rng.normal(0, (0.08, 0.35, 0.9)[i % 3], 6)
    save_weights(proto, params[f"late{i}"], models / f"late{i}.onnx")
scores = {}
batch([("train_late", s, p, "static") for p in params for s in LATE_TRAIN])
for name, p in params.items():
    groups = [read(ROOT / "runs/train_late" / name / s / "static/trials.json") for s in LATE_TRAIN]
    scores[name] = dict(
        wins=[sum(r["success"] for r in g) for g in groups], utility=utility(groups)
    )
required = scores["published"]["wins"][2]


def ranking(k):
    return (scores[k]["wins"][2] >= required, scores[k]["utility"])


winner = max((k for k in params if k != "published"), key=ranking)
weights = params[winner]
shutil.copyfile(models / (winner + ".onnx"), models / "late_selected.onnx")
ablation = weights.copy()
ablation[[1, 2, 4, 5]] = 0.0
save_weights(proto, ablation, models / "late_bias.onnx")
decision = dict(
    selected=winner,
    weights={k: v.tolist() for k, v in params.items()},
    scores=scores,
    sha256=digest(models / "late_selected.onnx"),
    bias_sha256=digest(models / "late_bias.onnx"),
    protocol_sha256=digest(ROOT / "late_protocol.py"),
)
(ROOT / "selection_late.json").write_text(json.dumps(decision, indent=2))
print(winner, scores[winner], scores["published"], flush=True)
batch(
    [
        ("test_late", s, p, k)
        for s in LATE_TEST
        for p in ("published", "late_selected", "late_bias")
        for k in ("static", "relay")
    ]
)
