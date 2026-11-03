"""Paired-prefix tracking metrics and the user-authorized sustained early stop."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import FIXED, evaluation_gate, require


def scheduled(iteration, maximum=250000):
    return iteration in (0, 1000, maximum) or iteration > 0 and iteration % 5000 == 0


def tracking(actual, reference, body_actual, body_reference, steps):
    import numpy as np
    require(0 < steps < min(len(actual), len(reference)), "No noninitial comparable prefix")
    a, r = actual[1:steps + 1], reference[1:steps + 1]
    ap = body_actual[1:steps + 1, 2:] - body_actual[1:steps + 1, 1:2]
    rp = body_reference[1:steps + 1, 2:] - body_reference[1:steps + 1, 1:2]
    require(all(np.isfinite(x).all() for x in (a, r, ap, rp)), "Nonfinite paired evidence")
    return dict(joint_rmse_rad=float(np.sqrt(np.mean((a[:, 7:29] - r[:, 7:29]) ** 2))),
                body_relative_rmse_m=float(np.sqrt(np.mean((ap - rp) ** 2))))


def compare(initial, current):
    import numpy as np
    require(initial["clip"] == current["clip"] and initial["seed"] == current["seed"], "Case identity changed")
    require(initial["reference_sha256"] == current["reference_sha256"], "Reference changed")
    common = min(initial["valid_steps"], current["valid_steps"])
    row = dict(clip=current["clip"], seed=current["seed"], common_steps=common,
               final_complete=current["complete"], initial_complete=initial["complete"])
    if common <= 0:
        return row
    for stage, case in (("initial", initial), ("final", current)):
        with np.load(case["trace"], allow_pickle=False) as data:
            scores = tracking(data["actual"], data["reference"], data["body_actual"], data["body_reference"], common)
        row.update({stage + "_" + key: value for key, value in scores.items()})
    return row


def progress_gate(baseline, current):
    before = {(r["clip"], r["seed"]): r for r in baseline["cases"]}
    pairs = [compare(before[(r["clip"], r["seed"])], r) for r in current["cases"]]
    # This provisional gate never grants the subsequent training stage; the
    # final library, media and checkpoint audit are independently required.
    gate = evaluation_gate(pairs, True, True, True)
    return dict(**gate, pairs=pairs, provisional_only=True, authorizes_morph_training=False)


def sustained_stop(history):
    if len(history) < 3:
        return False
    last = history[-3:]
    return all(row.get("status") == "PASSED" and (row.get("gate") or {}).get("passed")
               and (row.get("gate") or {}).get("provisional_only") is True for row in last)


def complete_library(rows, selected):
    keys = [row["relative"] for row in rows]
    return len(keys) == len(selected) and len(set(keys)) == len(keys) and set(keys) == set(selected)
