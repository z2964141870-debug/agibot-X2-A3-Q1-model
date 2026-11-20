"""Fit the rise motion and balance gains in delayed MuJoCo and rank by full relay performance."""

import ctypes
import json
import os
import sys
import subprocess
from pathlib import Path
import mujoco
import numpy as np
import onnx
from onnx import numpy_helper
from protocol import ROOT, EVALUATION
from run import run, read, sha

sys.path.insert(0, str(ROOT.parent))
from run_suite import EXPERT, BUNDLE


def fit():
    dest = ROOT / "motion"
    dest.mkdir(exist_ok=False)
    mj = Path(mujoco.__file__).resolve().parent
    libmj = next(mj.glob("libmujoco.so.*"))
    so = dest / "delayed_rise.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fopenmp",
            str(ROOT / "delayed_rise.cc"),
            "-I" + str(mj / "include"),
            str(libmj),
            "-Wl,-rpath," + str(mj),
            "-o",
            str(so),
        ],
        check=True,
    )
    native = ctypes.CDLL(str(so)).evaluate_delayed
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    native.argtypes = [ctypes.c_char_p, ctypes.c_int, ptr, ptr, ptr, ptr, ptr]
    cfg = read(BUNDLE / "config/q1_deploy_config.json")
    stand, lo, hi = (np.array(cfg[k], dtype=np.float64) for k in ("default_q", "q_lo", "q_hi"))
    model = onnx.load(EXPERT)
    initial = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    bank = initial["keys"].copy()
    gain_names = sorted(
        [
            k
            for k in initial
            if k.rsplit("_", 1)[-1].isdigit() and 474 <= int(k.rsplit("_", 1)[-1]) <= 485
        ],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    assert len(gain_names) == 12
    gains = np.asarray([initial[k].item() for k in gain_names])
    scales = np.tile([0.22, 0.20, 0.17, 0.32, 0.26], 5).tolist() + [
        0.14,
        0.27,
        0.055,
        0.14,
        0.32,
        0.065,
        0.15,
        0.18,
        0.25,
        0.28,
        0.05,
        0.10,
    ]
    scale = np.asarray(scales)
    rng = np.random.default_rng(21567)
    groups = ((0, 6), (3, 9), (4, 10), (14, 18), (17, 21))

    def decode(P):
        motion = np.broadcast_to(bank[4], (len(P), 5, 22)).astype(np.float64).copy()
        offsets = P[:, :25].reshape(-1, 5, 5)
        for j, ix in enumerate(groups):
            for joint in ix:
                motion[:, :, joint] += offsets[:, :, j]
        return np.ascontiguousarray(np.clip(motion, lo, hi)), np.ascontiguousarray(
            gains + P[:, 25:]
        )

    def save(p, name):
        m = onnx.load(EXPERT)
        keys, fb = decode(p[None])
        b = bank.copy()
        b[4] = keys[0]
        for t in m.graph.initializer:
            if t.name == "keys":
                t.CopyFrom(numpy_helper.from_array(b, "keys"))
            elif t.name in gain_names:
                t.CopyFrom(
                    numpy_helper.from_array(
                        np.array(fb[0, gain_names.index(t.name)], dtype=np.float32), t.name
                    )
                )
        onnx.checker.check_model(m)
        onnx.save(m, dest / (name + ".onnx"))

    def starts(split, terrain, kind):
        x = np.load(ROOT / "starts_replay" / f"{split}_{terrain}_{kind}.npy")
        assert len(x) > 0 and x.shape[1] == 79
        return x

    # Include static- and push-origin starts in every population evaluation.
    terrains = ("lag", "rough")
    pool = {
        (s, t, k): starts(s, t, k)
        for s in ("collect0", "collect1")
        for t in terrains
        for k in ("static", "relay")
    }
    validation = {
        (t, k): pool["collect1", t, k][
            np.linspace(0, len(pool["collect1", t, k]) - 1, 3, dtype=int)
        ]
        for t in terrains
        for k in ("static", "relay")
    }

    def scores(params, rows):
        all_scores = []
        for t in terrains:
            st = np.concatenate([rows[t, k] for k in ("static", "relay")])
            p = np.repeat(params, len(st), axis=0)
            motion, fb = decode(p)
            vals = np.empty(len(p))
            xml = ROOT / "runs/collect0" / t / "published/static_scene/scene/fall_hard.xml"
            rc = native(
                os.fsencode(xml),
                len(p),
                np.ascontiguousarray(np.tile(st, (len(params), 1))),
                motion,
                stand,
                vals,
                fb,
            )
            assert rc == 0 and np.isfinite(vals).all()
            all_scores.append(vals.reshape(len(params), len(st)))
        v = np.concatenate(all_scores, axis=1)
        return 0.75 * v.mean(1) + 0.25 * np.percentile(v, 20, axis=1)

    mean = np.zeros(len(scale))
    sigma = scale.copy()
    history = []
    block_best = -np.inf
    for gen in range(36):
        pop = mean + sigma * rng.normal(size=(32, len(scale)))
        pop[0] = mean
        rows = {
            (t, k): pool["collect0", t, k][rng.choice(len(pool["collect0", t, k]), 2, replace=True)]
            for t in terrains
            for k in ("static", "relay")
        }
        v = scores(pop, rows)
        elite = pop[np.argsort(v)[-6:]]
        best = pop[v.argmax()]
        val = float(scores(best[None], validation)[0])
        if gen % 12 == 0:
            block_best = -np.inf
        if val > block_best:
            block_best = val
            save(best, "motion_" + str(gen // 12))
        mean = 0.35 * mean + 0.65 * elite.mean(0)
        sigma = np.maximum(0.12 * scale, 0.4 * sigma + 0.6 * elite.std(0))
        history.append(
            dict(
                generation=gen + 1,
                pop_mean=float(v.mean()),
                pop_best=float(v.max()),
                validation=val,
            )
        )
        if gen % 6 == 5:
            print(gen + 1, val, flush=True)
    (dest / "fit.json").write_text(
        json.dumps(
            dict(
                history=history,
                start_counts={"/".join(k): len(v) for k, v in pool.items()},
                base_on_validation=float(scores(np.zeros((1, len(scale))), validation)[0]),
                seed=21567,
                screen_candidates=32 * 36,
            ),
            indent=2,
        )
    )


def select_and_test():
    arms = [f"motion_{i}" for i in range(3)]
    run(["tune"], ["lag", "rough"], arms)
    scores = {}
    for arm in arms:
        stat = []
        dyn = []
        hold = []
        releases = []
        for f in ("lag", "rough"):
            folder = ROOT / "runs/tune" / f / arm
            stat += read(folder / "static/trials.json")
            dyn += read(folder / "relay/relay.json")
            releases += [x["released"] for x in read(folder / "relay/receipt.json")["episodes"]]
        a = sum(x["success"] for x in stat)
        b = sum(x["success"] for x in dyn)
        score = (
            0.45 * a / len(stat)
            + 0.45 * b / len(dyn)
            + 0.05 * np.mean([min(50, x["terminal_stable_ticks"]) / 50 for x in stat])
            + 0.05 * np.mean(releases)
        )
        scores[arm] = dict(static=a, relay=b, score=float(score))
    winner = max(arms, key=lambda k: scores[k]["score"])
    record = dict(
        selected=winner,
        tune=scores,
        onnx_sha256=sha(ROOT / "motion" / (winner + ".onnx")),
        protocol_sha256=sha(ROOT / "protocol.py"),
    )
    (ROOT / "motion_selection.json").write_text(json.dumps(record, indent=2))
    print(record, flush=True)
    run(["test0", "test1"], EVALUATION, [winner])


if __name__ == "__main__":
    fit()
    select_and_test()
