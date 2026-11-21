"""Fit the supported prone rise on disjoint starts by cross-entropy search."""

import ctypes
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "h003_recovery_expert"
sys.path.insert(0, str(OLD))
import bench
from bench import mujoco
from numpy.ctypeslib import ndpointer

BUNDLE = Path(os.environ["Q1_V2B_BUNDLE"])
CFG = json.loads((BUNDLE / "config/q1_deploy_config.json").read_text())
STAND, LO, HI = [np.array(CFG[k]) for k in ("default_q", "q_lo", "q_hi")]
PAIRS = [
    (0, 6, 1),
    (1, 7, -1),
    (2, 8, -1),
    (3, 9, 1),
    (4, 10, 1),
    (5, 11, -1),
    (14, 18, 1),
    (15, 19, -1),
    (16, 20, -1),
    (17, 21, 1),
]
GROUPS = [
    (0, 6, 1),
    (3, 9, 1),
    (4, 10, 1),
    (14, 18, 1),
    (17, 21, 1),
    (15, 19, -1),
    (1, 7, -1),
]


def measured(name, seconds):
    with np.load(Path.home() / "projects/Q1_deploy/real_logs" / name) as log:
        order = [log["joint_names"].tolist().index(j) for j in CFG["joint_names"]]
        t = (log["joint_state_stamp"] - log["event_ns"].item()) * 1e-9
        return np.stack([np.interp(seconds, t, log["q"][:, j]) for j in order], axis=1)


def templates(style):
    keys = measured(
        "q1_official_standup_from_mcap.npz", [2.5, 3.7, 5.0, 5.9, 6.55, 6.98, 7.35, 7.8]
    )
    symmetric = keys.copy()
    for left, right, sign in PAIRS:
        symmetric[:, left] = (keys[:, left] + sign * keys[:, right]) / 2
        symmetric[:, right] = sign * symmetric[:, left]
    symmetric[:, 12:14] = 0
    if style == "symmetric":
        keys = symmetric
    if style == "teacher":
        with np.load(
            HERE.parent / "q1_recovery_review_20261121/native/static/trial_00.npz"
        ) as teacher:
            keys = teacher["qpos"][[16, 36, 70, 135, 155, 172, 183, 196], 7:].copy()
    roll = measured(
        "q1_official_standup_faceup_20261120.npz", [2.6, 3.05, 3.55, 4.05, 4.65, 5.1]
    )
    if style.startswith("supine"):
        import torch

        checkpoint = torch.load(
            OLD / "ppo_residual_01/actor_00200.pt", map_location="cpu"
        )
        keys = np.vstack([STAND, checkpoint["keys"][4].numpy(), STAND, STAND])
    return keys, roll


def fit(
    out,
    style,
    discovery=False,
    parent=None,
    limbs=False,
    toward_crouch=False,
    snapshots=None,
):
    out.mkdir(exist_ok=False)
    rng = np.random.default_rng(281103)
    if limbs:
        from limbs import add_limb_contacts

        model = add_limb_contacts(out / "scene")
    else:
        model = bench.scene(out / "scene")
    ctrl = bench.Q1V2bOnnxController(BUNDLE, ())

    def starts(seed):
        result = []
        for pose, qpos in bench.poses(ctrl, seed):
            if pose != ("supine" if style.startswith("supine") else "prone"):
                continue
            d = mujoco.MjData(model)
            d.qpos[:] = qpos
            d.ctrl[:] = qpos[7:]
            mujoco.mj_forward(model, d)
            mujoco.mj_step(model, d, nstep=20)
            result.append(np.r_[d.qpos, d.qvel])
        return np.array(result)

    train = np.concatenate([starts(s) for s in (7521, 7522, 7523)])
    val = starts(7524)[[0, 3, 6]]
    if snapshots is not None:
        train = np.concatenate(
            [np.load(snapshots / f"{s}.npy") for s in (7521, 7522, 7523)]
        )
        val = np.load(snapshots / "7524.npy")[[0, 3, 6]]
    template, roll = templates(style)
    duration = np.array([23, 42, 48, 32, 27, 27, 31, 28], dtype=np.int32)
    if style.startswith("supine"):
        duration = np.array([18, 16, 16, 16, 16, 18, 22, 26], dtype=np.int32)
    if style == "supine_fast":
        duration = np.array([16, 13, 13, 13, 13, 13, 16, 18], dtype=np.int32)
    crouch = np.zeros(25)
    if toward_crouch:
        if not limbs or style != "asymmetric":
            raise ValueError(
                "crouch search uses the prone hardware reference with arm-leg contacts"
            )
        name = "q1_official_standup_from_mcap.npz"
        template = measured(name, [2.58, 3.45, 4.05, 4.75, 5.35, 5.98, 6.43, 6.80])
        duration = np.array([20, 40, 30, 34, 30, 31, 23, 19], dtype=np.int32)
        with np.load(Path.home() / "projects/Q1_deploy/real_logs" / name) as log:
            idx = np.argmin(
                abs((log["imu_stamp"] - log["event_ns"].item()) * 1e-9 - 6.80)
            )
            x, y, z, w = log["orientation_xyzw"][idx]
        crouch = np.r_[
            template[-1],
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ]
    if snapshots is not None:
        template = measured(name, [6.58, 6.68, 6.83, 6.96, 7.05, 7.15, 7.25, 7.38])
        duration = np.array([12, 15, 16, 15, 14, 17, 15, 18], dtype=np.int32)
        with np.load(Path.home() / "projects/Q1_deploy/real_logs" / name) as log:
            idx = np.argmin(
                abs((log["imu_stamp"] - log["event_ns"].item()) * 1e-9 - 7.05)
            )
            x, y, z, w = log["orientation_xyzw"][idx]
        crouch = np.r_[
            measured(name, [7.05])[0],
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ]
    base_gain = np.array([0.0, 0.0, 0.6, 0.06, 0.45, 0.04])
    if parent is not None:
        with np.load(parent) as prior:
            template, base_gain, duration = (
                prior["keys"],
                prior["gain"],
                prior["duration"],
            )
    n = len(GROUPS)
    scales = np.r_[
        np.tile([0.35, 0.35, 0.30, 0.58, 0.37, 0.31, 0.17], 7),
        [0.8, 0.14, 0.7, 0.12, 0.4, 0.08],
    ]
    if parent is not None and not limbs:
        scales[: 3 * n] = 0.0
        scales[3 * n : 7 * n] *= 1.7
        scales[-6:] *= 1.6

    def decode(pars):
        keys = np.broadcast_to(template, (len(pars), 8, 22)).copy()
        variation = pars[:, : 7 * n].reshape(-1, 7, n)
        for j, (left, right, sign) in enumerate(GROUPS):
            keys[:, 1:, left] += variation[:, :, j]
            keys[:, 1:, right] += sign * variation[:, :, j]
        return np.ascontiguousarray(keys.clip(LO, HI)), np.ascontiguousarray(
            pars[:, 7 * n :] + base_gain
        )

    lib = ctypes.CDLL(str(HERE / "search.so"))
    ptr = ndpointer(dtype="float64", flags="C_CONTIGUOUS")
    lib.evaluate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        ptr,
        ndpointer(dtype="int32", flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ptr,
        ptr,
    ]
    scene_path = str((out / "scene/fall_hard.xml").resolve()).encode()

    def evaluate(params, rows):
        k, g = decode(params)
        result = np.empty((len(k), 5))
        if lib.evaluate(
            scene_path,
            len(k),
            np.ascontiguousarray(rows),
            k,
            g,
            STAND,
            duration,
            3 if toward_crouch else 2 if parent is not None else int(discovery),
            crouch,
            result,
        ):
            raise RuntimeError("physics evaluator failed")
        return result

    mean = np.zeros_like(scales)
    sigma = scales.copy()
    best = -np.inf
    winner = mean.copy()
    for epoch in range(65 if toward_crouch or style.startswith("supine") else 42):
        pool = mean + sigma * rng.normal(size=(36, len(mean)))
        pool[0] = mean
        if epoch:
            pool[1] = winner
        result = evaluate(
            pool, np.repeat(train[rng.integers(len(train))][None], len(pool), axis=0)
        )
        elite = pool[np.argsort(result[:, 0])[-6:]]
        selected = elite[-2:]
        values = evaluate(
            np.repeat(selected, len(val), axis=0), np.tile(val, (2, 1))
        ).reshape(2, len(val), 5)
        scores = 0.5 * values[:, :, 0].mean(1) + 0.5 * values[:, :, 0].min(1)
        i = scores.argmax()
        if scores[i] > best:
            best = float(scores[i])
            winner = selected[i].copy()
            keys, g = decode(winner[None])
            np.savez(
                out / "fit.npz", keys=keys[0], gain=g[0], roll=roll, duration=duration
            )
        print(
            json.dumps(
                dict(
                    epoch=epoch,
                    train=float(result[:, 0].max()),
                    validation=float(scores[i]),
                    best=best,
                    calm=float(values[i, :, 1].min()),
                    speed=float(values[i, :, 2].max()),
                    depth_mm=float(values[i, :, 3].max()),
                )
            ),
            flush=True,
        )
        mean = 0.25 * mean + 0.75 * elite.mean(0)
        sigma = np.maximum(0.15 * scales, 0.4 * sigma + 0.6 * elite.std(0))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("style", nargs="?", default="symmetric")
    parser.add_argument("--discovery", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--limbs", action="store_true")
    parser.add_argument("--toward-crouch", action="store_true")
    parser.add_argument("--snapshots", type=Path)
    args = parser.parse_args()
    if args.snapshots and not args.toward_crouch:
        parser.error("bridge fitting requires --toward-crouch")
    fit(
        args.output.resolve(),
        args.style,
        args.discovery,
        args.parent,
        args.limbs,
        args.toward_crouch,
        args.snapshots,
    )
