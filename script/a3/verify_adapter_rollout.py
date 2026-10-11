"""Independently check all actual E20 causal reference tokens and outcomes."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.controller_adapter_trial import OUTPUT
from script.a3.joint_forecast_replay import load_predictor
from script.a3.reference_contract import load_sim


def main():
    job = json.loads((OUTPUT / "job.json").read_text())
    experiment = job["experiment"]
    sim = load_sim()
    with np.load(OUTPUT / "training_windows.npz", allow_pickle=False) as dataset:
        permutation = dataset["permutation"].copy()
    rows = []
    for objective in job["objectives"]:
        model = load_predictor(OUTPUT / objective / "adapter_step_000200.npz")
        with np.load(OUTPUT / f"smoke_{objective}/001_walk_front_slow.inputs.npz", allow_pickle=False) as actual:
            with np.load(OUTPUT.parent / "a3_causal_reference_20261010_E18/smoke/001_walk_front_slow.inputs.npz", allow_pickle=False) as old:
                for key in ("observations", "actions"):
                    np.testing.assert_allclose(actual[key], old[key], atol=1e-6, rtol=1e-6)
        for sigma in job["noise_std_rad"]:
            condition = f"{objective}_noise{sigma:g}_seed0"
            directory = OUTPUT / condition
            for source in job["diagnostic_motions"]:
                name = Path(source["path"]).stem
                with np.load(directory / f"{name}.inputs.npz", allow_pickle=False) as stored:
                    observations, actions = stored["observations"].copy(), stored["actions"].copy()
                with np.load(directory / f"{name}.joint.npz", allow_pickle=False) as stored:
                    arrivals, anchors = stored["rows"].copy(), stored["robot_anchor_quat_wxyz"].copy()
                trace = json.loads((directory / f"{name}.timeseries.json").read_text())["motions"][0]
                metric = json.loads((directory / f"{name}.metrics.json").read_text())["motions"][0]
                _, quaternion, q, _ = sim.load_a3_flat_csv(Path(source["path"]), source_fps=30, frame_stride=4)
                times = np.arange(len(q))/30
                resampled = np.arange(0., times[-1]-1e-12, .02)
                clean = np.column_stack([np.interp(resampled, times, q[:, j]) for j in range(29)])[:, permutation]
                observed = q+np.random.default_rng(0).normal(0, sigma, q.shape)
                n = len(observations)
                if observations.shape != (len(clean), 1570) or actions.shape != (n, 29) or anchors.shape != (n, 4):
                    raise ValueError("Actual ABI/full duration/anchor shape mismatch")
                if not np.isfinite(observations).all() or not np.isfinite(actions).all() or not np.isfinite(anchors).all():
                    raise ValueError("Nonfinite actual policy state")
                np.testing.assert_allclose(actions, trace["raw_action_29"], atol=1e-6, rtol=1e-6)
                max_difference, bases = 0., []
                for tick in range(n):
                    native = tick*3//5
                    base = native*5//3
                    if native >= len(q)-1:
                        raise ValueError("EOF requires separate independent contract")
                    bases.append(base)
                    current = observed[native]
                    if native < 5:
                        nodes = np.repeat(current[None], 11, axis=0)
                    else:
                        x = (observed[native-5:native]-current).reshape(145)
                        residual = ((x-model["mean"])/model["std"]) @ model["weights"]+model["bias"]
                        velocity = (current-observed[native-1])*30
                        future = current+np.arange(1, 11)[:, None]/50*velocity+residual.reshape(10, 29)
                        nodes = np.r_[current[None], future]
                    support = np.r_[times[:native], times[native]+np.arange(11)/50]
                    values = np.r_[observed[:native], nodes]
                    requested = (base+np.arange(11))/50
                    predicted = np.column_stack([np.interp(requested, support, values[:, j]) for j in range(29)])[:, permutation]
                    command = np.r_[predicted[:10].reshape(-1), (np.diff(predicted, axis=0)*50).reshape(-1)]
                    if native == 0:
                        root = np.repeat(quaternion[:1], 10, axis=0)
                    else:
                        seen = Rotation.from_quat(quaternion[native-1:native+1, [1, 2, 3, 0]])
                        past = requested[:10] <= times[native]+1e-10
                        root = np.empty((10, 4))
                        if past.any():
                            root[past] = Slerp(times[native-1:native+1], seen)(np.minimum(requested[:10][past], times[native])).as_quat()[:, [3, 0, 1, 2]]
                        if (~past).any():
                            rate = (seen[-1]*seen[-2].inv()).as_rotvec()*30
                            root[~past] = (Rotation.from_rotvec((requested[:10][~past]-times[native])[:, None]*rate)*seen[-1]).as_quat()[:, [3, 0, 1, 2]]
                    orientation = sim.root_ori_diff_6d(anchors[tick], root).reshape(-1)
                    expected = np.r_[command, orientation].astype(np.float32)
                    np.testing.assert_allclose(observations[tick, :640], expected, atol=2e-6, rtol=2e-6)
                    expected_arrival = [tick/50, base/50, tick/50-base/50, native/30, native, 0, max(0., base/50+.2-native/30)]
                    np.testing.assert_allclose(arrivals[tick], expected_arrival, atol=1e-10, rtol=0)
                    max_difference = max(max_difference, float(np.abs(observations[tick, :640]-expected).max()))
                target = clean[np.asarray(bases)]
                np.testing.assert_allclose(trace["reference_q_29"], target, atol=1e-9, rtol=0)
                rmse = float(np.sqrt(np.mean((np.asarray(trace["q_state_29"])-target)**2)))
                if not math.isclose(rmse, metric["tracking"]["all_29_rmse"], rel_tol=1e-9):
                    raise ValueError("Actual joint metric independent recomputation failed")
                root_error = np.linalg.norm(np.asarray(trace["sim_root_pos_w"])-np.asarray(trace["ref_root_pos_w"]), axis=-1)
                np.testing.assert_allclose(root_error, trace["root_pos_error_m"], atol=1e-12, rtol=1e-10)
                rows.append(dict(condition=condition, motion=name, actual_reference_tokens_verified=n,
                    max_abs_difference=max_difference, independent_joint_rmse_rad=rmse,
                    inputs_sha256=sha256(directory / f"{name}.inputs.npz"),
                    timeseries_sha256=sha256(directory / f"{name}.timeseries.json")))
    result = dict(status="passed", experiment=experiment, actual_reference_inputs_verified=sum(r["actual_reference_tokens_verified"] for r in rows),
        zero_adapter_smoke_matches_e18_cv=True, rows=rows, proprio_independently_rebuilt=False,
        initialization_causal_verified=False, scope="RUNNING_REFERENCE_TOKENS_AND_RECOMPUTED_JOINT_METRICS_ONLY",
        code_sha256=sha256(Path(__file__)), backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "independent_rollout.json", result)
    print(json.dumps({k: result[k] for k in ("status", "actual_reference_inputs_verified", "zero_adapter_smoke_matches_e18_cv")}), flush=True)


if __name__ == "__main__":
    main()
