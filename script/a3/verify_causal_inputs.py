"""Independently rebuild all E18 command/orientation tokens from arrived samples."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import ROOT, VENDOR
from script.a3.reference_contract import load_sim
from script.a3.verify_joint_dynamics import independent_window


OUTPUT = ROOT / "data/experiments/a3_causal_reference_20261010_E18"


def main():
    job = json.loads((OUTPUT / "job.json").read_text())
    result = json.loads((OUTPUT / "result.json").read_text())
    sim = load_sim()
    model = mujoco.MjModel.from_xml_path(str(VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"))
    runtime = sim.build_loop_runtime(model, sim.DEFAULT_URDF)
    permutation = runtime.mapping.mj29_to_il
    with np.load(ROOT / "data/experiments/a3_future_reference_20261010_E14/ridge_joint_future.npz", allow_pickle=False) as stored:
        predictor = {key: stored[key].copy() for key in ("mean", "std", "weights", "bias")}
    rows = []
    initializations = []
    for condition in job["conditions"]:
        for source in job["motions"]:
            name = Path(source["path"]).stem
            directory = OUTPUT / condition["label"]
            trace_path = directory / f"{name}.joint.npz"
            inputs_path = directory / f"{name}.inputs.npz"
            with np.load(trace_path, allow_pickle=False) as stored:
                arrivals, anchors = stored["rows"].copy(), stored["robot_anchor_quat_wxyz"].copy()
            with np.load(inputs_path, allow_pickle=False) as stored:
                observations, actions = stored["observations"].copy(), stored["actions"].copy()
            trace = json.loads((directory / f"{name}.timeseries.json").read_text())["motions"][0]
            qroot, quaternion, q, _ = sim.load_a3_flat_csv(Path(source["path"]), source_fps=30, frame_stride=4)
            noise = np.random.default_rng(0).normal(0, condition["noise_std_rad"], q.shape)
            observed = q+noise
            n = len(observations)
            if anchors.shape != (n, 4) or not np.isfinite(anchors).all() or not np.allclose(np.linalg.norm(anchors, axis=1), 1, atol=1e-6, rtol=0):
                raise ValueError("Incomplete/nonunit actual robot-anchor captures")
            if observations.shape != (n, 1570) or actions.shape != (n, 29) or arrivals.shape != (n, 7):
                raise ValueError("Incomplete actual policy inputs")
            np.testing.assert_allclose(actions, trace["raw_action_29"], atol=1e-6, rtol=1e-6)
            latest = np.minimum(np.arange(n)*3//5, len(q)-1)
            if latest.max() == len(q)-1:
                raise ValueError("EOF handling needs a separate independent verifier")
            expected_source_read = latest/30
            np.testing.assert_allclose(arrivals[:, 3], expected_source_read, atol=1e-10, rtol=0)
            max_difference = 0.
            for tick, index in enumerate(latest):
                base = int(index*5//3)
                predicted = independent_window(observed, index, base, condition["mode"], predictor)[:, permutation]
                position, velocity = predicted[:10], np.diff(predicted, axis=0)*50
                command = np.concatenate((position.reshape(-1), velocity.reshape(-1)))
                times = np.arange(index+1)/30
                requested = base/50+np.arange(10)/50
                seen = Rotation.from_quat(quaternion[:index+1, [1, 2, 3, 0]])
                current = seen[-1]
                past = requested <= index/30+1e-10
                root_pred = np.empty((10, 4))
                if index == 0:
                    root_pred[:] = quaternion[0]
                else:
                    if past.any():
                        root_pred[past] = Slerp(times, seen)(np.minimum(requested[past], times[-1])).as_quat()[:, [3, 0, 1, 2]]
                    if (~past).any():
                        relative = current * seen[-2].inv()
                        rate = relative.as_rotvec() * 30
                        root_pred[~past] = (Rotation.from_rotvec((requested[~past]-times[-1])[:, None]*rate) * current).as_quat()[:, [3, 0, 1, 2]]
                ori = sim.root_ori_diff_6d(anchors[tick], root_pred).reshape(-1)
                expected = np.concatenate((command, ori)).astype(np.float32)
                actual = observations[tick, :640]
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                max_difference = max(max_difference, float(np.abs(actual-expected).max()))
            # The common official simulator reset uses a forward-difference qvel.
            initializations.append(dict(condition=condition["label"], motion=name,
                native_root_displacement_first_frame_m=float(np.linalg.norm(qroot[1]-qroot[0])),
                native_joint_velocity_first_frame_norm_rad_s=float(np.linalg.norm((q[1]-q[0])*30)),
                scope="OFFLINE_REFERENCE_RESET_QVEL; NOT_VERIFIED_CAUSAL_INITIALIZATION"))
            rows.append(dict(condition=condition["label"], motion=name, policy_inputs=n,
                future_source_reads_for_policy_tokens=0, command_and_orientation_max_abs_difference=max_difference,
                actions_match_timeseries=True, source_sha256=sha256(Path(source["path"])),
                inputs_sha256=sha256(inputs_path), trace_sha256=sha256(trace_path)))
    receipt = dict(status="passed", experiment="E18", actual_policy_inputs_verified=sum(r["policy_inputs"] for r in rows),
        conditions_verified=len(job["conditions"]), motions_verified=len(rows), rows=rows,
        job_sha256=sha256(OUTPUT / "job.json"), result_sha256=sha256(OUTPUT / "result.json"),
        code_sha256=sha256(Path(__file__)), scope="ACTUAL_POLICY_REFERENCE_TOKENS_CAUSAL; MATCHED_OFFLINE_INITIALIZATION",
        initialization_causal_verified=False, initialization_diagnostics=initializations,
        all_replays_completed=all(v["full_policy_reference_causal_guard_passed"] for v in result["results"].values()),
        policy_updates=0, backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "independent_inputs.json", receipt)
    atomic_json(ROOT / "data/manifests/a3_causal_inputs_verified_20261010_E18.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "actual_policy_inputs_verified", "motions_verified", "initialization_causal_verified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
