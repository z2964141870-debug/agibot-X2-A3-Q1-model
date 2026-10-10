"""Independent E16 source/metric reconstruction and E17 command dynamics audit."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import ROOT, VENDOR
from script.a3.reference_contract import load_sim


SOURCE = ROOT / "data/experiments/a3_joint_ablation_20261010_E16"
OUTPUT = ROOT / "data/experiments/a3_joint_dynamics_20261010_E17"


def independent_window(observed, index, base, mode, model):
    arrived = observed[:index+1]
    times = np.arange(index+1) / 30
    query = base/50 + np.arange(11)/50
    current = arrived[-1].copy()
    if index < 5 or mode == "hold":
        forecast = np.tile(current, (10, 1))
    elif mode == "cv":
        slope = (arrived[-1]-arrived[-2]) * 30
        forecast = current + np.arange(1, 11)[:, None]/50 * slope
    elif mode == "smooth_cv":
        t = np.arange(-5, 1)/30
        slope = ((t-t.mean())[:, None] * (arrived[-6:]-arrived[-6:].mean(0))).sum(0) / np.sum((t-t.mean())**2)
        current = arrived[-6:].mean(0) - slope*t.mean()
        forecast = current + np.arange(1, 11)[:, None]/50 * slope
    elif mode in ("E12", "E14"):
        feature = (arrived[-6:-1]-current).reshape(145)
        predicted = ((feature-model["mean"])/model["std"]) @ model["weights"] + model["bias"]
        forecast = current + predicted.reshape(10, 29)
    else:
        raise ValueError("Unsupported causal command")
    support_t = np.concatenate((times[:-1], times[-1]+np.arange(11)/50))
    support_q = np.concatenate((arrived[:-1], current[None], forecast))
    return np.column_stack([np.interp(query, support_t, support_q[:, j]) for j in range(29)])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "result.json").exists():
        print("E17 already complete; result preserved")
        return 0
    job = json.loads((SOURCE / "job.json").read_text())
    expected = json.loads((SOURCE / "result.json").read_text())
    sim = load_sim()
    asset = VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
    mapping = sim.build_loop_runtime(mujoco.MjModel.from_xml_path(str(asset)), sim.DEFAULT_URDF).mapping
    permutation = mapping.mj29_to_il
    models = {}
    for mode in ("E12", "E14"):
        with np.load(ROOT / f"data/experiments/a3_future_reference_20261010_{mode}/ridge_joint_future.npz", allow_pickle=False) as stored:
            models[mode] = {key: stored[key].copy() for key in ("mean", "std", "weights", "bias")}
    receipts, aggregate = [], {}
    smoke = np.load(SOURCE / "smoke/001_walk_front_slow.inputs.npz", allow_pickle=False)
    if smoke["observations"].shape != (10, 1570) or smoke["actions"].shape != (10, 29):
        raise ValueError("Smoke did not capture the actual policy ABI")
    for condition in job["conditions"]:
        mode, sigma = condition["mode"], condition["noise_std_rad"]
        sums = dict(position_sq=0., velocity_sq=0., command_count=0, jump_sq=0., jump_count=0,
                    action_sq=0., action_count=0, action_jump_sq=0., action_jump_count=0, action_clip_count=0,
                    tracking_sq=0., tracking_count=0, steps=0)
        for source in job["motions"]:
            directory = SOURCE / condition["label"]
            name = Path(source["path"]).stem
            trace_path = directory / f"{name}.timeseries.json"
            metric_path = directory / f"{name}.metrics.json"
            trace = json.loads(trace_path.read_text())["motions"][0]
            metrics = json.loads(metric_path.read_text())["motions"][0]
            with np.load(directory / f"{name}.joint.npz", allow_pickle=False) as stored:
                actual_arrivals, actual_errors = stored["rows"].copy(), stored["errors"].copy()
            if sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("Registered source changed")
            root, _, q, _ = sim.load_a3_flat_csv(Path(source["path"]), source_fps=30, frame_stride=4)
            native_times = np.arange(len(q))/30
            sampled_times = np.arange(0., native_times[-1]-1e-12, .02)
            clean = np.column_stack([np.interp(sampled_times, native_times, q[:, j]) for j in range(29)])[:, permutation]
            velocity = np.diff(clean, axis=0)*50
            velocity = np.concatenate((velocity, velocity[-1:]))
            noise = np.random.default_rng(0).normal(0, sigma, q.shape)
            observed = q+noise
            n = len(trace["policy_tick"])
            if n != len(clean) or not np.array_equal(trace["policy_tick"], np.arange(n)):
                raise ValueError("Full replay count/ticks differ from source duration")
            positions, velocities, bases, arrivals, input_errors = [], [], [], [], []
            for tick in range(n):
                index = min(tick*3//5, len(q)-1)
                if index == len(q)-1:
                    raise ValueError("Unexpected EOF case; must audit separately")
                base = min(index*5//3, n-1)
                ids = np.minimum(base+np.arange(11), n-1)
                if mode == "oracle_aligned":
                    pose, vel = clean[ids[:10]], velocity[ids[:10]]
                else:
                    predicted = independent_window(observed, index, base, mode, models.get(mode))[:, permutation]
                    pose, vel = predicted[:10], np.diff(predicted, axis=0)*50
                positions.append(pose)
                velocities.append(vel)
                error = pose-clean[ids[:10]]
                input_errors.append([np.sum(error**2), error.size, np.abs(error).max()])
                bases.append(base)
                arrivals.append([tick/50, base/50, tick/50-base/50, index/30, index, 0,
                                 max(0., base/50+.2-index/30)])
                if condition["label"] == "cv_noise0_seed0" and name == "001_walk_front_slow" and tick < 10:
                    command = np.concatenate((pose.reshape(-1), vel.reshape(-1)))
                    np.testing.assert_allclose(smoke["observations"][tick, :580], command, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(actual_arrivals, arrivals, atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual_errors, input_errors, atol=1e-9, rtol=1e-8)
            target = clean[np.asarray(bases)]
            np.testing.assert_allclose(trace["reference_q_29"], target, atol=1e-9, rtol=0)
            state, action = np.asarray(trace["q_state_29"]), np.asarray(trace["raw_action_29"])
            root_error = np.linalg.norm(np.asarray(trace["sim_root_pos_w"])-np.asarray(trace["ref_root_pos_w"]), axis=1)
            np.testing.assert_allclose(root_error, trace["root_pos_error_m"], atol=1e-12, rtol=1e-10)
            rmse = float(np.sqrt(np.mean((state-target)**2)))
            if not math.isclose(rmse, metrics["tracking"]["all_29_rmse"], rel_tol=1e-9):
                raise ValueError("Independent robot joint error differs from metrics")
            poses, vels = np.asarray(positions), np.asarray(velocities)
            future_ids = np.minimum(np.asarray(bases)[:, None]+np.arange(10), n-1)
            true_pos, true_vel = clean[future_ids], velocity[future_ids]
            for key, value in dict(position_sq=np.sum((poses-true_pos)**2), velocity_sq=np.sum((vels-true_vel)**2),
                    command_count=poses.size, jump_sq=np.sum(np.diff(poses[:, 0], axis=0)**2), jump_count=(n-1)*29,
                    action_sq=np.sum(action**2), action_count=action.size,
                    action_jump_sq=np.sum(np.diff(action, axis=0)**2), action_jump_count=(n-1)*29,
                    action_clip_count=np.sum(np.abs(action) >= 20), tracking_sq=np.sum((state-target)**2),
                    tracking_count=state.size, steps=n).items():
                sums[key] += float(value)
            receipts.append(dict(condition=condition["label"], motion=name, frames=n,
                arrival_guard_passed=True, clean_target_rebuilt=True, independent_joint_rmse_rad=rmse,
                metrics_sha256=sha256(metric_path), timeseries_sha256=sha256(trace_path),
                future_velocity_rmse_rad_s=float(np.sqrt(np.mean((vels-true_vel)**2))),
                action_abs_p95=float(np.quantile(np.abs(action), .95)),
                action_delta_rms=float(np.sqrt(np.mean(np.diff(action, axis=0)**2)))))
        aggregate[condition["label"]] = dict(position_rmse_rad=math.sqrt(sums["position_sq"]/sums["command_count"]),
            velocity_rmse_rad_s=math.sqrt(sums["velocity_sq"]/sums["command_count"]),
            current_command_jump_rms_rad=math.sqrt(sums["jump_sq"]/sums["jump_count"]),
            action_rms=math.sqrt(sums["action_sq"]/sums["action_count"]),
            action_delta_rms=math.sqrt(sums["action_jump_sq"]/sums["action_jump_count"]),
            action_clip_fraction=sums["action_clip_count"]/sums["action_count"],
            robot_joint_rmse_rad=math.sqrt(sums["tracking_sq"]/sums["tracking_count"]), frames=int(sums["steps"]))
        if not math.isclose(aggregate[condition["label"]]["robot_joint_rmse_rad"],
                            expected["results"][condition["label"]]["joint_rmse_rad"], rel_tol=1e-9):
            raise ValueError("Independent aggregate differs from E16")
    smoke.close()
    result = dict(experiment="E17", status="complete", code_sha256=sha256(Path(__file__)),
        source_job_sha256=sha256(SOURCE / "job.json"), verified_motions=len(receipts),
        actual_smoke_inputs_verified=10, aggregate=aggregate, per_motion=receipts,
        scope="INDEPENDENT_DIAGNOSTIC_RECOMPUTATION_NOT_CAUSAL_ATTRIBUTION_OR_POLICY_UPGRADE",
        policy_updates=0, backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "result.json", result)
    atomic_json(ROOT / "data/manifests/a3_joint_dynamics_20261010_E17.json", result)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
