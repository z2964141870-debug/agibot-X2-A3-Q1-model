"""Reconstruct boundary PD demand from saved states; never integrate physics."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "project_tools"), str(HERE.parent / "supermimic_q1_sole")]
from prepare_q1_gait_review import FIXED, identity, write_csv
from reward_signal import saved_clip
from saved_control_usage import phase_rows


def preceding_states(initial, states):
    initial, states = np.asarray(initial), np.asarray(states)
    if (initial.shape != (57,) or states.ndim != 2 or states.shape[1] != 57 or not len(states)
            or not np.isfinite(initial).all() or not np.isfinite(states).all()):
        raise ValueError("Expected initial57 and finite nonempty post-states57")
    return np.concatenate((initial[None], states[:-1]), axis=0)


def pd_demand(target, position, velocity, kp, kv, force_range):
    target, position, velocity, kp, kv, force_range = [
        np.asarray(value, dtype=np.float64) for value in (target, position, velocity, kp, kv, force_range)]
    if (target.ndim != 2 or not target.size or position.shape != target.shape or velocity.shape != target.shape
            or kp.shape != target.shape[1:] or kv.shape != kp.shape or force_range.shape != (len(kp), 2)
            or not all(np.isfinite(value).all() for value in (target, position, velocity, kp, kv, force_range))
            or (kp <= 0).any() or (kv < 0).any() or (force_range[:, 0] >= 0).any()
            or (force_range[:, 1] <= 0).any()):
        raise ValueError("Expected finite aligned targets/q/qd, positive KP, nonnegative KV and signed force limits")
    requested = kp * (target - position) - kv * velocity
    clipped = np.clip(requested, force_range[:, 0], force_range[:, 1])
    ratio = np.where(requested >= 0, requested / force_range[:, 1], requested / force_range[:, 0])
    return requested, clipped, ratio


def model_contract(model, names):
    if len(names) != 22 or len(set(names)) != 22 or model.nu != 22 or model.na != 0:
        raise ValueError("Expected native22 direct position actuators without activation state")
    joints = np.array([model.joint(name).id for name in names])
    actuators = []
    for joint in joints:
        found = np.flatnonzero((model.actuator_trntype == 0) & (model.actuator_trnid[:, 0] == joint))
        if len(found) != 1:
            raise ValueError("Expected exactly one joint-transmission actuator per joint")
        actuators.append(int(found[0]))
    a = np.array(actuators)
    kp, kv = model.actuator_gainprm[a, 0], -model.actuator_biasprm[a, 2]
    gear = np.zeros((22, 6))
    gear[:, 0] = 1
    gain, bias = np.zeros((22, 10)), np.zeros((22, 10))
    gain[:, 0], bias[:, 1], bias[:, 2] = kp, -kp, -kv
    checks = [np.array_equal(model.actuator_gear[a], gear), np.array_equal(model.actuator_gainprm[a], gain),
        np.array_equal(model.actuator_biasprm[a], bias), (model.actuator_dyntype[a] == 0).all(),
        (model.actuator_gaintype[a] == 0).all(), (model.actuator_biastype[a] == 1).all(),
        not model.actuator_ctrllimited[a].any(), model.actuator_forcelimited[a].all(),
        not model.actuator_delay[a].any(), not model.actuator_history[a].any(),
        not model.jnt_actfrclimited[joints].any(), not model.jnt_actgravcomp[joints].any(),
        not model.body_gravcomp.any(), (model.jnt_type[joints] == 3).all(),
        int(model.opt.disableflags) == 0, int(model.opt.enableflags) == 0,
        np.isclose(model.opt.timestep, .005, rtol=0, atol=1e-12)]
    if not all(checks):
        raise ValueError("Model is not the zero-delay, unit-gear, unclamped-control direct PD contract")
    limits = model.actuator_forcerange[a].copy()
    pd_demand(np.zeros((1, 22)), np.zeros((1, 22)), np.zeros((1, 22)), kp, kv, limits)
    return dict(joint_ids=joints, actuator_ids=a, qpos_ids=model.jnt_qposadr[joints],
                qvel_ids=model.jnt_dofadr[joints], kp=kp.copy(), kv=kv.copy(), force_range=limits)


def metrics(requested, clipped, ratio, residual_change):
    return dict(elements=int(requested.size), requested_abs_p95_Nm=float(np.quantile(np.abs(requested), .95)),
        requested_abs_max_Nm=float(np.abs(requested).max()), limit_ratio_p95=float(np.quantile(ratio, .95)),
        limit_ratio_max=float(ratio.max()), boundary_over_limit_elements=int((ratio > 1).sum()),
        boundary_over_limit_fraction=float((ratio > 1).mean()),
        clipping_abs_max_Nm=float(np.abs(requested - clipped).max()),
        residual_held_state_force_change_rms_Nm=float(np.sqrt(np.mean(residual_change ** 2))))


def crosscheck_actuation(mujoco, model, data, contract, pre, target, expected):
    # Only joint transmission/actuation is evaluated; root/contact dynamics are not replayed.
    before = float(data.time)
    data.qpos[:] = model.keyframe("stand").qpos
    data.qvel[:] = 0
    data.qpos[contract["qpos_ids"]] = pre[13:35]
    data.qvel[contract["qvel_ids"]] = pre[35:57]
    data.ctrl[contract["actuator_ids"]] = target
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    mujoco.mj_fwdActuation(model, data)
    actual = data.actuator_force[contract["actuator_ids"]]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
    np.testing.assert_allclose(data.qfrc_actuator[contract["qvel_ids"]], expected, rtol=0, atol=2e-10)
    if data.time != before:
        raise ValueError("Actuation-only crosscheck advanced time")
    return float(np.max(np.abs(actual - expected)))


def plot(output, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col", constrained_layout=True)
    for col, clip in enumerate(FIXED[:2]):
        for line, color in (("base", "#087f8c"), ("ppo", "#b55034")):
            labels = list(dict.fromkeys(key[1] for key in series if key[0] == line))
            for label in ("initial", labels[-1]):
                values = np.median(np.stack([series[line, label, repeat, clip] for repeat in range(3)]), axis=0)
                seconds = np.arange(len(values)) * .02
                for row, joint in enumerate((6, 9, 10)):
                    axes[row, col].plot(seconds, values[:, joint], color=color,
                        ls=":" if label == "initial" else "-", lw=1, label=f"{line} {label}")
                for frame, value in enumerate(values):
                    rows.append(dict(line=line, label=label, clip=clip, control_start_seconds=float(seconds[frame]),
                        right_hip_pitch_request_Nm=float(value[6]), right_knee_request_Nm=float(value[9]),
                        right_ankle_pitch_request_Nm=float(value[10])))
        for row, (name, limit) in enumerate((("Right hip pitch", 36), ("Right knee", 36), ("Right ankle pitch", 22))):
            axes[row, col].axhline(limit, color="#444444", ls="--", lw=.7)
            axes[row, col].axhline(-limit, color="#444444", ls="--", lw=.7)
            axes[row, col].set_ylabel(name + " (Nm)")
            axes[row, col].grid(alpha=.2)
        axes[0, col].set_title("Stop walk" if col == 0 else "Start walk")
        axes[-1, col].set_xlabel("Control-start time (s)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Saved Q1 boundary PD demand: framewise medians over three repeats", fontsize=12)
    fig.supxlabel("Algebraic request at saved pre-state, not recorded substep torque. Dashed lines: model force limits.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(output / ("boundary_pd_demand." + extension), dpi=150)
    plt.close(fig)
    write_csv(output / "median_boundary_timeseries.csv", rows)
    return len(rows)


def run(args):
    import mujoco

    if args.output.exists():
        raise FileExistsError(args.output)
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    prior_path = args.control / "summary.json"
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "SAVED_CONTROL_USAGE_REVIEW_READY" or prior["saved_clips"] != 45
            or prior["new_integration_steps"] or prior["policy_inference_calls"]):
        raise ValueError("Expected completed saved-only control analysis")
    historical = {row["path"]: row for row in prior["trace_sources"]}
    previous_sources = {row["path"]: row for row in prior["sources"]}
    tracked = {}
    def remember(path, expected=None):
        value = identity(path)
        if expected is not None and value != expected:
            raise ValueError("Saved source lineage differs: " + str(path))
        tracked[str(path)] = value
        return value
    remember(Path(__file__))
    for path in (HERE / "saved_control_usage.py", HERE.parent / "supermimic_q1_sole/reward_signal.py",
                 HERE.parent / "project_tools/prepare_q1_gait_review.py", prior_path):
        remember(path, previous_sources.get(str(path)))
    scene, config_path = args.runtime / "q1_assets/scene.xml", args.runtime / "q1_config.yaml"
    for path in (scene, config_path, native / "control.json", native / "manager_terms.py",
                 HERE.parent / "supermimic_q1_gait_comp/physical.py"):
        if str(path) not in previous_sources:
            raise ValueError("Missing prior frozen-source lineage")
        remember(path, previous_sources[str(path)])
    for path in (args.phase / "summary.json", args.phase / "foot_timeseries.csv"):
        remember(path, previous_sources[str(path)])
    config = json.loads(config_path.read_text())["env"]
    for name in ("startup_context", "reset_context"):
        if config["events"][name]["params"]["randomize_pd_gains"] is not False:
            raise ValueError("PD-gain randomization not disabled")
    if config["ctrl_dt"] != .02 or config["sim_dt"] != .005:
        raise ValueError("Expected20ms control and5ms model step")
    package_root = Path(importlib.util.find_spec("unilab").origin).parent.parent
    backend_paths = [package_root / name for name in (
        "unilab/base/entity.py", "unilab/base/torch_env.py", "unilab/envs/manager_based_rl_env.py",
        "unisim/backend/mjwarp/backend.py", "mujoco_warp/_src/forward.py")]
    backend_identities = [remember(path) for path in backend_paths]
    names = json.loads((native / "control.json").read_text())["joints"]
    model = mujoco.MjModel.from_xml_path(str(scene))
    contract, data = model_contract(model, names), mujoco.MjData(model)
    kp, kv, limits = (contract[key] for key in ("kp", "kv", "force_range"))
    phases = phase_rows(args.phase / "foot_timeseries.csv")
    joint_rows, foot_rows, traces, series, crosschecks = [], [], [], {}, []
    npz_fields = None
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        selection_path, library_path = trial / "run/checkpoint_selection.json", trial / "evaluation/evaluation_library.json"
        for path in (selection_path, library_path, trial / "plan.json"):
            remember(path, previous_sources[str(path)])
        plan = json.loads((trial / "plan.json").read_text())
        plan_paths = {row["path"] for row in plan["source"]}
        if any(str(path) in plan_paths for path in backend_paths):
            raise ValueError("Revisit backend historical provenance: an expected gap has changed")
        selection = json.loads(selection_path.read_text())
        if selection["user_accepted_checkpoint"] is not None:
            raise ValueError("Expected original null adoption")
        library = json.loads(library_path.read_text())["splits"]["train"]
        indices = [next(index for index, row in enumerate(library) if row["name"] == clip) for clip in FIXED]
        for candidate in selection["candidates"]:
            label = candidate["label"]
            for repeat, trace_info in enumerate(candidate["traces"]):
                trace = Path(trace_info["path"])
                remember(trace, historical[str(trace)])
                if trace_info != historical[str(trace)]:
                    raise ValueError("Original candidate trace identity differs")
                result_path = trace.with_suffix(".json")
                remember(result_path, historical[str(result_path)])
                result = json.loads(result_path.read_text())
                with np.load(trace, allow_pickle=False) as archive:
                    fields = sorted(archive.files)
                    if npz_fields is not None and fields != npz_fields:
                        raise ValueError("Saved trace field inventory differs")
                    npz_fields = fields
                    arrays = {key: archive[key][:, indices].copy() for key in
                        ("states", "targets", "force", "slip", "frames", "alive", "residual")}
                    arrays.update(initial=archive["initial"][indices], steps=archive["steps"][indices],
                                  reference_steps=archive["reference_steps"][indices])
                for slot, index in enumerate(indices):
                    clip = FIXED[slot]
                    value = saved_clip(arrays, slot, result["per_clip"][index]["steps"])
                    target, residual = value["targets"].astype(float), arrays["residual"][:len(value["frames"]), slot].astype(float)
                    pre = preceding_states(arrays["initial"][slot], value["states"])
                    requested, clipped, ratio = pd_demand(target, pre[:, 13:35], pre[:, 35:57], kp, kv, limits)
                    nominal = target.copy()
                    nominal[:, :14] -= residual
                    _, nominal_force, _ = pd_demand(nominal, pre[:, 13:35], pre[:, 35:57], kp, kv, limits)
                    change = clipped - nominal_force
                    for frame in sorted({0, len(pre) // 2, len(pre) - 1}):
                        crosschecks.append(crosscheck_actuation(mujoco, model, data, contract,
                            pre[frame], target[frame], clipped[frame]))
                    header = dict(line=line, label=label, repeat=repeat, clip=clip,
                        training_seconds=result["training_seconds"], steps=len(pre))
                    for joint, name in enumerate(names):
                        joint_rows.append(dict(**header, joint=name, kp_Nm_per_rad=float(kp[joint]),
                            kv_Nms_per_rad=float(kv[joint]), force_min_Nm=float(limits[joint, 0]),
                            force_max_Nm=float(limits[joint, 1]),
                            **metrics(*(array[:, joint:joint + 1] for array in (requested, clipped, ratio, change)))))
                    for foot, side in enumerate(("left", "right")):
                        phase = phases[line, label, repeat, clip, side]
                        np.testing.assert_array_equal([int(row["global_reference_frame"]) for row in phase], value["frames"])
                        force = np.array([float(row["native_force_N"]) for row in phase])
                        np.testing.assert_array_equal(force, value["force"][:, foot])
                        raised = np.array([row["intent_swing"] == "True" and float(row["reference_sole_z_m"]) >= .01 for row in phase])
                        for condition, mask in dict(full=np.ones(len(pre), bool), raised_reference_10mm=raised,
                                                    aligned_post_state_loaded=force > 5.).items():
                            joints = slice(foot * 6, foot * 6 + 6)
                            foot_rows.append(dict(**header, side=side, condition=condition, frames=int(mask.sum()),
                                **(metrics(*(array[mask, joints] for array in (requested, clipped, ratio, change))) if mask.any() else {})))
                    series[line, label, repeat, clip] = requested
                traces.extend((identity(trace), identity(result_path)))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_boundary_pd_only=True)), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "joint_boundary_pd.csv", joint_rows)
    write_csv(args.output / "foot_boundary_pd.csv", foot_rows)
    curve_rows = plot(args.output, series)
    if list(tracked.values()) != [identity(Path(path)) for path in tracked]:
        raise ValueError("Analysis sources changed during offline analysis")
    summary = dict(status="SAVED_PD_DEMAND_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()), trace_sources=traces,
        prior_control_summary=identity(prior_path), saved_clips=len(series), saved_states=sum(len(value) for value in series.values()),
        joint_rows=len(joint_rows), foot_condition_rows=len(foot_rows), median_timeseries_rows=curve_rows,
        cpu_actuation_only_checks=len(crosschecks), cpu_algebra_max_abs_difference_Nm=max(crosschecks),
        current_package_versions={name: importlib.metadata.version(name) for name in ("mujoco", "mujoco-warp", "unilab")},
        current_backend_sources=backend_identities, backend_source_historically_frozen=False,
        backend_provenance="Current sources hashed before/after; neither original trial plan froze these backend Python files.",
        native_trace_fields=npz_fields, recorded_actuator_torque=False, recorded_substep_states=False,
        controller=dict(joints=names, kp_Nm_per_rad=kp.tolist(), kv_Nms_per_rad=kv.tolist(),
            force_range_Nm=limits.tolist(), control_dt_s=.02, physics_dt_s=.005, physics_substeps_per_control=4,
            formula="kp*(target-q_pre)-kv*qd_pre; model force clamp; unit hinge gear; no activation/delay"),
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0, goal_complete=False,
        user_accepted_checkpoint=None,
        limits=["This is a frozen-scene/current-backend algebraic reconstruction at20ms control boundaries, not recorded applied torque.",
            "Only first-substep input states are saved. The other three5ms input states and full-interval saturation are unknown.",
            "CPU actuation checks evaluate position/velocity/transmission/actuation without integration, policy, or contact replay.",
            "Backend package sources were not in the original frozen plans; current hashes cannot prove historical runtime immutability.",
            "Post-state native foot-force conditions label the aligned saved endpoint, not instantaneous start-of-step contact.",
            "Residual held-state force change is an instantaneous algebraic comparison, not a counterfactual trajectory or benefit.",
            "No boundary clipping cannot rule out internal-substep saturation or establish enough torque for a new motion.",
            "PD targets need not equal reference joint angles. Gaps alone do not establish soft gains, latency, or a faulty model."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "cpu_actuation_only_checks")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "base", "ppo", "phase", "control", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
