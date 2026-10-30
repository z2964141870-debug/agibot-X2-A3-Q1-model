"""Analyze saved native substeps without constructing a policy or simulator."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DT = .005


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty analysis table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runs(mask):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != bool:
        raise ValueError("Expected one boolean sample mask")
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def stats(value, mask):
    selected = np.asarray(value)[mask]
    return dict(samples=int(mask.sum()), mean=float(selected.mean()) if selected.size else None,
                rms=float(np.sqrt(np.mean(selected ** 2))) if selected.size else None,
                abs_p95=float(np.quantile(np.abs(selected), .95)) if selected.size else None,
                abs_max=float(np.max(np.abs(selected))) if selected.size else None)


def pd_force(native, contract):
    ids = np.asarray(contract["native_actuator_ids_in_q1_joint_order"])
    if ids.shape != (22,) or set(ids.tolist()) != set(range(22)):
        raise ValueError("Expected a permutation of22 native actuator IDs")
    coeff = contract["actuator_coefficients"]
    gain = np.asarray(coeff["actuator_gainprm"])[0, ids, 0]
    bias = np.asarray(coeff["actuator_biasprm"])[0, ids]
    limits = np.asarray(coeff["actuator_forcerange"])[0, ids]
    ctrl, length, velocity, force = [np.asarray(native[name], dtype=float)[:, 0, ids] for name in
        ("pre_ctrl", "solve_actuator_length", "solve_actuator_velocity", "solve_actuator_force")]
    request = gain * ctrl + bias[:, 0] + bias[:, 1] * length + bias[:, 2] * velocity
    expected = np.clip(request, limits[:, 0], limits[:, 1])
    error = float(np.max(np.abs(force - expected)))
    if error > 2e-4:
        raise ValueError("Saved native actuator force differs from its recorded direct-PD contract")
    near = (np.isclose(force, limits[:, 0], rtol=0, atol=1e-4)
            | np.isclose(force, limits[:, 1], rtol=0, atol=1e-4))
    return request, force, near, error


def validate_clocks(native, controls):
    if not np.array_equal(native["control_index"], np.repeat(np.arange(controls), 4)):
        raise ValueError("Missing, repeated or reordered control indices")
    if not np.array_equal(native["substep_index"], np.tile(np.arange(4), controls)):
        raise ValueError("Missing, repeated or reordered physical substeps")
    before, after = native["pre_time"][:, 0], native["post_time"][:, 0]
    if (not np.allclose(after-before, DT, rtol=0, atol=2e-6)
            or not np.allclose(before[1:], after[:-1], rtol=0, atol=2e-6)):
        raise ValueError("Native solve/pre/post clocks are not contiguous5ms samples")
    if not np.array_equal(native["pre_qpos"][1:], native["post_qpos"][:-1]):
        raise ValueError("Native position continuity differs between observed substeps")
    return before.astype(float)-float(before[0])


def foot_masks(force, height, side, threshold):
    low = (force[:, side] <= threshold) & (force[:, 1-side] > threshold)
    return dict(all=np.ones(len(force), dtype=bool), loaded=force[:, side] > threshold,
                low_other_loaded=low, low_near5=low & (height[:, side] <= .005),
                low_high10=low & (height[:, side] > .010))


def analyze_case(case_id, line, summary, native, trace):
    spec, contract = summary["case"], summary["contract"]
    controls = summary["controlled_transitions"]
    count = 4 * controls
    if (summary["optimizer_updates"] or count != summary["completed_physical_substeps"]
            or any(summary[key] != count for key in ("attempted_native_substeps", "returned_native_substeps"))
            or controls > spec["maximum_controls"] or int(trace["steps"][0]) != controls):
        raise ValueError("Observation counters exceed bounds or disagree")
    if any(not np.isfinite(value).all() for value in native.values()):
        raise ValueError("Nonfinite native samples")
    seconds = validate_clocks(native, controls)
    request, effort, near, error = pd_force(native, contract)
    force = native["foot_force_world_z_abs_N"][:, 0]
    height = native["solve_collision_sole_z_m"][:, 0]
    slip = native["foot_force_weighted_slip_mps"][:, 0]
    names = contract["q1_joint_names"]
    qi, vi = (np.asarray(contract[key]) for key in
              ("qpos_ids_in_q1_joint_order", "qvel_ids_in_q1_joint_order"))
    position_change = (native["post_qpos"][:, 0, qi]-native["pre_qpos"][:, 0, qi]).astype(float)
    velocity = native["pre_qvel"][:, 0, vi]
    constraint = native["solve_qfrc_constraint"][:, 0, vi]
    root = native["pre_qpos"][:, 0, 3:7].astype(float)
    root /= np.linalg.norm(root, axis=1, keepdims=True)
    tilt = np.arccos(np.clip(1-2*(root[:, 1]**2+root[:, 2]**2), -1, 1))
    label = dict(case_id=case_id, line=line, condition=spec["condition"], clip=spec["clip"], repeat=spec["repeat"])
    feet, joints, segments = [], [], []
    for side, side_name in enumerate(("left", "right")):
        for threshold in (2.5, 5., 10.):
            masks = foot_masks(force, height, side, threshold)
            low = masks["low_other_loaded"]
            high = low & (height[:, side] > .010)
            low_runs, high_runs = runs(low), runs(high)
            feet.append(dict(**label, side=side_name, force_threshold_N=threshold, samples=count,
                sole_z_p95_mm=float(np.quantile(height[:, side], .95)*1000),
                sole_z_max_mm=float(height[:, side].max()*1000),
                high10_samples=int((height[:, side] > .010).sum()),
                low_other_loaded_samples=int(low.sum()), low_near5_samples=int(masks["low_near5"].sum()),
                low_near5_fraction=float(masks["low_near5"].sum()/low.sum()) if low.any() else None,
                low_high10_samples=int(high.sum()), low_high10_runs_ge12=sum(b-a >= 12 for a,b in high_runs),
                longest_low_run_samples=max((b-a for a,b in low_runs), default=0),
                loaded_slip_mean_mps=stats(slip[:, side], masks["loaded"])["mean"],
                root_tilt_rms_rad=stats(tilt, np.ones(count, bool))["rms"]))
            if threshold != 5.:
                continue
            for mask_name, mask in masks.items():
                if mask_name in ("low_other_loaded", "low_near5", "low_high10"):
                    for first, stop in runs(mask):
                        segments.append(dict(**label, side=side_name, mask=mask_name,
                            first_sample=int(first), stop_sample_exclusive=int(stop), samples=int(stop-first),
                            first_s=float(seconds[first]), last_s=float(seconds[stop-1]),
                            endpoint_span_s=float(seconds[stop-1]-seconds[first]),
                            sample_exposure_s=float((stop-first)*DT)))
                for joint, name in enumerate(names):
                    if not name.startswith(side_name+"_") or joint >= 12:
                        continue
                    force_stats = stats(effort[:, joint], mask)
                    joints.append(dict(**label, side=side_name, joint=name, mask=mask_name,
                        samples=int(mask.sum()), effort_abs_p95_Nm=force_stats["abs_p95"],
                        effort_abs_max_Nm=force_stats["abs_max"],
                        near_limit_samples=int(near[mask, joint].sum()),
                        near_limit_fraction=float(near[mask, joint].mean()) if mask.any() else None,
                        near_limit_sample_exposure_s=float(near[mask, joint].sum()*DT),
                        requested_abs_p95_Nm=stats(request[:, joint], mask)["abs_p95"],
                        joint_delta_5ms_rms_rad=stats(position_change[:, joint], mask)["rms"],
                        joint_velocity_rms_radps=stats(velocity[:, joint], mask)["rms"],
                        constraint_generalized_rms_Nm=stats(constraint[:, joint], mask)["rms"]))
    series = np.column_stack((seconds, height*1000, force, slip, effort[:, [4,10,6,9]], tilt))
    metadata = dict(**label, controls=controls, substeps=count, pd_force_max_error_Nm=error,
        policy_batches=summary["policy_batches"], initial_match=summary["initial_match"],
        failed=bool(trace["failed"][0]), gait=summary["gait"], tracking=summary["tracking"])
    return metadata, feet, joints, segments, series


def plot(output, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_prefixes = []
    for line in ("base", "ppo"):
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True, sharex="col")
        conditions = ("initial", "trained301") if line == "base" else ("zero_head", "trained600")
        for col, word in enumerate(("stop", "start")):
            selected = {key: value for key,value in series.items() if key[0] == line and ("_"+word+"_") in key[3]}
            length = min(len(value) for value in selected.values())
            for condition in conditions:
                values = np.median(np.stack([value[:length] for key,value in selected.items() if key[1] == condition]), axis=0)
                for side, color in enumerate(("#147d92", "#bc4d31")):
                    for row, column in enumerate((1+side, 3+side, 7+side)):
                        axes[row,col].plot(values[:,0], values[:,column], color=color,
                            ls=":" if condition == conditions[0] else "-", lw=1,
                            label=("L " if side == 0 else "R ")+condition)
            axes[0,col].axhline(10, color="#555555", ls="--", lw=.6)
            axes[1,col].axhline(5, color="#555555", ls="--", lw=.6)
            axes[2,col].axhline(22, color="#555555", ls="--", lw=.6)
            axes[2,col].axhline(-22, color="#555555", ls="--", lw=.6)
            axes[0,col].set_title("Stop walk" if word == "stop" else "Start walk")
            axes[2,col].set_xlabel("Native solve/pre time (s)")
            for row, unit in enumerate(("Collision sole height (mm)", "Foot world-Z force magnitude (N)", "Ankle pitch actuator force (Nm)")):
                axes[row,col].set_ylabel(unit); axes[row,col].grid(alpha=.2)
        axes[0,0].legend(fontsize=8, ncol=2)
        fig.suptitle(line.upper()+": saved5ms solve samples; framewise median of three repeats", fontsize=12)
        fig.supxlabel("Fresh one-world observations; sampled forces/heights are simultaneous. Curves do not establish causality.", fontsize=9)
        for ext in ("png", "pdf"):
            fig.savefig(output/(line+"_native_substeps."+ext), dpi=150)
        plt.close(fig)
        plot_prefixes.append(line+"_native_substeps")
    return plot_prefixes


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs, metadata, feet, joints, segments, series, sample_rows = {}, [], [], [], [], {}, []
    for line in ("base", "ppo"):
        root = getattr(args,line)
        result_path = root/"results.json"
        inputs[str(result_path)] = identity(result_path)
        results = json.loads(result_path.read_text())
        if results["status"] != "OBSERVATION_COMPLETED" or results["optimizer_updates"] or len(results["cases"]) != 12:
            raise ValueError("Expected twelve completed zero-optimizer observation cases")
        counted_controls = counted_steps = 0
        for item in results["cases"]:
            archives = {}
            for record in item["artifacts"]:
                actual = identity(record["path"])
                if actual != record:
                    raise ValueError("Observation archive identity mismatch")
                inputs[record["path"]] = actual
                with np.load(record["path"], allow_pickle=False) as source:
                    if Path(record["path"]).name != "initial_inputs.npz":
                        archives[Path(record["path"]).stem] = {key:source[key] for key in source.files}
            m,f,j,s,values = analyze_case(len(metadata),line,item,archives["native_substeps"],archives["control_trajectory"])
            metadata.append(m); feet.extend(f); joints.extend(j); segments.extend(s)
            series[line,m["condition"],m["repeat"],m["clip"]] = values
            counted_controls += m["controls"]; counted_steps += m["substeps"]
            for value in values:
                sample_rows.append(dict(case_id=m["case_id"], **{key:format(float(number),".8g") for key,number in zip(
                    ("solve_time_s","sole_left_mm","sole_right_mm","force_left_N","force_right_N",
                     "slip_left_mps","slip_right_mps","ankle_left_Nm","ankle_right_Nm","hip_right_Nm","knee_right_Nm","root_tilt_rad"),value)}))
        if (counted_controls != results["controlled_transitions"] or counted_controls > 3180
                or counted_steps != results["completed_physical_substeps"] or counted_steps > 12720):
            raise ValueError("Line totals disagree with bounded case totals")
    for value in inputs.values():
        if identity(value["path"]) != value:
            raise ValueError("Observation archive changed while being read")
    args.output.mkdir(parents=True)
    for name,rows in (("foot_conditions.csv",feet),("joint_conditions.csv",joints),
                      ("sample_runs.csv",segments),("native_solve_series.csv",sample_rows)):
        write_csv(args.output/name, rows)
    plot(args.output,series)
    summary = dict(status="SAVED_NATIVE_SUBSTEP_ANALYSIS_READY", source_base_commit=args.source_base_commit,
        cases=metadata, sources=list(inputs.values()), foot_rows=len(feet), joint_rows=len(joints),
        run_rows=len(segments), native_sample_rows=len(sample_rows),
        native_controls_analyzed=sum(m["controls"] for m in metadata),
        native_substeps_analyzed=sum(m["substeps"] for m in metadata),
        new_physical_or_policy_calls=0, optimizer_updates=0,
        conditional_masks="Same-time solve force/height; no reference-phase interpolation",
        sample_clock="5ms samples; run endpoint span is (n-1)*5ms, exposure n*5ms; neither proves continuous contact",
        physical_trajectory_equivalence_verified=False, historical_replay=False,
        original_training_seconds=901.178391956928, goal_status="active", user_accepted_checkpoint=None)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:summary[key] for key in ("status","native_controls_analyzed","native_substeps_analyzed","foot_rows","joint_rows","run_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
