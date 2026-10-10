"""Validate A3 CSV round trips and causal buffering without policy weights."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.fullchain_support import DATA, VENDOR, digest, mark, write_json


def load_sim():
    path = VENDOR / "gear_sonic/scripts/sim2sim_a3_mujoco.py"
    spec = importlib.util.spec_from_file_location("yuanqi_a3_pinned_sim", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def causal_window(source_times, arrival_time, nominal_delay=.18, lookahead_s=.18):
    times = np.asarray(source_times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source times must be finite and strictly increasing")
    if (not np.isfinite(arrival_time) or not np.isfinite(nominal_delay) or
            not np.isfinite(lookahead_s) or lookahead_s < .18 or nominal_delay < lookahead_s):
        raise ValueError("Finite arrival and at least180ms lookahead coverage are required")
    latest = np.searchsorted(times, arrival_time + 1e-10, side="right") - 1
    if latest < 0:
        return None
    base = min(arrival_time - nominal_delay, times[latest] - lookahead_s)
    base = np.floor((base + 1e-10) * 50) / 50
    if base < times[0] - 1e-10:
        return None
    requested = base + np.arange(10) / 50
    upper = np.searchsorted(times, requested - 1e-10, side="left")
    lower = np.maximum(0, upper - 1)
    if np.any(upper > latest):
        raise ValueError("Causal buffer requested a sample not yet available")
    return {"base": float(base), "requested": requested, "lower": lower, "upper": upper,
            "arrival_time": float(arrival_time), "reference_age_s": float(arrival_time - base)}


def validate_references():
    sim = load_sim()
    outcomes = []
    for csv in sorted((DATA / "mocap").glob("*/reference.csv")):
        root, quaternion, dof, rows = sim.load_a3_flat_csv(csv, source_fps=30, frame_stride=1)
        with np.load(csv.with_name("kinematic_trace.npz"), allow_pickle=False) as trace:
            qpos, times = trace["qpos"], trace["time_s"]
            names = trace["joint_names"].tolist()
        model_spec = sim.load_urdf_actuated_joints(sim.DEFAULT_URDF)[0]
        # CSV SDK order and URDF order can differ; compare by names, never array position.
        expected_names = [sim.A3_CSV_JOINT_NAMES[index] for index in sim.A3_POLICY_TO_SDK_IDX]
        from script.a3.mocap_adapter import A3Retargeter
        solver = A3Retargeter()
        indices = [names.index(name) for name in expected_names]
        expected = qpos[:, solver.q_indices][:, indices]
        position_error = float(np.max(np.abs(root - qpos[:, :3])))
        rotation_error = float((Rotation.from_quat(quaternion, scalar_first=True) *
                               Rotation.from_quat(qpos[:, 3:7], scalar_first=True).inv()).magnitude().max())
        joint_error = float(np.max(np.abs(dof - expected)))
        duration_error = abs((rows - 1) / 30 - times[-1])
        if max(position_error, rotation_error, joint_error, duration_error) > 1e-6:
            raise ValueError(f"CSV roundtrip mismatch: {csv}")
        outcomes.append({"source": str(csv), "sha256": digest(csv), "rows": rows,
                         "reference_fps_after_stride": 30, "raw_fps": 30, "frame_stride": 1,
                         "position_max_error_m": position_error, "rotation_max_error_rad": rotation_error,
                         "joint_max_error_rad": joint_error, "duration_error_s": duration_error,
                         "active_joint_count": len(model_spec)})
    for csv in sorted((VENDOR / "a3_data/agibot_a3").glob("*.csv")):
        root, quaternion, dof, rows = sim.load_a3_flat_csv(csv, source_fps=30, frame_stride=4)
        selected_duration = (len(root) - 1) / 30
        raw_duration = (rows - 1) / 120
        if not np.isfinite(root).all() or not np.isfinite(dof).all() or abs(selected_duration - raw_duration) > 1 / 30:
            raise ValueError("Official sampling/duration validation failed")
        outcomes.append({"source": str(csv), "sha256": digest(csv), "rows": rows,
                         "raw_fps": 120, "reference_fps_after_stride": 30, "frame_stride": 4,
                         "raw_duration_s": raw_duration, "selected_duration_s": selected_duration})
    write_json(DATA / "reference_contract.json", outcomes)
    mark("reference_contract", "passed", cases=len(outcomes), evidence=str(DATA / "reference_contract.json"))
    return outcomes


def buffer_replay():
    results = []
    for path in sorted((DATA / "mocap").glob("*/body_reference.npz")):
        with np.load(path, allow_pickle=False) as source:
            times, positions = source["time_s"], source["position_m_world"]
        rows, maximum_reads_ahead = [], 0
        for arrival in np.arange(int(times[-1] * 50) + 1) / 50:
            window = causal_window(times, arrival)
            if window is None:
                continue
            upper, lower, requested = window["upper"], window["lower"], window["requested"]
            ratio = np.divide(requested - times[lower], times[upper] - times[lower],
                              out=np.zeros(10), where=times[upper] != times[lower])
            buffered = positions[lower] * (1 - ratio[:, None, None]) + positions[upper] * ratio[:, None, None]
            direct = np.empty_like(buffered)
            for body in range(positions.shape[1]):
                for axis in range(3):
                    direct[:, body, axis] = np.interp(requested, times, positions[:, body, axis])
            difference = float(np.max(np.abs(direct - buffered)))
            if difference > 1e-10 or times[upper].max() > arrival + 1e-9:
                raise ValueError("Causal and delayed offline reference differ or read future samples")
            rows.append([arrival, window["base"], window["reference_age_s"], float(times[upper].max()), difference])
        rows = np.asarray(rows)
        if not len(rows):
            raise ValueError("No valid buffered windows")
        np.savez_compressed(path.with_name("buffer_trace.npz"),
                            columns=np.asarray(["arrival_s", "target_base_s", "age_s", "latest_sample_s", "max_error_m"]),
                            rows=rows, usage=np.asarray("SIMULATED_ARRIVALS_NOT_LIVE_LATENCY"))
        results.append({"reference": str(path), "windows": len(rows), "nominal_delay_ms": 180,
                        "reference_age_min_ms": float(rows[:, 2].min() * 1000),
                        "reference_age_max_ms": float(rows[:, 2].max() * 1000),
                        "offline_vs_buffered_max_error_m": float(rows[:, 4].max()),
                        "future_sample_reads": maximum_reads_ahead,
                        "scope": "REFERENCE_INPUT_ONLY_NO_POLICY_OR_LIVE_DEVICE"})
    write_json(DATA / "buffer_summary.json", results)
    mark("reference_buffer", "passed", evidence=str(DATA / "buffer_summary.json"),
         scope="causal_reference_replay_only_policy_comparison_waits_for_official_pt")


def validate_motionlibs():
    import joblib
    from gear_sonic.data_process.convert_soma_csv_to_motion_lib import load_a3_flat_csv, convert_sequence
    output = DATA / "motionlib/mocap_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for source in sorted((DATA / "mocap").glob("*/body_reference.npz")):
        csv = source.with_name("reference.csv")
        name = source.parent.name
        path = output / f"{name}.pkl"
        if not path.exists():
            entry = convert_sequence(load_a3_flat_csv(str(csv), robot="a3_29"), fps=30)
            joblib.dump({name: entry}, path, compress=True)
        payload = joblib.load(path)
        if set(payload) != {name}:
            raise ValueError("Unexpected motion identity")
        entry = payload[name]
        with np.load(source, allow_pickle=False) as human:
            frames, duration = len(human["time_s"]), float(human["time_s"][-1])
            if human["usage"].item() != "TEST_ONLY" or human["training_allowed"].item():
                raise ValueError("Diagnostic source usage changed")
        if entry["fps"] != 30 or entry["dof"].shape != (frames, 29):
            raise ValueError("Incorrect frame count, FPS or DOFs")
        if abs((frames - 1) / entry["fps"] - duration) > 1e-6:
            raise ValueError("Conversion changed duration")
        for field in ("root_trans_offset", "root_rot", "pose_aa", "dof"):
            if len(entry[field]) != frames or not np.isfinite(entry[field]).all():
                raise ValueError("Nonfinite or truncated motionlib")
        if not np.allclose(np.linalg.norm(entry["root_rot"], axis=1), 1, atol=1e-6):
            raise ValueError("Nonunit motionlib root rotations")
        outcomes.append({"source": str(csv), "source_sha256": digest(csv), "output": str(path),
                         "sha256": digest(path), "frames": frames, "duration_s": duration,
                         "fps": 30, "active_joints": 29, "usage": "TEST_ONLY", "training_allowed": False})
    for role, count in (("train", 16), ("heldout", 4)):
        paths = sorted((DATA / "motionlib" / role).rglob("*.pkl"))
        if len(paths) != count:
            raise ValueError("Official motionlib split coverage mismatch")
        for path in paths:
            entries = joblib.load(path)
            if set(entries) != {path.stem}:
                raise ValueError("Official motion identity mismatch")
            entry = entries[path.stem]
            if entry["fps"] != 30 or entry["dof"].shape[1] != 29:
                raise ValueError("Official motionlib contract mismatch")
            if any(not np.isfinite(entry[key]).all() for key in ("root_trans_offset", "root_rot", "pose_aa", "dof")):
                raise ValueError("Nonfinite official motionlib")
    write_json(DATA / "motionlib_validation.json", {"mocap": outcomes, "official_train": 16,
               "official_heldout": 4, "scope": "FORMAT_ONLY_NOT_DYNAMIC_OR_POSE_QUALITY_ACCEPTANCE"})
    mark("motionlib", "passed", evidence=str(DATA / "motionlib_validation.json"),
         scope="format_only_mocap_training_not_allowed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "buffer", "motionlib"))
    args = parser.parse_args()
    if args.action == "validate":
        print(json.dumps(validate_references(), indent=2))
    elif args.action == "buffer":
        buffer_replay()
    else:
        validate_motionlibs()


if __name__ == "__main__":
    main()
