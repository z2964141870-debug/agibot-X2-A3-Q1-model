"""Compare saved X2 stand references with identical logged feedback and action history."""
from __future__ import annotations

import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ACTION_HISTORY_START = 680 + 10 * (3 + 31 + 31)
ACTION_HISTORY_END = ACTION_HISTORY_START + 10 * 31
MJ_TO_IL = (0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12, 17, 21, 23, 25,
            27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16)


def rows(path):
    result = {}
    with path.open() as source:
        reader = csv.reader(line for line in source if "\0" not in line)
        fields = next(reader)
        for row in reader:
            if len(row) == len(fields):
                if row[0] in result:
                    raise ValueError(f"duplicate timestamp in {path.name}")
                result[row[0]] = [float(value) for value in row[1:]]
    return result


def feedback(root, count):
    names = ("imu", "joint_pos", "joint_vel", "action_il")
    data = [rows(root / f"{name}.csv") for name in names]
    stamps = sorted(set.intersection(*(set(table) for table in data)), key=float)[:count]
    if len(stamps) != count:
        raise ValueError("not enough common feedback and action ticks")
    imu, q, dq, recorded_action = (np.asarray([x[t] for t in stamps]) for x in data)
    previous_action = np.concatenate((np.zeros((1, 31)), recorded_action[:-1]))
    samples = np.concatenate((imu, q, dq, previous_action), axis=1)
    clocks = np.asarray(stamps, dtype=np.float64)
    if not np.isfinite(samples).all() or (np.diff(clocks) <= 0).any():
        raise ValueError("non-finite data or non-increasing feedback clock")
    return samples, recorded_action, clocks - clocks[0]


def idle_windows(path, times, yaws, phase):
    from gear_sonic.utils.pose_pipeline.fallback import IdleStandReplay
    from gear_sonic.utils.pose_pipeline.wire import load_x2m2, rebase_quats_xyzw_by_yaw

    q, quat, _ = load_x2m2(path)
    replay = IdleStandReplay(q, quat)
    out = []
    for i, yaw in enumerate(yaws):
        latest, orientation = replay.current(phase + i)
        previous, _ = replay.current(phase + i - 1)
        vel = (latest - previous) / (times[i] - times[i - 1]) if i else np.zeros(31)
        future_q, future_quat, future_vel = replay.future_window(phase + i)
        orientations = rebase_quats_xyzw_by_yaw(np.vstack((orientation, future_quat)), yaw)
        out.append(np.concatenate((np.vstack((latest, future_q)), np.vstack((vel, future_vel)),
                                   orientations), axis=1))
    return np.asarray(out)


def planner_windows(path, times, yaws, match_heading):
    from gear_sonic.utils.pose_pipeline.wire import rebase_quats_xyzw_by_yaw, yaw_from_quat_wxyz

    with path.open("rb") as source:
        tape = np.fromfile(source, dtype="<f4", count=1200 * 40).reshape(-1, 40)
    invalid = np.flatnonzero(np.diff(tape[:, 0]) <= 0)
    if invalid.size:
        tape = tape[:invalid[0] + 1]
    clock = tape[:, 0].astype(np.float64) - tape[0, 0]
    if clock[-1] < times[-1] + .9:
        raise ValueError("planner tape does not cover the complete reference window")
    q, angles = tape[:, 9:40], tape[:, 5:9]
    velocity = np.vstack((np.zeros(31), np.diff(q, axis=0) / np.diff(clock)[:, None]))
    starting_yaw = yaw_from_quat_wxyz(angles[0, [3, 0, 1, 2]])
    out = []
    for t, yaw in zip(times, yaws):
        indices = np.maximum(0, np.searchsorted(clock, t + np.arange(10) * .1, side="right") - 1)
        ref_quat = angles[indices]
        if match_heading:
            ref_quat = rebase_quats_xyzw_by_yaw(ref_quat, yaw - starting_yaw)
        out.append(np.concatenate((q[indices], velocity[indices], ref_quat), axis=1))
    return np.asarray(out)


def observations(binary, base, windows):
    vectors = np.concatenate((base, windows.reshape(len(base), 660)), axis=1)
    result = subprocess.run([str(binary.resolve())], input=vectors.astype("<f8").tobytes(),
                            capture_output=True, check=True, timeout=20)
    output = np.frombuffer(result.stdout, dtype="<f4").reshape(len(base), 1670)
    if not np.isfinite(output).all():
        raise ValueError("non-finite native observations")
    oldest_to_current = np.maximum(0, np.arange(len(base))[:, None] + np.arange(-9, 1))
    expected_history = base[oldest_to_current, 69:].reshape(len(base), 310).astype(np.float32)
    np.testing.assert_array_equal(output[:, ACTION_HISTORY_START:ACTION_HISTORY_END], expected_history)
    return output


def predict(session, inputs, *, recompute_history=False):
    if not recompute_history:
        return np.concatenate([session.run(None, {"obs": row[None, :]})[0] for row in inputs])
    # Native builder writes oldest-first history grouped by term. Replace only
    # the previous-action term, and leave measured feedback and reference fixed.
    history = deque([np.zeros(31, dtype=np.float32)] * 10, maxlen=10)
    outputs = []
    for row in inputs:
        obs = row.copy()
        obs[ACTION_HISTORY_START:ACTION_HISTORY_END] = np.concatenate(history)
        action = session.run(None, {"obs": obs[None, :]})[0][0]
        outputs.append(action)
        history.append(np.clip(action, -20, 20))
    return np.asarray(outputs)


def examine_history(session, probe, base, refs):
    obs = observations(probe, base, refs)
    groups = {"all": MJ_TO_IL, "legs": MJ_TO_IL[:12], "waist": MJ_TO_IL[12:15],
              "arms_head": MJ_TO_IL[15:], "waist_roll_only": (MJ_TO_IL[14],)}
    out = {}
    baseline = predict(session, obs)
    for name, indices in groups.items():
        edited = obs.copy()
        history = edited[:, ACTION_HISTORY_START:ACTION_HISTORY_END].reshape(len(obs), 10, 31)
        history[:, :, indices] = 0
        altered = predict(session, edited)
        waist = np.abs(np.degrees(altered[:, 8] * .48510106))
        out[name] = dict(waist_abs_raw_target_p95_deg=float(np.percentile(waist, 95)),
                         waist_action_median_absolute_change=float(np.median(np.abs(altered[:, 8] - baseline[:, 8]))))
    return dict(calls=len(base) * (len(groups) + 1),
                baseline_waist_p95_deg=float(np.percentile(np.abs(np.degrees(baseline[:, 8] * .48510106)), 95)),
                action_history_group_set_to_zero=out, hardware_commands=0)


def main():
    from gear_sonic.utils.pose_pipeline.wire import yaw_from_quat_wxyz
    import onnxruntime as ort

    parser = argparse.ArgumentParser(__doc__)
    for name in ("model", "probe", "feedback", "idle", "planner"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--factorial", action="store_true",
                        help="Separate reference, measured feedback and previous-action effects")
    parser.add_argument("--diagnose-history", action="store_true",
                        help="Screen which past-action groups influence waist output for idle")
    args = parser.parse_args()
    if not 20 <= args.frames <= 180:
        parser.error("--frames must be 20–180 pre-engage ticks")
    base, logged_action, times = feedback(args.feedback, args.frames)
    yaws = [yaw_from_quat_wxyz(row[:4]) for row in base]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    (input_info,) = session.get_inputs()
    if input_info.name != "obs" or input_info.shape not in ([1, 1670], ["batch", 1670]):
        raise ValueError("unexpected v16 pose policy input")
    if args.diagnose_history:
        print(json.dumps(examine_history(session, args.probe, base, idle_windows(args.idle, times, yaws, 0)),
                         indent=2, allow_nan=False))
        return
    groups = [(f"idle_phase_{phase}", idle_windows(args.idle, times, yaws, phase))
              for phase in (0, 25, 50)]
    groups += [(name, planner_windows(args.planner, times, yaws, match_heading=matched))
               for name, matched in (("planner_as_logged", False), ("planner_heading_matched", True))]
    logged_roll = logged_action[:, 8]
    summary = {}
    feedback_obs = {}
    if args.factorial:
        groups = [pair for pair in groups if pair[0] in ("idle_phase_0", "planner_heading_matched")]
    cases = []
    for reference, refs in groups:
        for feedback_mode in (("recorded", "first_frame_held") if args.factorial else ("recorded",)):
            state = base.copy()
            if feedback_mode == "first_frame_held":
                state[:, :69] = base[0, :69]
            for history in (("logged", "zero", "self_predicted") if args.factorial else ("logged",)):
                complete = state.copy()
                if history == "zero":
                    complete[:, 69:] = 0
                cases.append((reference, refs, feedback_mode, history, complete))
    for reference, refs, feedback_mode, history, state in cases:
        name = f"{reference}/{feedback_mode}/{history}" if args.factorial else reference
        obs = observations(args.probe, state, refs)
        key = (feedback_mode, history)
        if key in feedback_obs:
            np.testing.assert_array_equal(feedback_obs[key], obs[:, 680:])
        else:
            feedback_obs[key] = obs[:, 680:].copy()
        actions = predict(session, obs, recompute_history=history == "self_predicted")
        if actions.shape != (len(base), 31) or not np.isfinite(actions).all():
            raise ValueError("unexpected v16 pose action")
        roll = actions[:, 8]
        summary[name] = dict(roll_action_min=float(roll.min()), roll_action_max=float(roll.max()),
                             waist_median_abs_action_error_vs_log=float(np.median(np.abs(roll-logged_roll))),
                             waist_abs_raw_target_p95_deg=float(np.percentile(np.abs(np.degrees(roll * .48510106)), 95)),
                             waist_abs_raw_target_max_deg=float(np.max(np.abs(np.degrees(roll * .48510106)))),
                             action_clip_would_activate_ticks=int((np.max(np.abs(actions), axis=1) > 20).sum()))
    print(json.dumps(dict(frames=len(base), calls=len(base) * len(cases),
        shared_native_proprioception_bit_equal_within_cases=True, reference_windows="offline reconstruction",
        history=("logged, zero, or candidate-fed action history" if args.factorial
                 else "logged previous actions held identical across sources"),
        anchor_orientation_override=os.environ.get("X2_ANCHOR_ORI_MODE"), groups=summary,
        hardware_commands=0), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
