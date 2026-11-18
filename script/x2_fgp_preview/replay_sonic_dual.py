"""Exercise a saved planner–SMPL–planner sequence entirely offline."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess

import numpy as np

from replay_sonic_stand import (
    ACTION_HISTORY_END, ACTION_HISTORY_START, feedback, observations, planner_windows,
)


def smpl_inputs(probe, receiver, base, times, heads):
    capture = np.loadtxt(receiver, delimiter=",")
    if capture.ndim != 2 or capture.shape[1] != 77 or not np.isfinite(capture).all():
        raise ValueError("expected recorded SMPL receive clock, 72 points, WXYZ")
    capture = capture[np.argsort(capture[:, 0])]
    if (np.diff(capture[:, 0]) <= 0).any():
        raise ValueError("invalid SMPL receive clock")
    human_time = capture[:, 0] - capture[0, 0]
    if human_time[-1] < 8.25 + times[-1]:
        raise ValueError("SMPL capture does not cover the test span")
    # Preserve the observed cadence. Use the final few seconds of the saved
    # garment run, so the test includes the actual arm movement.
    sample_time = 8.25 + times
    slots = sample_time[:, None] - .18 + np.arange(10) * .02
    indices = np.searchsorted(human_time, slots, side="right") - 1
    if (indices < 0).any():
        raise ValueError("missing history before the first SMPL window")
    measured = base[:, 7:38][:, [19, 20, 21, 26, 27, 28]].copy()
    measured[np.ix_(heads, [1, 2, 4, 5])] = 0
    anchor_index = np.full(len(times), 0)
    if heads.any():
        anchor_start = int(np.flatnonzero(heads)[0])
        anchor_index[anchor_start:] = anchor_start
    frames = capture[indices, 1:]
    payload = np.concatenate((base[:, :4], base[anchor_index, :4],
                              frames[anchor_index, -1, -4:], measured,
                              frames.reshape(len(times), 760)), axis=1)
    result = subprocess.run([str(probe.resolve()), "smpl"], input=payload.astype("<f8").tobytes(),
                            capture_output=True, timeout=20, check=True)
    features = np.frombuffer(result.stdout, dtype="<f4").reshape(len(times), 840)
    if not np.isfinite(features).all():
        raise ValueError("invalid native SMPL features")
    return features


def rollout(session, pose, smpl, times, heads):
    history = deque([np.zeros(31, dtype=np.float32)] * 10, maxlen=10)
    rows = []
    start = None
    previous_head = False
    for i, head in enumerate(heads):
        if head != previous_head:
            origin = history[-1].copy()
            start = times[i]
        g1, prop = pose[i, :680], pose[i, 680:].copy()
        prop[ACTION_HISTORY_START - 680:ACTION_HISTORY_END - 680] = np.concatenate(history)
        obs = np.concatenate((g1, smpl[i], prop, np.asarray([head], np.float32)))
        model = session.run(["action"], {"obs": obs[None, :]})[0][0]
        if model.shape != (31,) or not np.isfinite(model).all():
            raise ValueError("invalid dual-head action")
        blended = model
        if start is not None:
            fraction = min((times[i] - start) / .2, 1.)
            weight = .5 - .5 * np.cos(np.pi * fraction)
            blended = ((1 - weight) * origin + weight * model).astype(np.float32)
        clipped = np.clip(blended, -20, 20)
        history.append(clipped)
        rows.append((model.copy(), clipped.copy()))
        previous_head = head
    return np.asarray(rows)


def report(actions, first, last):
    raw, crossfaded = np.degrees(actions[:, :, 8] * .48510106).T
    jumps = np.abs(np.diff(crossfaded, prepend=crossfaded[0]))
    return dict(
        raw_waist_peak_deg=float(np.max(np.abs(raw))),
        crossfaded_waist_peak_deg=float(np.max(np.abs(crossfaded))),
        crossfaded_largest_step_deg=float(max(jumps)),
        crossfaded_largest_step_tick=int(np.argmax(jumps)),
        transition_steps_deg=[float(jumps[i]) for i in (first, last)],
        latent_action_clip_ticks=int((np.max(np.abs(actions[:, 0]), axis=1) > 20).sum()),
        segments={label: float(np.max(np.abs(crossfaded[part]))) for label, part in (
            ("planner_entry", slice(0, first)), ("tracking", slice(first, last)),
            ("planner_return", slice(last, None)))},
    )


def main():
    import onnxruntime as ort
    from gear_sonic.utils.pose_pipeline.wire import yaw_from_quat_wxyz

    parser = argparse.ArgumentParser(__doc__)
    for name in ("model", "probe", "feedback", "planner", "receiver"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    base, _, times = feedback(args.feedback, 180)
    yaws = [yaw_from_quat_wxyz(row[:4]) for row in base]
    pose = observations(args.probe, base, planner_windows(args.planner, times, yaws, match_heading=True))
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    if (session.get_inputs()[0].name != "obs" or session.get_inputs()[0].shape[-1] != 2511
            or meta.get("graph_kind") != "native_dual_head" or meta.get("ori_mode") != "full"):
        raise ValueError("unexpected native dual-head observation contract")
    start, end = 34, 145
    switch_heads = (np.arange(len(times)) >= start) & (np.arange(len(times)) < end)
    cases = {}
    for name, heads in (("planner_only", np.zeros(len(times), dtype=bool)), ("switching", switch_heads)):
        features = smpl_inputs(args.probe, args.receiver, base, times, heads)
        cases[name] = report(rollout(session, pose, features, times, heads), start, end)
    print(json.dumps(dict(ticks=len(times), model_calls=2 * len(times),
                          source_window_s=[8.25, 8.25 + times[-1]], scheduled_switch_ticks=[start, end],
                          reconstruction="one recorded feedback trajectory kept exogenous; fixed switch schedule",
                          cases=cases, hardware_commands=0), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
