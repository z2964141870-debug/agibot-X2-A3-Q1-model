"""Training-only next-state targets from the actual executed transition."""

import numpy as np

from student_causal_fit import causal_records


def target_indices(present):
    present = np.asarray(present)
    if present.shape != (29,) or present.dtype != np.bool_ or present.sum() != 22:
        raise ValueError("Require the actual 22-of-29 Q1 morphology")
    joints = np.flatnonzero(present)
    return np.concatenate((joints, joints+29, np.arange(58, 64)))


def transition_records(data, traces, terminal_frames, observation, present, *, aggregate):
    """Future state is supervision only; history and f0 are retained verbatim."""
    records = causal_records(data)
    indices = target_indices(present)
    count = len(records["clip_id"])
    if (data["frame"].shape != (count,) or data["actual_state"].shape != (count, 79)
            or not np.issubdtype(data["frame"].dtype, np.integer)):
        raise ValueError("Missing preserved physical sample identities")
    current, following, behavior, keep = [], [], [], []
    dropped = []
    maximum_difference = 0.
    for clip in np.unique(records["clip_id"]):
        clip = int(clip)
        trace = traces[clip]
        states, actions = trace["actual"], trace["canonical_action"]
        if (states.shape != (len(actions)+1, 79) or actions.shape[1:] != (29,)
                or not np.isfinite(states).all() or not np.isfinite(actions).all()):
            raise ValueError("Invalid saved transition sequence")
        obs = np.asarray(observation(states), dtype=np.float32)
        if obs.shape != (len(states), 93) or not np.isfinite(obs).all():
            raise ValueError("Require frozen-normalized native observations")
        for row in np.flatnonzero(records["clip_id"] == clip):
            frame = int(data["frame"][row])
            if not 0 <= frame < len(actions):
                raise ValueError("Next-state target crosses a source boundary")
            if not np.array_equal(data["actual_state"][row], states[frame]):
                raise ValueError("Current physical state differs from the saved trace")
            label = trace["teacher_action"][frame] if aggregate else actions[frame]
            if not np.allclose(records["teacher_action"][row], label, atol=2e-5, rtol=0.):
                raise ValueError("Teacher query changed its actual physical state")
            if aggregate and not np.array_equal(data["behavior_action"][row], actions[frame]):
                raise ValueError("The next-state label must condition on the executed mixture")
            newest = records["history"][row].reshape(10, 93)[-1]
            difference = float(np.max(np.abs(newest-obs[frame])))
            maximum_difference = max(maximum_difference, difference)
            if not np.allclose(newest, obs[frame], atol=3e-5, rtol=3e-5):
                raise ValueError("Saved newest observation differs from the actual state")
            terminal = terminal_frames[clip]
            if terminal is not None and frame+1 >= terminal:
                dropped.append(dict(clip_id=clip, frame=frame))
                continue
            keep.append(row)
            current.append(obs[frame, indices])
            following.append(obs[frame+1, indices])
            behavior.append(actions[frame, present])
    order = np.argsort(keep)
    rows = np.asarray(keep, dtype=np.int64)[order]
    if not len(rows) or set(records["clip_id"][rows]) != set(records["clip_id"]):
        raise ValueError("Transition masking removed an entire owned source")
    result = {key:value[rows].copy() for key, value in records.items()}
    result.update(current_state=np.asarray(current, dtype=np.float32)[order],
        next_state=np.asarray(following, dtype=np.float32)[order],
        executed_action=np.asarray(behavior, dtype=np.float32)[order],
        frame=data["frame"][rows].copy())
    return result, dict(records=len(rows), terminal_rows_dropped=dropped,
        target_indices=indices.tolist(), target_features=50, includes_gravity=True,
        previous_action_target=False, behavior_is_recorded_executed_action=True,
        maximum_newest_history_difference=maximum_difference,
        next_state_is_training_label_only=True)
