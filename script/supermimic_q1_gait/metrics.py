"""Physical gait diagnostics and a fixed, root-tolerant checkpoint ranking."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def enrich(trace, mdp):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

    ids = [mdp.kin.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")]
    shape = trace["states"].shape[:2]
    states = torch.from_numpy(trace["states"].reshape(-1, 57)).to(mdp.device if hasattr(mdp, "device") else mdp.motion.q.device)
    feet = [mdp.kin.forward(*unflatten_state(block), all_bodies=True)[0][:, ids].cpu().numpy()
            for block in states.split(8192)]
    trace["feet"] = np.concatenate(feet).reshape(*shape, 2, 3)
    ref = trace["states"].copy()
    ref[..., :3] = trace["reference_p"]
    ref[..., 3:7] = trace["reference_quat"]
    ref[..., 13:35] = trace["reference_q"]
    refs = torch.from_numpy(ref.reshape(-1, 57)).to(states.device)
    feet = [mdp.kin.forward(*unflatten_state(block), all_bodies=True)[0][:, ids].cpu().numpy()
            for block in refs.split(8192)]
    trace["reference_feet"] = np.concatenate(feet).reshape(*shape, 2, 3)
    return trace


def gait_clip(trace, index, count=None):
    count = int(trace["steps"][index]) if count is None else int(count)
    if count < 1:
        raise ValueError("Empty common physical prefix")
    state, feet, ref = (trace[key][:count, index] for key in ("states", "feet", "reference_feet"))
    root, reference_root = state[:, :3], trace["reference_p"][:count, index]
    origin = trace["initial"][index, :2]
    direction = reference_root[-1, :2] - reference_root[0, :2]
    norm = np.linalg.norm(direction)
    unit = direction / max(norm, 1e-8)
    from evaluation import local_wrists
    local = local_wrists(feet, root, state[:, 3:7])
    reference_local = local_wrists(ref, reference_root, trace["reference_quat"][:count, index])
    loaded = trace["force"][:count, index] > 5.
    w, x, y, z = state[:, 3:7].T
    tilt = np.arccos(np.clip(1 - 2 * (x*x + y*y), -1, 1))
    delta = np.diff(trace["targets"][:count, index], axis=0)
    height = feet[..., 2] - np.percentile(ref[..., 2], 5, axis=0)
    reference_height = ref[..., 2] - np.percentile(ref[..., 2], 5, axis=0)
    swing = reference_height > .02
    errors = (local - reference_local)[swing]
    single = np.where(loaded[:, 0] & ~loaded[:, 1], 1,
                      np.where(loaded[:, 1] & ~loaded[:, 0], 2, 0))
    sustained, last, run = [], 0, 0
    for value in single:
        run = run + 1 if value == last else 1
        last = value
        if value and run == 3 and (not sustained or sustained[-1] != value):
            sustained.append(int(value))
    return {"steps": int(trace["steps"][index]), "common_prefix_steps": count,
            "failed": bool(trace["failed"][index]),
            "completion": float(trace["steps"][index] / trace["reference_steps"][index]),
            "forward_displacement_m": float(np.dot(root[-1, :2] - origin, unit)),
            "reference_displacement_m": float(norm),
            "root_displacement_m": float(np.linalg.norm(root[-1, :2] - origin)),
            "foot_lift_p95_m": np.percentile(height, 95, axis=0).tolist(),
            "reference_foot_lift_p95_m": np.percentile(reference_height, 95, axis=0).tolist(),
            "foot_lift_tracking_rmse_m": float(np.sqrt(np.mean((height-reference_height)**2))),
            "swing_local_foot_rmse_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "loaded_slip_mps": float(trace["slip"][:count, index][loaded].mean()) if loaded.any() else None,
            "torso_tilt_rms_rad": float(np.sqrt(np.mean(tilt**2))),
            "knee_reference_rmse_rad": float(np.sqrt(np.mean((state[:, [16, 22]] - trace["reference_q"][:count, index, [3, 9]])**2))),
            "target_delta_rms_rad": float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
            "sustained_single_support_alternations": max(len(sustained)-1, 0),
            "support_note": "Distinct single-support sides each sustained >=60ms; diagnostic, not a validated step counter."}


def gait_summary(trace, rows):
    from evaluation import FIXED_DEMOS
    return {row["name"]: gait_clip(trace, i) for i, row in enumerate(rows) if row["name"] in FIXED_DEMOS}


def checkpoint_rank(summaries):
    """Lower is preferable; missing/nonfinite evidence never improves a rank."""
    from evaluation import FIXED_DEMOS
    failures, nonforward, scores = 0, 0, []
    for summary in summaries:
        failures += int(summary["summary"]["validation"]["failures"])
        for name in FIXED_DEMOS:
            clip = summary["gait"][name]
            failures += int(clip["failed"] or clip["completion"] < 1)
        for name in FIXED_DEMOS[:2]:
            clip = summary["gait"][name]
            nonforward += int(clip["forward_displacement_m"] <= .05)
            values = (clip["loaded_slip_mps"], clip["foot_lift_tracking_rmse_m"],
                      clip["torso_tilt_rms_rad"], clip["target_delta_rms_rad"])
            if any(v is None or not np.isfinite(v) for v in values):
                return [999999, 999999, 1e30]
            scores.append(values[0] / .2 + values[1] / .05 + values[2] / .25 + values[3] / .1)
    return [failures, nonforward, float(np.mean(scores))]
