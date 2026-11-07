"""Deterministic saved-transition selection and physical-only calibration metrics."""

import numpy as np


def selected_transitions(data, traces, terminal, *, aggregate, quota=64):
    frames, clips, current = (np.asarray(data[k]) for k in ("frame", "clip_id", "actual_state"))
    count = len(frames)
    if (not 1 <= quota <= 64 or frames.shape != clips.shape or frames.shape != (count,)
            or current.shape != (count, 79) or not np.issubdtype(frames.dtype, np.integer)
            or not np.issubdtype(clips.dtype, np.integer) or not np.isfinite(current).all()):
        raise ValueError("Require finite saved state79 with integer clip/frame identities")
    groups, excluded = {}, []
    seen = set()
    for index, (clip, frame) in enumerate(zip(clips, frames)):
        clip, frame = int(clip), int(frame)
        if (clip, frame) in seen:
            raise ValueError("Duplicate saved transition identity")
        seen.add((clip, frame))
        trace = traces[clip]
        states, actions = trace["actual"], trace["canonical_action"]
        if (states.shape != (len(actions)+1, 79) or actions.shape[1:] != (29,)
                or not 0 <= frame < len(actions) or not np.isfinite(actions).all()
                or not np.isfinite(states).all() or not np.array_equal(current[index], states[frame])):
            raise ValueError("Saved transition differs from its physical trace")
        label = trace["teacher_action"][frame] if aggregate else actions[frame]
        if not np.allclose(data["teacher_action"][index], label, atol=2e-5, rtol=0.):
            raise ValueError("Saved teacher label differs from its actual state")
        if aggregate and not np.array_equal(data["behavior_action"][index], actions[frame]):
            raise ValueError("Use recorded executed behavior, not the queried teacher")
        if terminal[clip] is not None and frame+1 >= terminal[clip]:
            excluded.append(dict(clip_id=clip, frame=frame))
            continue
        groups.setdefault(clip, []).append((frame, index))
    if set(groups) != set(map(int, np.unique(clips))) or len(groups) > quota:
        raise ValueError("Every owned source needs at least one nonterminal transition")
    selected = []
    for rank, clip in enumerate(sorted(groups)):
        rows = sorted(groups[clip])
        take = min(len(rows), quota//len(groups)+(rank < quota % len(groups)))
        for position in np.linspace(0, len(rows)-1, take, dtype=np.int64):
            frame, index = rows[int(position)]
            trace = traces[clip]
            selected.append(dict(clip_id=clip, frame=frame, dataset_row=index,
                current=trace["actual"][frame].copy(), following=trace["actual"][frame+1].copy(),
                executed=trace["canonical_action"][frame].copy()))
    return selected, dict(candidate_rows=count, selected_rows=len(selected), sources=len(groups),
        terminal_rows_excluded=excluded, selection="sorted source quotas, evenly spaced valid rows",
        test_sources_read=False, recorded_behavior=True)


def nominal_context(robot, spec, settings):
    """Reconstruct the nominal common plant; source hashes must be checked first."""
    import re

    if settings.get("action_scale") != .5 or settings.get("modeled_action_delay", False):
        raise ValueError("Only the verified nominal no-delay common plant is registered")
    ids = [i for i in range(robot.model.ngeom)
           if re.fullmatch(spec.foot_geom_regex, robot.model.geom(i).name or "")]
    if not ids:
        raise ValueError("The frozen native foot regex matches no physical geoms")
    friction = np.asarray(robot.model.geom_friction)[ids, 0]
    if not np.isfinite(friction).all() or (friction < 0).any() or not np.all(friction == friction[0]):
        raise ValueError("A single native friction context cannot describe these geoms")
    context = np.zeros(62, dtype=np.float32)
    context[0] = friction[0]
    return context, dict(kind="SOURCE_AND_XML_RECONSTRUCTED_NOMINAL_NOT_RECORDED_TENSOR",
        friction=float(context[0]), foot_geoms=[robot.model.geom(i).name for i in ids],
        encoder_bias="zero from preserved randomize=False collector and nominal reset",
        torso_com_offset="zero from unchanged compiled XML and nominal reset",
        action_delay_seconds="zero; explicit modeled_action_delay=False required",
        no_simulator_created=True, domain="common Primebot CPU plant, not original training dynamics")


def physical_errors(prediction, target, present):
    prediction, target, present = np.asarray(prediction), np.asarray(target), np.asarray(present)
    if (prediction.shape != target.shape or prediction.shape[-1:] != (269,)
            or present.shape != (29,) or present.dtype != np.bool_ or present.sum() != 22
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("Require finite native physical states and exact Q1 joint mask")
    delta = prediction-target
    rms = lambda x: np.sqrt(np.mean(np.square(x), axis=-1))
    quat = prediction[..., 100:156].reshape(-1, 14, 4)
    true = target[..., 100:156].reshape(-1, 14, 4)
    qnorm, tnorm = np.linalg.norm(quat, axis=-1), np.linalg.norm(true, axis=-1)
    if np.any(qnorm < 1e-6) or np.any(tnorm < 1e-6):
        raise ValueError("Degenerate physical orientation")
    dot = np.abs(np.sum(quat*true, axis=-1)/(qnorm*tnorm))
    angles = 2*np.arccos(np.clip(dot, 0., 1.))
    return dict(joint_position_rad=rms(delta[..., :29][..., present]),
        joint_velocity_rad_s=rms(delta[..., 29:58][..., present]),
        body_position_m=rms(delta[..., 58:100])*np.sqrt(3),
        body_orientation_rad=rms(angles), body_linear_velocity_m_s=rms(delta[..., 156:198])*np.sqrt(3),
        body_angular_velocity_rad_s=rms(delta[..., 198:240])*np.sqrt(3),
        command_memory_rms=rms(delta[..., 240:269][..., present]))


def summarize(rows, *, improvement_ratio=.95, maximum_source_ratio=1.25):
    metrics = tuple(rows[0]["wm"])
    sources = []
    for key in sorted({(r["origin"], r["split"], r["clip_id"]) for r in rows}):
        group = [r for r in rows if (r["origin"], r["split"], r["clip_id"]) == key]
        item = dict(origin=key[0], split=key[1], clip_id=key[2], samples=len(group))
        for method in ("wm", "identity"):
            item[method] = {m:float(np.mean([r[method][m] for r in group])) for m in metrics}
        sources.append(item)
    groups, passed = [], True
    physical = ("joint_position_rad", "body_position_m")
    for key in sorted({(r["origin"], r["split"]) for r in rows}):
        source = [r for r in sources if (r["origin"], r["split"]) == key]
        item = dict(origin=key[0], split=key[1], sources=len(source), source_weighting="equal")
        for method in ("wm", "identity"):
            item[method] = {m:float(np.mean([r[method][m] for r in source])) for m in metrics}
        item["ratios"] = {m:(item["wm"][m]/item["identity"][m]
            if item["identity"][m] > 1e-12 else None) for m in metrics}
        if key[1] == "validation":
            item["calibration_screen_passed"] = all(item["ratios"][m] is not None
                and item["ratios"][m] <= improvement_ratio for m in physical) and all(
                    r["wm"][m] <= maximum_source_ratio*max(r["identity"][m], 1e-12)
                    for r in source for m in physical)
            passed = passed and item["calibration_screen_passed"]
        groups.append(item)
    if {g["origin"] for g in groups if g["split"] == "validation"} != {"original", "mixed"}:
        raise ValueError("Both independent saved validation origins are required")
    return dict(groups=groups, per_source=sources, calibration_screen_passed=passed,
        screening_rule=dict(validation_mean_ratio_max=improvement_ratio, per_source_ratio_max=maximum_source_ratio,
            metrics=list(physical), command_memory_excluded=True),
        natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None)
