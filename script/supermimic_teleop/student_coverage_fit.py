"""Freeze R2 statistics and decoder while testing broader owned-state coverage."""

from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import batch, validation
from student_coverage_data import balanced_indices, cells, combine
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss, current_student


def evaluate_cells(student, data, sources, present, device):
    rows, group_risks = [], []
    for group, clips in cells(data, sources).items():
        clip_risks = []
        for clip, origins in clips.items():
            origin_risks = []
            for origin, indices in origins.items():
                total = 0.
                with torch.no_grad():
                    for start in range(0, len(indices), 128):
                        current = indices[start:start+128]
                        goal, history, target = batch(data, current, device)
                        total += float(action_distillation_loss(student(history, goal), target, present))*len(current)
                risk = total/len(indices)
                rows.append(dict(group_id=group, source_clip_id=clip, state_origin=origin,
                    records=len(indices), action_mse=risk))
                origin_risks.append(risk)
            clip_risks.append(float(np.mean(origin_risks)))
        group_risks.append(float(np.mean(clip_risks)))
    return float(np.mean(group_risks)), rows


def fit(native, origins, payload, settings, plan, output, deadline_ns, provenance):
    data, sources = combine(origins, plan)
    output = Path(output)
    if any((output/n).exists() for n in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("Coverage outputs are single-use")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    student = current_student(native, payload["current_statistics"]).train()
    student.encoder.load_state_dict(payload["encoder"], strict=True)
    teacher_sha, decoder_sha = module_sha(native.policy), module_sha(student.decoder)
    if teacher_sha != payload["frozen_teacher_policy_sha256"] or decoder_sha != payload["decoder_sha256"]:
        raise ValueError("Coverage warm start or decoder belongs to another teacher")
    device, present = next(student.parameters()).device, native.present
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    groups = cells(data["train"], sources)
    exposures = {f"{clip}:{origin}":0 for clips in groups.values() for clip, values in clips.items() for origin in values}
    trace, best, updates, stale, initial_anchors = [], None, 0, 0, None
    started, stop_reason = time.monotonic_ns(), "MAXIMUM_UPDATES"

    def evaluate(step, loss=None, gradient=None):
        nonlocal best, stale, initial_anchors
        expanded, rows = evaluate_cells(student, data["validation"], sources, present, device)
        anchors = [validation(student, values["validation"], present, device)[0] for values in origins[:2]]
        if not np.isfinite([expanded, *anchors]).all():
            raise ValueError("Nonfinite coverage validation risk")
        if initial_anchors is None:
            initial_anchors = anchors
        eligible = all(value <= initial+1e-12 for value, initial in zip(anchors, initial_anchors))
        if best is None or (eligible and expanded < best["risk"]):
            best, stale = dict(update=step, risk=expanded, anchors=anchors, state=cpu_state(student.encoder)), 0
        else:
            stale += 1
        trace.append(dict(update=step, loss=loss, gradient_norm=gradient, validation_risk=expanded,
            validation=rows, fixed_anchor_r1_risk=anchors[0], fixed_anchor_mixed_risk=anchors[1],
            fixed_anchor_non_regression=eligible, stale_validations=stale,
            source_sample_counts=dict(exposures), elapsed_seconds=(time.monotonic_ns()-started)/1e9))
        save_json(output/"training_trace.json", trace)
        save_json(output/"training_progress.json", trace[-1])

    evaluate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            stop_reason = "RESERVED_WALL_BUDGET"
            break
        indices = balanced_indices(groups, rng, settings["batch_size"], (update-1)*settings["batch_size"])
        for clip, origin in zip(data["train"]["clip_id"][indices], data["train"]["origin_id"][indices]):
            exposures[f"{int(clip)}:{int(origin)}"] += 1
        goal, history, target = batch(data["train"], indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = action_distillation_loss(student(history, goal), target, present)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
            settings["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        updates = update
        if update % settings["validation_every_updates"] == 0:
            evaluate(update, float(loss.detach().cpu()), float(gradient.detach().cpu()))
            if stale >= settings["early_stop_stale_validations"]:
                stop_reason = "VALIDATIONS_WITHOUT_ELIGIBLE_IMPROVEMENT"
                break
    if updates and trace[-1]["update"] != updates:
        evaluate(updates, float(loss.detach().cpu()), float(gradient.detach().cpu()))
    counters = [int(v["step"].item()) for v in optimizer.state.values()]
    if not updates or len(counters) != 6 or any(v != updates for v in counters):
        raise ValueError("Coverage actual Adam steps are incomplete")
    if module_sha(native.policy) != teacher_sha or module_sha(student.decoder) != decoder_sha:
        raise ValueError("Frozen teacher or decoder changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
        method="owned_source_coverage", provenance=provenance, current_statistics=payload["current_statistics"],
        human_real_future_frames=0, history_frames=10, history_frame_dim=226,
        frozen_teacher_policy_sha256=teacher_sha, decoder_sha256=decoder_sha,
        actual_updates=updates, actual_adam_steps=counters, selected_update=best["update"],
        selected_validation_risk=best["risk"], selected_fixed_anchor_risks=best["anchors"],
        selection_split="validation", test_used_for_selection=False,
        selection_weighting="equal source groups then clips then available state origins",
        selection_rule="expanded validation minimum subject to both fixed anchors no worse than update0",
        stop_reason=stop_reason, original_training_restarted=False, user_accepted_checkpoint=None)
    for name, value in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder), optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, value)
        with (output/name).open("xb") as stream:
            torch.save(value, stream)
        audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", initial_validation_risk=trace[0]["validation_risk"],
        initial_fixed_anchor_risks=initial_anchors, validation_improved=best["update"] != 0,
        elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9, source_sample_counts=exposures,
        source_coverage={s:dict(records=len(d["clip_id"]), clips=len(set(d["clip_id"])),
            groups=len(cells(d, sources))) for s, d in data.items()},
        normalization_refitted=False, input_arrays_read=["f0", "history", "teacher_action", "clip_id"],
        test_sources_read=False, source_preprocessing="offline_unverified", natural_teleoperation_qualified=False,
        hardware_latency_ms=None)
    save_json(output/"fit_result.json", result)
    return result
