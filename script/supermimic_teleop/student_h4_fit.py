"""H4-only continuation of the preserved R1 encoder, with validation-only selection."""

from pathlib import Path
import time

import numpy as np
import torch

from student_fit import batch_goal, cpu_state, module_sha, save_json, validation
from student_models import action_distillation_loss, short_preview_student

H4_COLUMN = 2


def source_groups(data):
    count = len(data["clip_id"])
    if (data["prefix"].shape != (count, 5)
            or not np.issubdtype(data["prefix"].dtype, np.integer)
            or np.any(data["prefix"] < 0)
            or np.any(data["prefix"] > np.asarray((10, 5, 4, 2, 1)))):
        raise ValueError("Invalid preserved H4 prefix counts")
    eligible = np.flatnonzero(data["prefix"][:, H4_COLUMN] > 0)
    groups = {int(key): eligible[data["clip_id"][eligible] == key]
              for key in np.unique(data["clip_id"])}
    if not groups or any(not len(values) for values in groups.values()):
        raise ValueError("Every owned source must have an available H4 first target")
    return groups


def sample_indices(groups, rng, size, offset):
    if size <= 0 or offset < 0 or not groups:
        raise ValueError("Require nonempty H4 groups and a valid sampling cursor")
    keys = sorted(groups)
    rows = np.asarray([rng.choice(groups[keys[i % len(keys)]])
                       for i in range(offset, offset+size)], dtype=np.int64)
    rng.shuffle(rows)
    return rows


def evaluate(student, data, present, device):
    student.eval()
    groups = source_groups(data)
    rows, total = [], 0.
    with torch.no_grad():
        for clip, indices in sorted(groups.items()):
            subtotal = 0.
            for start in range(0, len(indices), 128):
                current = indices[start:start+128]
                goal, history, target, _ = batch_goal(data, current,
                    np.full(len(current), H4_COLUMN), device)
                subtotal += float(action_distillation_loss(student(history, goal), target, present))*len(current)
            total += subtotal
            rows.append(dict(clip_id=clip, samples=len(indices), action_mse=subtotal/len(indices)))
    student.train()
    return float(np.mean([r["action_mse"] for r in rows])), total/sum(r["samples"] for r in rows), rows


def fit(native, data, payload, settings, output, deadline_ns, provenance):
    if set(data) != {"train", "validation"}:
        raise ValueError("Fit accepts train and validation only, never test")
    if set(data["train"]["clip_id"]) & set(data["validation"]["clip_id"]):
        raise ValueError("Source identities cross train and validation")
    groups = {split: source_groups(values) for split, values in data.items()}
    if (payload.get("schema") != "q1_low_latency_student_encoder_v1"
            or payload.get("interface") != "human_prefix_adapter"
            or payload.get("selection_split") != "validation"
            or payload.get("test_used_for_selection") is not False
            or payload.get("selected_update") != 5000):
        raise ValueError("Warm start must be the real validation-selected R1 update5000")
    output = Path(output)
    if any((output/name).exists() for name in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("H4 continuation outputs are single-use")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    student = short_preview_student(native).train()
    student.encoder.load_state_dict(payload["encoder"], strict=True)
    teacher_sha, decoder_sha = module_sha(native.policy), module_sha(student.decoder)
    if (teacher_sha != payload["frozen_teacher_policy_sha256"]
            or decoder_sha != payload["decoder_sha256"]):
        raise ValueError("H4 warm start belongs to another teacher or decoder")
    device, present = next(student.parameters()).device, native.present
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    requested, visible = 0, np.zeros(5, dtype=np.int64)
    exposure = {str(key):0 for key in groups["train"]}
    trace, best, updates, stale = [], None, 0, 0
    started, stop_reason = time.monotonic_ns(), "MAXIMUM_UPDATES"

    def validate(step, loss=None, gradient=None):
        nonlocal best, stale
        risk, weighted, rows = evaluate(student, data["validation"], present, device)
        _, horizons = validation(student, data["validation"], present, device)
        if not np.isfinite([risk, weighted, *[r["action_mse"] for r in horizons]]).all():
            raise ValueError("Nonfinite H4 validation result")
        if best is None or risk < best["risk"]:
            best, stale = dict(update=step, risk=risk, weighted_risk=weighted,
                               state=cpu_state(student.encoder)), 0
        else:
            stale += 1
        trace.append(dict(update=step, loss=loss, gradient_norm=gradient,
            validation_risk=risk, sample_weighted_h4_risk=weighted, validation=rows,
            diagnostic_horizons=horizons, selected_update_so_far=best["update"],
            stale_validations=stale, requested_h4_samples=requested,
            visible_prefix_counts=visible.tolist(), source_sample_counts=dict(exposure),
            elapsed_seconds=(time.monotonic_ns()-started)/1e9))
        save_json(output/"training_trace.json", trace)
        save_json(output/"training_progress.json", trace[-1])

    validate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            stop_reason = "RESERVED_WALL_BUDGET"
            break
        indices = sample_indices(groups["train"], rng, settings["batch_size"],
                                 (update-1)*settings["batch_size"])
        goal, history, labels, counts = batch_goal(data["train"], indices,
            np.full(len(indices), H4_COLUMN), device)
        requested += len(indices)
        visible += np.bincount(counts, minlength=5)
        for clip in data["train"]["clip_id"][indices]:
            exposure[str(int(clip))] += 1
        optimizer.zero_grad(set_to_none=True)
        loss = action_distillation_loss(student(history, goal), labels, present)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
            settings["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        updates = update
        if update % settings["validation_every_updates"] == 0:
            validate(update, float(loss.detach().cpu()), float(gradient.detach().cpu()))
            if stale >= settings["early_stop_stale_validations"]:
                stop_reason = "VALIDATIONS_WITHOUT_IMPROVEMENT"
                break
    if updates and trace[-1]["update"] != updates:
        validate(updates, float(loss.detach().cpu()), float(gradient.detach().cpu()))
    counters = [int(v["step"].item()) for v in optimizer.state.values()]
    if (not updates or len(counters) != len(list(student.trainable_parameters()))
            or any(value != updates for value in counters)):
        raise ValueError("Incomplete real H4 Adam counters")
    if module_sha(native.policy) != teacher_sha or module_sha(student.decoder) != decoder_sha:
        raise ValueError("H4 fit changed the frozen teacher or decoder")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_student_encoder_v1", interface="human_prefix_adapter",
        method="h4_only_continuation", allowed_request_horizons=[4], human_real_future_frames=4,
        maximum_preview_ns=80_000_000, provenance=provenance,
        decoder_sha256=decoder_sha, frozen_teacher_policy_sha256=teacher_sha,
        actual_updates=updates, actual_adam_steps=counters,
        selection_split="validation", selection_weighting="equal owned source clips at requested H4",
        test_used_for_selection=False, selected_update=best["update"],
        selected_validation_risk=best["risk"], selected_sample_weighted_h4_risk=best["weighted_risk"],
        selected_encoder_sha256=None, stop_reason=stop_reason,
        original_training_restarted=False, user_accepted_checkpoint=None)
    last_state = cpu_state(student.encoder)
    last_optimizer = cpu_tree(torch, optimizer.state_dict())
    student.encoder.load_state_dict(best["state"], strict=True)
    common["selected_encoder_sha256"] = module_sha(student.encoder)
    for name, value in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=last_state, optimizer=last_optimizer,
                             checkpoint_update=updates, used_for_selection=False))):
        audit_cpu_tensors(torch, value)
        with (output/name).open("xb") as stream:
            torch.save(value, stream)
        audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", validation_improved=best["update"] != 0,
        initial_validation_risk=trace[0]["validation_risk"],
        initial_sample_weighted_h4_risk=trace[0]["sample_weighted_h4_risk"],
        requested_h4_samples=requested, visible_prefix_counts=visible.tolist(),
        source_sample_counts=exposure, elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
        new_physical_transitions=0, normalization_refitted=False, test_sources_read=False,
        source_preprocessing="offline_unverified", natural_teleoperation_qualified=False,
        hardware_latency_ms=None)
    save_json(output/"fit_result.json", result)
    return result
