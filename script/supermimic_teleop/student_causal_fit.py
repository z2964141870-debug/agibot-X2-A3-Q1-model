"""Fit an f0/history encoder on preserved teacher states, without future inputs."""

from pathlib import Path
import time

import numpy as np
import torch

from student_fit import cpu_state, module_sha, save_json
from student_inputs import FRAME_DIM, GEOMETRY_DIM, HISTORY, fit_current_statistics
from student_models import action_distillation_loss, current_student


INPUT_FIELDS = ("f0", "history", "teacher_action", "clip_id")


def causal_records(data):
    selected = {name: np.asarray(data[name]) for name in INPUT_FIELDS}
    count = len(selected["clip_id"])
    shapes = dict(f0=(count, HISTORY, FRAME_DIM), history=(count, 930),
                  teacher_action=(count, 29), clip_id=(count,))
    if not count or any(selected[name].shape != shape or not np.isfinite(selected[name]).all()
                        for name, shape in shapes.items()):
        raise ValueError("Require finite matched f0/history/action records")
    if not np.issubdtype(selected["clip_id"].dtype, np.integer):
        raise ValueError("Preserve integer source identities")
    values = selected["f0"]
    valid, exact, age = (values[..., GEOMETRY_DIM+i] for i in range(3))
    if (not np.isin(valid, (0., 1.)).all() or not np.isin(exact, (0., 1.)).all()
            or np.any(exact > valid) or np.any((exact == 1.) & (age != 0.))
            or not np.all(valid[:, -1] == 1.)
            or np.any(age < 0.) or np.any(age > .200001)
            or np.any(values[valid == 0.] != 0.)):
        raise ValueError("Invalid current support, validity flags or bounded capture age")
    return selected


def source_groups(data):
    return {int(key): np.flatnonzero(data["clip_id"] == key)
            for key in np.unique(data["clip_id"])}


def sample_sources(groups, rng, count, update):
    sources = np.asarray(sorted(groups))
    chosen = sources[(np.arange(count)+update) % len(sources)]
    rng.shuffle(chosen)
    return np.asarray([rng.choice(groups[int(key)]) for key in chosen], dtype=np.int64)


def batch(data, indices, device):
    return tuple(torch.as_tensor(data[name][indices], device=device)
                 for name in ("f0", "history", "teacher_action"))


def validation(student, data, present, device):
    student.eval()
    rows = []
    with torch.no_grad():
        for source, indices in source_groups(data).items():
            total = 0.
            for start in range(0, len(indices), 128):
                current = indices[start:start+128]
                goal, history, target = batch(data, current, device)
                total += float(action_distillation_loss(student(history, goal), target, present))*len(current)
            rows.append(dict(source_clip_id=source, records=len(indices), action_mse=total/len(indices)))
    student.train()
    return float(np.mean([row["action_mse"] for row in rows])), rows


def fit_r2(native, data, settings, output, deadline_ns, provenance):
    if set(data) != {"train", "validation"}:
        raise ValueError("Fit accepts train and validation only, never test data")
    selected = {split: causal_records(values) for split, values in data.items()}
    if set(selected["train"]["clip_id"]) & set(selected["validation"]["clip_id"]):
        raise ValueError("Preserved source identities cross train/validation splits")
    output = Path(output)
    if any((output/name).exists() for name in ("selected.pt", "last.pt", "training_trace.json", "current_statistics.json")):
        raise FileExistsError("Training outputs are single-use")
    statistics = fit_current_statistics(selected["train"]["f0"], split="train")
    save_json(output/"current_statistics.json", statistics)
    torch.manual_seed(2026110702)
    rng = np.random.default_rng(2026110702)
    before_teacher = module_sha(native.policy)
    student = current_student(native, statistics).train()
    device = next(student.parameters()).device
    present = native.present.to(device)
    before_decoder = module_sha(student.decoder)
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    groups = source_groups(selected["train"])
    requested = {source: 0 for source in groups}
    trace, best, updates, stale = [], None, 0, 0
    started = time.monotonic_ns()
    stop_reason = "MAXIMUM_UPDATES"

    def validate(step, loss=None, gradient=None):
        nonlocal best, stale
        risk, rows = validation(student, selected["validation"], present, device)
        if not np.isfinite(risk):
            raise ValueError("Nonfinite validation score")
        if best is None or risk < best["risk"]:
            best = dict(update=step, risk=risk, state=cpu_state(student.encoder))
            stale = 0
        else:
            stale += 1
        record = dict(update=step, loss=loss, gradient_norm=gradient, validation_risk=risk,
            validation=rows, stale_validations=stale,
            source_sample_counts={str(key): value for key, value in requested.items()},
            elapsed_seconds=(time.monotonic_ns()-started)/1e9)
        trace.append(record)
        save_json(output/"training_trace.json", trace)
        save_json(output/"training_progress.json", record)

    validate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            stop_reason = "RESERVED_WALL_BUDGET"
            break
        indices = sample_sources(groups, rng, settings["batch_size"], update)
        for source in groups:
            requested[source] += int(np.count_nonzero(selected["train"]["clip_id"][indices] == source))
        goal, history, target = batch(selected["train"], indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss = action_distillation_loss(student(history, goal), target, present)
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
    if not updates:
        raise RuntimeError("No optimization step fit inside the reserved training budget")
    counters = [int(value["step"].item()) for value in optimizer.state.values()]
    if not counters or any(count != updates for count in counters):
        raise ValueError("Actual Adam steps differ from completed updates")
    if module_sha(native.policy) != before_teacher or module_sha(student.decoder) != before_decoder:
        raise ValueError("Original teacher or frozen shared decoder changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
        provenance=provenance, current_statistics=statistics, human_real_future_frames=0,
        history_frames=HISTORY, history_frame_dim=FRAME_DIM,
        decoder_sha256=before_decoder, frozen_teacher_policy_sha256=before_teacher,
        actual_updates=updates, selected_update=best["update"], selected_validation_risk=best["risk"],
        selection_split="validation", selection_weighting="equal source clips", test_used_for_selection=False,
        actual_adam_steps=counters, stop_reason=stop_reason, original_training_restarted=False,
        user_accepted_checkpoint=None)
    for filename, payload in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                             optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, payload)
        with (output/filename).open("xb") as stream:
            torch.save(payload, stream)
        audit_cpu_tensors(torch, torch.load(output/filename, map_location="cpu", weights_only=True))
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP",
        elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
        initial_validation_risk=trace[0]["validation_risk"], validation_improved=best["update"] != 0,
        source_sample_counts={str(key): value for key, value in requested.items()},
        train_records=len(selected["train"]["clip_id"]), validation_records=len(selected["validation"]["clip_id"]),
        input_arrays_read=list(INPUT_FIELDS), privileged_teacher_label_scope="same preserved actual robot state",
        test_sources_read=False, source_preprocessing="offline_unverified",
        new_physical_transitions=0, natural_teleoperation_conclusion=None, hardware_latency_ms=None)
    save_json(output/"fit_result.json", result)
    return result
