"""R1 action distillation on collected states; validation selects, test is absent."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from student_models import action_distillation_loss, masked_preview_goal, short_preview_student


HORIZONS = (10, 5, 4, 2, 1)


def module_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def save_json(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False)+"\n"
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def batch_goal(data, indices, columns, device):
    counts = data["prefix"][indices, columns]
    keep = counts > 0
    indices, counts = indices[keep], counts[keep]
    if not len(indices):
        raise ValueError("Selected batch has no available first target; do not fit an all-zero goal")
    full = torch.as_tensor(data["full_goal"][indices], device=device)
    goal = masked_preview_goal(full[:, :2160], full[:, -60:],
                              torch.as_tensor(counts, dtype=torch.long, device=device))
    history = torch.as_tensor(data["history"][indices], device=device)
    labels = torch.as_tensor(data["teacher_action"][indices], device=device)
    return goal, history, labels, counts


def sampling_groups(data):
    groups = []
    for column, horizon in enumerate(HORIZONS):
        eligible = np.flatnonzero(data["prefix"][:, column] > 0)
        by_source = {int(key): eligible[data["clip_id"][eligible] == key]
                     for key in np.unique(data["clip_id"][eligible])}
        if not by_source:
            raise ValueError("No available first target for requested H"+str(horizon))
        groups.append(by_source)
    return groups


def sample_batch(groups, rng, size, update):
    columns = (np.arange(size)+update) % len(HORIZONS)
    rng.shuffle(columns)
    indices = []
    for column in columns:
        sources = groups[column]
        source = int(rng.choice(tuple(sources)))
        indices.append(int(rng.choice(sources[source])))
    return np.asarray(indices), columns


def validation(student, data, present, device):
    student.eval()
    values = []
    with torch.no_grad():
        for column, horizon in enumerate(HORIZONS):
            eligible = np.flatnonzero(data["prefix"][:, column] > 0)
            if not len(eligible):
                raise ValueError("Validation has no valid samples for requested H"+str(horizon))
            total = 0.
            for start in range(0, len(eligible), 128):
                indices = eligible[start:start+128]
                goal, history, target, _ = batch_goal(data, indices,
                    np.full(len(indices), column), device)
                total += float(action_distillation_loss(student(history, goal), target, present))*len(indices)
            values.append(dict(horizon=horizon, samples=len(eligible), total_records=len(data["prefix"]),
                unavailable_first_target=len(data["prefix"])-len(eligible), action_mse=total/len(eligible)))
    student.train()
    return float(np.mean([value["action_mse"] for value in values])), values


def fit_r1(native, data, settings, output, deadline_ns, provenance):
    if set(data) != {"train", "validation"}:
        raise ValueError("Fit accepts train and validation only, never test data")
    output = Path(output)
    if any((output/name).exists() for name in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("Training outputs are single-use")
    torch.manual_seed(2026110701)
    rng = np.random.default_rng(2026110701)
    student = short_preview_student(native).train()
    device = next(student.parameters()).device
    present = native.present.to(device)
    before_teacher = module_sha(native.policy)
    before_decoder = module_sha(student.decoder)
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    train = data["train"]
    groups = sampling_groups(train)
    trace, best, updates = [], None, 0
    requested = np.zeros(5, dtype=np.int64)
    visible = np.zeros(11, dtype=np.int64)
    started, stale_validations = time.monotonic_ns(), 0
    stop_reason = "MAXIMUM_UPDATES"

    def validate(step, loss=None, grad=None):
        nonlocal best, stale_validations
        risk, rows = validation(student, data["validation"], present, device)
        if not np.isfinite(risk):
            raise ValueError("Nonfinite validation score")
        if best is None or risk < best["risk"]:
            best = dict(update=step, risk=risk, state=cpu_state(student.encoder))
            stale_validations = 0
        else:
            stale_validations += 1
        record = dict(update=step, loss=loss, gradient_norm=grad, validation_risk=risk,
            validation=rows, elapsed_seconds=(time.monotonic_ns()-started)/1e9,
            requested_horizon_counts=requested.tolist(), actual_visible_prefix_counts=visible.tolist(),
            stale_validations=stale_validations)
        trace.append(record)
        save_json(output/"training_trace.json", trace)
        save_json(output/"training_progress.json", record)

    validate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            stop_reason = "RESERVED_WALL_BUDGET"
            break
        indices, columns = sample_batch(groups, rng, settings["batch_size"], update)
        requested += np.bincount(columns, minlength=5)
        goal, history, labels, counts = batch_goal(train, indices, columns, device)
        visible += np.bincount(counts, minlength=11)
        optimizer.zero_grad(set_to_none=True)
        loss = action_distillation_loss(student(history, goal), labels, present)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
            settings["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        updates = update
        if update % settings["initial_validation_then_every_updates"] == 0:
            validate(update, float(loss.detach().cpu()), float(grad.detach().cpu()))
            if stale_validations >= 6:
                stop_reason = "SIX_VALIDATIONS_WITHOUT_IMPROVEMENT"
                break
    if updates and trace[-1]["update"] != updates:
        validate(updates, float(loss.detach().cpu()), float(grad.detach().cpu()))
    if not updates:
        raise RuntimeError("No optimization step fit inside the reserved collection/training budget")
    counters = [int(value["step"].item()) for value in optimizer.state.values()]
    if not counters or any(count != updates for count in counters):
        raise ValueError("Actual Adam steps differ from completed updates")
    if module_sha(native.policy) != before_teacher or module_sha(student.decoder) != before_decoder:
        raise ValueError("Original teacher or frozen shared decoder changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree

    common = dict(schema="q1_low_latency_student_encoder_v1", interface="human_prefix_adapter",
        provenance=provenance, decoder_sha256=before_decoder, frozen_teacher_policy_sha256=before_teacher,
        actual_updates=updates, selection_split="validation", test_used_for_selection=False,
        selected_update=best["update"], selected_validation_risk=best["risk"],
        actual_adam_steps=counters, stop_reason=stop_reason, original_training_restarted=False,
        user_accepted_checkpoint=None)
    for filename, payload in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                             optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, payload)
        with (output/filename).open("xb") as stream:
            torch.save(payload, stream)
        audit_cpu_tensors(torch, torch.load(output/filename, map_location="cpu", weights_only=True))
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
        validation_improved=best["update"] != 0, initial_validation_risk=trace[0]["validation_risk"],
        requested_horizon_counts=requested.tolist(), actual_visible_prefix_counts=visible.tolist(),
        test_sources_read=False, natural_teleoperation_conclusion=None, hardware_latency_ms=None)
    save_json(output/"fit_result.json", result)
    return result
