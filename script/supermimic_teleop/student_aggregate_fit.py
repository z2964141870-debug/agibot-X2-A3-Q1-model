"""One warm-start f0 fit with explicit original/new state-source balancing."""

from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import batch, causal_records, source_groups, validation
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss, current_student


def balanced_indices(original, aggregate, rng, count, update):
    if count < 2:
        raise ValueError("A batch must include both state origins")
    selected = []
    for origin, data in enumerate((original, aggregate)):
        groups = source_groups(data)
        sources = sorted(groups)
        size = (count+1-origin)//2
        chosen = [int(rng.choice(groups[sources[(i+update) % len(sources)]])) for i in range(size)]
        selected.extend((origin, index) for index in chosen)
    rng.shuffle(selected)
    return selected


def checked_splits(original, aggregate):
    result = []
    for values in (original, aggregate):
        if set(values) != {"train", "validation"}:
            raise ValueError("Require train and validation only")
        result.append({split: causal_records(data) for split, data in values.items()})
    train = set().union(*(set(value["train"]["clip_id"]) for value in result))
    valid = set().union(*(set(value["validation"]["clip_id"]) for value in result))
    if train & valid:
        raise ValueError("Original source ownership crosses training and validation")
    for split in ("train", "validation"):
        if set(result[0][split]["clip_id"]) != set(result[1][split]["clip_id"]):
            raise ValueError("Aggregation must retain every original source group")
    return result


def fit(native, original, aggregate, payload, settings, output, deadline_ns, provenance):
    original, aggregate = checked_splits(original, aggregate)
    output = Path(output)
    if any((output/name).exists() for name in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("Aggregation training outputs are single-use")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    student = current_student(native, payload["current_statistics"]).train()
    student.encoder.load_state_dict(payload["encoder"], strict=True)
    before_teacher, before_decoder = module_sha(native.policy), module_sha(student.decoder)
    if (before_teacher != payload["frozen_teacher_policy_sha256"]
            or before_decoder != payload["decoder_sha256"]):
        raise ValueError("Warm start belongs to another teacher or decoder")
    device, present = next(student.parameters()).device, native.present
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    trace, best, stale, updates = [], None, 0, 0
    exposures = {f"{origin}:{source}": 0 for origin, data in enumerate((original, aggregate))
                 for source in source_groups(data["train"])}
    started = time.monotonic_ns()
    stop_reason = "MAXIMUM_UPDATES"

    def evaluate(step, loss=None, gradient=None):
        nonlocal best, stale
        risks, rows = [], []
        for origin, data in enumerate((original, aggregate)):
            risk, detail = validation(student, data["validation"], present, device)
            risks.append(risk)
            rows.extend(dict(value, state_origin=origin) for value in detail)
        risk = float(np.mean(risks))
        if not np.isfinite(risk):
            raise ValueError("Nonfinite aggregate validation risk")
        if best is None or risk < best["risk"]:
            best, stale = dict(update=step, risk=risk, state=cpu_state(student.encoder)), 0
        else:
            stale += 1
        trace.append(dict(update=step, loss=loss, gradient_norm=gradient, validation_risk=risk,
            validation=rows, validation_origin_risks=risks, stale_validations=stale,
            source_sample_counts=dict(exposures), elapsed_seconds=(time.monotonic_ns()-started)/1e9))
        save_json(output/"training_trace.json", trace)
        save_json(output/"training_progress.json", trace[-1])

    evaluate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            stop_reason = "RESERVED_WALL_BUDGET"
            break
        choices = balanced_indices(original["train"], aggregate["train"], rng,
                                    settings["batch_size"], update)
        tensors = []
        for origin, data in enumerate((original, aggregate)):
            indices = np.asarray([index for key, index in choices if key == origin], dtype=np.int64)
            tensors.append(batch(data["train"], indices, device))
            for source in data["train"]["clip_id"][indices]:
                exposures[f"{origin}:{int(source)}"] += 1
        goal, history, target = (torch.cat([part[i] for part in tensors]) for i in range(3))
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
                stop_reason = "VALIDATIONS_WITHOUT_IMPROVEMENT"
                break
    if updates and trace[-1]["update"] != updates:
        evaluate(updates, float(loss.detach().cpu()), float(gradient.detach().cpu()))
    if not updates:
        raise RuntimeError("No actual aggregation fit update inside the deadline")
    counters = [int(row["step"].item()) for row in optimizer.state.values()]
    if not counters or any(value != updates for value in counters):
        raise ValueError("Aggregation Adam counters differ from completed updates")
    if module_sha(native.policy) != before_teacher or module_sha(student.decoder) != before_decoder:
        raise ValueError("Original teacher or frozen decoder changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
        method="one_round_student_state_aggregation", provenance=provenance,
        current_statistics=payload["current_statistics"], human_real_future_frames=0,
        history_frames=10, history_frame_dim=226, decoder_sha256=before_decoder,
        frozen_teacher_policy_sha256=before_teacher, actual_updates=updates,
        selected_update=best["update"], selected_validation_risk=best["risk"], selection_split="validation",
        selection_weighting="equal state origins then equal source clips", test_used_for_selection=False,
        actual_adam_steps=counters, stop_reason=stop_reason, original_training_restarted=False,
        user_accepted_checkpoint=None)
    for name, values in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                             optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, values)
        with (output/name).open("xb") as stream:
            torch.save(values, stream)
        audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", source_sample_counts=exposures,
        elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
        initial_validation_risk=trace[0]["validation_risk"], validation_improved=best["update"] != 0,
        state_origin_records={str(i): {split:len(value[split]["clip_id"]) for split in value}
                              for i, value in enumerate((original, aggregate))},
        input_arrays_read=["f0", "history", "teacher_action", "clip_id"],
        test_sources_read=False, source_preprocessing="offline_unverified",
        natural_teleoperation_conclusion=None, hardware_latency_ms=None)
    save_json(output/"fit_result.json", result)
    return result
