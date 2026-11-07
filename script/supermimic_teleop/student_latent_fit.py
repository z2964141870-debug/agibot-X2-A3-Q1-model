"""Offline teacher-latent supervision against a preserved matched action control."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_aggregate_fit import balanced_indices, checked_splits
from student_causal_fit import causal_records, source_groups, validation
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss, current_student


def latent_records(data, native):
    records = causal_records(data)
    full = np.asarray(data["full_goal"])
    if (full.shape != (len(records["clip_id"]), 2254) or not np.isfinite(full).all()
            or not np.all(full[:, 2160:2194] == 1.)):
        raise ValueError("Require the saved full teacher goal, only as an offline label")
    device = next(native.policy.parameters()).device
    labels, maximum = [], 0.
    with torch.no_grad():
        for start in range(0, len(full), 128):
            goal = torch.as_tensor(full[start:start+128], device=device)
            history = torch.as_tensor(records["history"][start:start+128], device=device)
            target = torch.as_tensor(records["teacher_action"][start:start+128], device=device)
            latent = native.policy.human_encoder(goal)
            action = native.policy.decoder(latent, history)
            difference = float((action[:, native.present]-target[:, native.present]).abs().max())
            maximum = max(maximum, difference)
            if (not torch.isfinite(latent).all() or not torch.isfinite(action).all()
                    or not torch.allclose(action[:, native.present], target[:, native.present],
                                          atol=2e-5, rtol=2e-5)):
                raise ValueError("Teacher latent does not reproduce the saved same-state action")
            labels.append(latent.detach().cpu().numpy().copy())
    result = dict(records, teacher_latent=np.concatenate(labels))
    return result, dict(records=len(full), latent_width=result["teacher_latent"].shape[1],
        maximum_action_difference=maximum, future_goal_is_offline_label_only=True,
        student_input_arrays=["f0", "history"], labels_read_by_deployed_student=False)


def fit_latent(native, origins, payload, settings, control, control_trace, output, deadline_ns, provenance):
    checked_splits(*origins)
    output = Path(output)
    if any((output/name).exists() for name in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("Latent-supervision outputs are single-use")
    before_teacher, before_decoder = module_sha(native.policy), module_sha(native.policy.decoder)
    if (before_teacher != payload["frozen_teacher_policy_sha256"]
            or before_decoder != payload["decoder_sha256"]
            or control.get("actual_updates") != settings["maximum_updates"]
            or control.get("arm") != "action_only" or control.get("auxiliary_weight") != 0
            or control.get("actual_adam_steps") != [settings["maximum_updates"]]*6):
        raise ValueError("Preserve the actual trained matched control and frozen teacher")
    device, present = next(native.policy.parameters()).device, native.present
    for data in origins:
        for values in data.values():
            labels = values["teacher_latent"]
            if labels.shape != (len(values["clip_id"]), native.policy.cfg.latent) or not np.isfinite(labels).all():
                raise ValueError("Invalid detached teacher latent labels")
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    student = current_student(native, payload["current_statistics"]).train()
    student.encoder.load_state_dict(payload["encoder"], strict=True)
    initial_sha = module_sha(student.encoder)
    if initial_sha != control["initial_encoder_sha256"]:
        raise ValueError("Reused action control has another encoder initialization")
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    controls = {row["update"]: row for row in control_trace}
    expected_steps = [0]+list(range(settings["validation_every_updates"], settings["maximum_updates"]+1,
                                  settings["validation_every_updates"]))
    if sorted(controls) != expected_steps:
        raise ValueError("Reused control has missing or extra validation checkpoints")
    trace, best = [], None
    digest = hashlib.sha256()
    counts = {f"{origin}:{clip}":0 for origin, data in enumerate(origins)
              for clip in sorted(set(data["train"]["clip_id"]))}
    started = time.monotonic_ns()

    def evaluate(step, loss=None, action_loss=None, latent_loss=None, gradient=None):
        nonlocal best
        risks, details, latent_risks = [], [], []
        for origin, data in enumerate(origins):
            risk, rows = validation(student, data["validation"], present, device)
            risks.append(risk)
            details.extend(dict(row, state_origin=origin) for row in rows)
            groups = []
            student.eval()
            with torch.no_grad():
                for indices in source_groups(data["validation"]).values():
                    total = 0.
                    for start in range(0, len(indices), 128):
                        ids = indices[start:start+128]
                        goal, label = (torch.as_tensor(data["validation"][key][ids], device=device)
                                       for key in ("f0", "teacher_latent"))
                        total += float((student.encoder(goal)-label).square().mean())*len(ids)
                    groups.append(total/len(indices))
            latent_risks.append(float(np.mean(groups)))
            student.train()
        risk = float(np.mean(risks))
        reference = controls[step]
        if (not np.isfinite(risk) or not np.isfinite(latent_risks).all()
                or reference["batch_identity_sha256"] != digest.hexdigest()
                or reference["source_sample_counts"] != counts):
            raise ValueError("Reused control and latent fit differ in exact batch exposures")
        if step == 0 and abs(risk-reference["validation_risk"]) > 2e-7:
            raise ValueError("Initial control action risk differs on preserved rows")
        if best is None or risk < best["risk"]:
            best = dict(update=step, risk=risk, state=cpu_state(student.encoder))
        trace.append(dict(update=step, loss=loss, action_loss=action_loss, latent_loss=latent_loss,
            gradient_norm=gradient, validation_risk=risk, validation_origin_risks=risks,
            validation=details, validation_latent_mse=float(np.mean(latent_risks)),
            validation_latent_origin_mse=latent_risks,
            reused_control_validation_risk=reference["validation_risk"],
            batch_identity_sha256=digest.hexdigest(), source_sample_counts=dict(counts),
            elapsed_seconds=(time.monotonic_ns()-started)/1e9))
        save_json(output/"training_trace.json", trace)
        save_json(output/"progress.json", trace[-1])

    evaluate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            raise RuntimeError("No unequal-exposure result may replace the registered latent comparison")
        choices = balanced_indices(origins[0]["train"], origins[1]["train"], rng,
                                   settings["batch_size"], update)
        digest.update(np.asarray(choices, dtype="<i8").tobytes())
        tensors = []
        for origin, data in enumerate(origins):
            ids = np.asarray([i for key, i in choices if key == origin], dtype=np.int64)
            value = data["train"]
            for clip in value["clip_id"][ids]:
                counts[f"{origin}:{int(clip)}"] += 1
            tensors.append(tuple(torch.as_tensor(value[key][ids], device=device) for key in
                                 ("f0", "history", "teacher_action", "teacher_latent")))
        goal, history, target, label = (torch.cat([part[i] for part in tensors]) for i in range(4))
        optimizer.zero_grad(set_to_none=True)
        latent = student.encoder(goal)
        action_error = action_distillation_loss(student.decoder(latent, history), target, present)
        latent_error = (latent-label).square().mean()
        loss = action_error+settings["latent_weight"]*latent_error
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
            settings["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        if update % settings["validation_every_updates"] == 0:
            evaluate(update, float(loss.detach()), float(action_error.detach()),
                     float(latent_error.detach()), float(gradient.detach()))
    counters = [int(value["step"].item()) for value in optimizer.state.values()]
    if (counters != [settings["maximum_updates"]]*6 or digest.hexdigest() != control["batch_identity_sha256"]
            or counts != control["source_sample_counts"] or module_sha(native.policy) != before_teacher
            or module_sha(student.decoder) != before_decoder):
        raise ValueError("Adam exposures, teacher or shared decoder changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
        method="offline_teacher_latent_distillation", provenance=provenance,
        current_statistics=payload["current_statistics"], human_real_future_frames=0,
        history_frames=10, history_frame_dim=226, decoder_sha256=before_decoder,
        frozen_teacher_policy_sha256=before_teacher, actual_updates=settings["maximum_updates"],
        actual_adam_steps=counters, latent_weight=settings["latent_weight"],
        selected_update=best["update"], selected_validation_risk=best["risk"], selection_split="validation",
        selection_weighting="equal state origins then equal source clips", test_used_for_selection=False,
        teacher_latent_label_deployed=False, auxiliary_head_deployed=False,
        original_training_restarted=False, user_accepted_checkpoint=None)
    for name, values in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                             optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, values)
        with (output/name).open("xb") as stream:
            torch.save(values, stream)
        audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
    student.encoder.load_state_dict(best["state"], strict=True)
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", initial_encoder_sha256=initial_sha,
        selected_encoder_sha256=module_sha(student.encoder),
        batch_identity_sha256=digest.hexdigest(), source_sample_counts=counts,
        initial_validation_risk=trace[0]["validation_risk"], validation_improved=best["update"] != 0,
        reused_control_actual_updates=control["actual_updates"], control_retrained=False,
        elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9, new_physical_transitions=0,
        test_sources_read=False, hardware_latency_ms=None, natural_teleoperation_conclusion=None)
    save_json(output/"fit_result.json", result)
    return result
