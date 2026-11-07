"""Physical distillation with exact exposure reuse of the settled action control."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_aggregate_fit import balanced_indices, checked_splits
from student_causal_fit import source_groups, validation
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss, current_student
from wm_probe_worker import tensor_state_sha


def fit_physical(native, physical, origins, payload, settings, control, control_trace,
                 output, deadline_ns, provenance):
    checked_splits(*origins)
    output = Path(output)
    if any((output/n).exists() for n in ("selected.pt", "last.pt", "training_trace.json")):
        raise FileExistsError("Physical fit outputs are single-use")
    before_teacher, before_decoder = module_sha(native.policy), module_sha(native.policy.decoder)
    before_wm = tensor_state_sha(physical.model.state_dict())
    if (before_teacher != payload["frozen_teacher_policy_sha256"]
            or before_decoder != payload["decoder_sha256"]
            or control.get("actual_updates") != settings["maximum_updates"]
            or control.get("arm") != "action_only" or control.get("auxiliary_weight") != 0
            or control.get("actual_adam_steps") != [settings["maximum_updates"]]*6):
        raise ValueError("Require the actual matched control and frozen teacher/decoder")
    device, present = next(native.policy.parameters()).device, native.present
    for origin in origins:
        for data in origin.values():
            n = len(data["clip_id"])
            if (data["physical_state"].shape != (n, 269) or data["wm_context"].shape != (n, 62)
                    or not np.isfinite(data["physical_state"]).all()
                    or not np.isfinite(data["wm_context"]).all()):
                raise ValueError("Missing same-row physical WM states and nominal context")
            labels = []
            for start in range(0, n, 128):
                state, target, context = (torch.as_tensor(data[key][start:start+128], device=device)
                    for key in ("physical_state", "teacher_action", "wm_context"))
                labels.append(physical.targets(state, target, context).cpu().numpy().copy())
            data["teacher_physical"] = np.concatenate(labels)
    torch.manual_seed(settings["seed"])
    rng = np.random.default_rng(settings["seed"])
    student = current_student(native, payload["current_statistics"]).train()
    student.encoder.load_state_dict(payload["encoder"], strict=True)
    initial_sha = module_sha(student.encoder)
    if initial_sha != control["initial_encoder_sha256"]:
        raise ValueError("Reused control has another encoder initialization")
    optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
    controls = {row["update"]: row for row in control_trace}
    expected = [0]+list(range(settings["validation_every_updates"], settings["maximum_updates"]+1,
                             settings["validation_every_updates"]))
    if sorted(controls) != expected or len(controls) != len(control_trace):
        raise ValueError("Reused control has missing, duplicate or extra validation points")
    trace, best, digest = [], None, hashlib.sha256()
    counts = {f"{origin}:{clip}": 0 for origin, data in enumerate(origins)
              for clip in sorted(set(data["train"]["clip_id"]))}
    started = time.monotonic_ns()

    def evaluate(step, loss=None, action_loss=None, physical_loss=None, gradient=None):
        nonlocal best
        risks, details, physical_risks = [], [], []
        for origin, data in enumerate(origins):
            risk, rows = validation(student, data["validation"], present, device)
            risks.append(risk)
            details.extend(dict(row, state_origin=origin) for row in rows)
            groups = []
            student.eval()
            with torch.no_grad():
                for ids in source_groups(data["validation"]).values():
                    total = 0.
                    for start in range(0, len(ids), 128):
                        indices = ids[start:start+128]
                        goal, history, state, context, target = (torch.as_tensor(
                            data["validation"][key][indices], device=device) for key in
                            ("f0", "history", "physical_state", "wm_context", "teacher_physical"))
                        total += float(physical.loss(state, student(history, goal), context, target))*len(indices)
                    groups.append(total/len(ids))
            physical_risks.append(float(np.mean(groups)))
            student.train()
        risk, reference = float(np.mean(risks)), controls[step]
        if (not np.isfinite(risk) or not np.isfinite(physical_risks).all()
                or reference["batch_identity_sha256"] != digest.hexdigest()
                or reference["source_sample_counts"] != counts):
            raise ValueError("Reused control and physical fit differ in exact batch exposures")
        if step == 0 and abs(risk-reference["validation_risk"]) > 2e-7:
            raise ValueError("Initial control action risk differs on the preserved rows")
        if best is None or risk < best["risk"]:
            best = dict(update=step, risk=risk, state=cpu_state(student.encoder))
        trace.append(dict(update=step, loss=loss, action_loss=action_loss, physical_loss=physical_loss,
            gradient_norm=gradient, validation_risk=risk, validation_origin_risks=risks,
            validation=details, validation_physical_mse=float(np.mean(physical_risks)),
            validation_physical_origin_mse=physical_risks,
            reused_control_validation_risk=reference["validation_risk"],
            batch_identity_sha256=digest.hexdigest(), source_sample_counts=dict(counts),
            wm_step_calls=physical.calls, elapsed_seconds=(time.monotonic_ns()-started)/1e9))
        save_json(output/"training_trace.json", trace)
        save_json(output/"progress.json", trace[-1])

    evaluate(0)
    for update in range(1, settings["maximum_updates"]+1):
        if time.monotonic_ns() >= deadline_ns-30_000_000_000:
            raise RuntimeError("Do not compare an incomplete physical fit with the full control")
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
                ("f0", "history", "teacher_action", "physical_state", "wm_context", "teacher_physical")))
        goal, history, target, state, context, teacher_features = (
            torch.cat([part[i] for part in tensors]) for i in range(6))
        optimizer.zero_grad(set_to_none=True)
        action = student(history, goal)
        action_error = action_distillation_loss(action, target, present)
        physical_error = physical.loss(state, action, context, teacher_features)
        loss = action_error+settings["physical_weight"]*physical_error
        if not torch.isfinite(action).all() or not torch.isfinite(loss):
            raise ValueError("Nonfinite physical-distillation action or loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(student.trainable_parameters(),
            settings["gradient_norm_limit"], error_if_nonfinite=True)
        optimizer.step()
        if update % settings["validation_every_updates"] == 0:
            evaluate(update, float(loss.detach()), float(action_error.detach()),
                     float(physical_error.detach()), float(gradient.detach()))
    counters = [int(value["step"].item()) for value in optimizer.state.values()]
    if (counters != [settings["maximum_updates"]]*6 or digest.hexdigest() != control["batch_identity_sha256"]
            or counts != control["source_sample_counts"] or module_sha(native.policy) != before_teacher
            or module_sha(student.decoder) != before_decoder
            or tensor_state_sha(physical.model.state_dict()) != before_wm
            or any(p.grad is not None or p.requires_grad for p in physical.model.parameters())):
        raise ValueError("Adam exposures or frozen teacher/decoder/WM changed")
    from upper_mps_forecast import audit_cpu_tensors, cpu_tree
    common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
        method="frozen_fp32_wm_physical_distillation", provenance=provenance,
        current_statistics=payload["current_statistics"], human_real_future_frames=0,
        history_frames=10, history_frame_dim=226, decoder_sha256=before_decoder,
        frozen_teacher_policy_sha256=before_teacher, frozen_wm_tensor_sha256=before_wm,
        actual_updates=settings["maximum_updates"], actual_adam_steps=counters,
        physical_weight=settings["physical_weight"], selected_update=best["update"],
        selected_validation_risk=best["risk"], selection_split="validation",
        selection_weighting="equal state origins then equal source clips", test_used_for_selection=False,
        world_model_deployed=False, auxiliary_head_deployed=False, original_training_restarted=False,
        user_accepted_checkpoint=None)
    for name, values in (("selected.pt", dict(common, encoder=best["state"])),
            ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                             optimizer=cpu_tree(torch, optimizer.state_dict())))):
        audit_cpu_tensors(torch, values)
        with (output/name).open("xb") as stream:
            torch.save(values, stream)
        audit_cpu_tensors(torch, torch.load(output/name, map_location="cpu", weights_only=True))
    student.encoder.load_state_dict(best["state"], strict=True)
    result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", initial_encoder_sha256=initial_sha,
        selected_encoder_sha256=module_sha(student.encoder), batch_identity_sha256=digest.hexdigest(),
        source_sample_counts=counts, initial_validation_risk=trace[0]["validation_risk"],
        validation_improved=best["update"] != 0, reused_control_actual_updates=control["actual_updates"],
        control_retrained=False, wm_step_calls=physical.calls, scalar_backwards=settings["maximum_updates"],
        elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9, new_physical_transitions=0,
        test_sources_read=False, hardware_latency_ms=None, natural_teleoperation_conclusion=None)
    save_json(output/"fit_result.json", result)
    return result
