"""Paired residual adaptation with identical batches and validation-only selection."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from student_causal_fit import batch, validation
from student_coverage_data import balanced_indices, cells, combine
from student_coverage_fit import evaluate_cells
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss
from student_proprio_models import ARMS, residual_student, zero_adapter


def fit_pair(native, origins, payload, candidate, settings, output, deadline_ns, provenance):
    if (set(settings) != {"seed", "maximum_updates", "batch_size", "learning_rate",
            "gradient_norm_limit", "validation_every_updates"}
            or any(type(settings[k]) is not int or settings[k] <= 0
                for k in ("seed", "maximum_updates", "batch_size", "validation_every_updates"))
            or settings["maximum_updates"] % settings["validation_every_updates"]
            or not np.isfinite([settings["learning_rate"], settings["gradient_norm_limit"]]).all()
            or min(settings["learning_rate"], settings["gradient_norm_limit"]) <= 0):
        raise ValueError("Require finite paired settings and equal complete validation exposure")
    data, sources = combine(origins, candidate)
    output = Path(output)
    if any((output/arm).exists() for arm in ARMS):
        raise FileExistsError("Residual arms are single-use")
    teacher_sha = module_sha(native.policy)
    if (teacher_sha != payload["frozen_teacher_policy_sha256"]
            or module_sha(native.policy.decoder) != payload["decoder_sha256"]):
        raise ValueError("Warm start belongs to another frozen teacher or decoder")
    runs = []
    initial_adapter_sha = None
    for arm in ARMS:
        if time.monotonic_ns() >= deadline_ns-60_000_000_000:
            raise RuntimeError("Insufficient remaining paired fit budget")
        folder = output/arm
        folder.mkdir(exist_ok=False)
        torch.manual_seed(settings["seed"])
        rng = np.random.default_rng(settings["seed"])
        student = residual_student(native, payload, arm).train()
        encoder_sha, decoder_sha = module_sha(student.encoder), module_sha(student.decoder)
        adapter_sha = module_sha(student.adapter)
        if initial_adapter_sha is None:
            initial_adapter_sha = adapter_sha
        if adapter_sha != initial_adapter_sha or not zero_adapter(cpu_state(student.adapter)):
            raise ValueError("Both arms must start with the same exactly-zero latent residual")
        device = next(student.parameters()).device
        present = native.present.to(device)
        optimizer = torch.optim.Adam(student.trainable_parameters(), lr=settings["learning_rate"])
        groups = cells(data["train"], sources)
        exposures = {f"{clip}:{origin}":0 for clips in groups.values()
            for clip, values in clips.items() for origin in values}
        batch_digest = hashlib.sha256()
        trace, best, initial_anchors, updates = [], None, None, 0
        started = time.monotonic_ns()

        def evaluate(step, loss=None, gradient=None):
            nonlocal best, initial_anchors
            expanded, rows = evaluate_cells(student, data["validation"], sources, present, device)
            anchors = [validation(student, values["validation"], present, device)[0] for values in origins[:2]]
            if not np.isfinite([expanded, *anchors]).all():
                raise ValueError("Nonfinite residual validation risk")
            if initial_anchors is None:
                initial_anchors = anchors
            eligible = all(value <= initial+1e-12 for value, initial in zip(anchors, initial_anchors))
            if best is None or (eligible and expanded < best["risk"]):
                best = dict(update=step, risk=expanded, anchors=anchors, state=cpu_state(student.adapter))
            trace.append(dict(update=step, loss=loss, gradient_norm=gradient,
                validation_risk=expanded, validation=rows, fixed_anchor_r1_risk=anchors[0],
                fixed_anchor_mixed_risk=anchors[1], fixed_anchor_non_regression=eligible,
                batch_sha256=batch_digest.hexdigest(), source_sample_counts=dict(exposures),
                elapsed_seconds=(time.monotonic_ns()-started)/1e9))
            save_json(folder/"training_trace.json", trace)
            save_json(output/"training_progress.json", dict(arm=arm, **trace[-1]))

        evaluate(0)
        for update in range(1, settings["maximum_updates"]+1):
            if time.monotonic_ns() >= deadline_ns-30_000_000_000:
                raise RuntimeError("Paired exposure must not be truncated by the wall budget")
            indices = balanced_indices(groups, rng, settings["batch_size"], (update-1)*settings["batch_size"])
            batch_digest.update(indices.astype("<i8", copy=False).tobytes())
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
        if not updates or trace[-1]["update"] != updates:
            raise ValueError("Require a completed terminal validation point")
        counters = [int(v["step"].item()) for v in optimizer.state.values()]
        if len(counters) != 4 or any(v != updates for v in counters):
            raise ValueError("Require four actual residual Adam counters")
        if (module_sha(native.policy) != teacher_sha or module_sha(student.encoder) != encoder_sha
                or module_sha(student.decoder) != decoder_sha):
            raise ValueError("Frozen teacher, human encoder or shared decoder changed")
        from upper_mps_forecast import audit_cpu_tensors, cpu_tree
        common = dict(schema="q1_f0_proprio_residual_v1", interface="f0_plus_history", arm=arm,
            current_statistics=payload["current_statistics"], provenance=provenance,
            frozen_teacher_policy_sha256=teacher_sha, decoder_sha256=decoder_sha,
            base_encoder_sha256=encoder_sha, base_encoder=cpu_state(student.encoder),
            human_real_future_frames=0, history_frames=10, history_frame_dim=226,
            robot_history_features=930, adapter_hidden_width=128,
            residual_rule="z+(1-abs(z))*tanh(adapter(history))",
            actual_updates=updates, actual_adam_steps=counters, initial_adapter_sha256=adapter_sha,
            selected_update=best["update"], selected_validation_risk=best["risk"],
            selected_fixed_anchor_risks=best["anchors"], selection_split="validation",
            selection_rule="expanded minimum with both fixed anchors no worse than update0",
            test_used_for_selection=False, normalization_refitted=False,
            original_training_restarted=False, user_accepted_checkpoint=None)
        for name, value in (("selected.pt", dict(common, adapter=best["state"])),
                ("last.pt", dict(common, adapter=cpu_state(student.adapter),
                    optimizer=cpu_tree(torch, optimizer.state_dict())))):
            audit_cpu_tensors(torch, value)
            with (folder/name).open("xb") as stream:
                torch.save(value, stream)
            audit_cpu_tensors(torch, torch.load(folder/name, map_location="cpu", weights_only=True))
        result = {k:v for k,v in common.items() if k != "base_encoder"}
        result.update(status="TRAINED_PENDING_CLOSED_LOOP", initial_validation_risk=trace[0]["validation_risk"],
            selected_adapter_is_zero=zero_adapter(best["state"]), validation_improved=best["update"] != 0,
            final_validation_risk=trace[-1]["validation_risk"], initial_fixed_anchor_risks=initial_anchors,
            batch_sha256=batch_digest.hexdigest(), source_sample_counts=exposures,
            elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
            trainable_parameters=sum(p.numel() for p in student.trainable_parameters()),
            input_arrays_read=["f0", "history", "teacher_action", "clip_id"],
            natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None)
        save_json(folder/"fit_result.json", result)
        runs.append(result)
        del student, optimizer
    if (runs[0]["batch_sha256"] != runs[1]["batch_sha256"]
            or runs[0]["source_sample_counts"] != runs[1]["source_sample_counts"]
            or runs[0]["initial_validation_risk"] != runs[1]["initial_validation_risk"]
            or runs[0]["initial_fixed_anchor_risks"] != runs[1]["initial_fixed_anchor_risks"]):
        raise ValueError("Residual arms do not have identical initialization and exposure")
    result = dict(status="PAIRED_RESIDUAL_TRAINED_PENDING_CLOSED_LOOP", arms=runs,
        matched_initial_actions=True, matched_batch_order_and_exposure=True,
        data_coverage={s:dict(records=len(d["clip_id"]), clips=len(set(d["clip_id"])),
            groups=len(cells(d, sources))) for s,d in data.items()},
        source_preprocessing="offline_unverified", human_real_future_frames=0,
        test_sources_read=False, new_physical_steps=0, new_rendered_frames=0,
        new_optimizer_updates=sum(v["actual_updates"] for v in runs),
        natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None,
        user_accepted_checkpoint=None)
    save_json(output/"fit_result.json", result)
    return result
