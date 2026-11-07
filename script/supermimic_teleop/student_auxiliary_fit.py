"""Paired action-only versus behavior-conditioned next-state auxiliary fitting."""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from student_aggregate_fit import balanced_indices, checked_splits
from student_causal_fit import source_groups, validation
from student_fit import cpu_state, module_sha, save_json
from student_models import action_distillation_loss, current_student


class TransitionHead(nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent+50+22, 128), nn.ELU(), nn.Linear(128, 50))

    def forward(self, latent, current, executed):
        return current+self.net(torch.cat((latent, current, executed), -1))


def fit_pair(native, origins, payload, settings, output, deadline_ns, provenance):
    if len(origins) != 2:
        raise ValueError("Preserve both original and mixed state origins")
    checked_splits(*origins)
    for data in origins:
        for split, value in data.items():
            n = len(value["clip_id"])
            for key, shape in (("current_state", (n, 50)), ("next_state", (n, 50)),
                               ("executed_action", (n, 22))):
                if value[key].shape != shape or not np.isfinite(value[key]).all():
                    raise ValueError("Missing finite behavior-conditioned training labels")
    output = Path(output)
    if any((output/name).exists() for name in ("action_only", "next_state", "paired_fit_result.json")):
        raise FileExistsError("Paired training outputs are single-use")
    before_teacher = module_sha(native.policy)
    before_decoder = module_sha(native.policy.decoder)
    if (before_teacher != payload["frozen_teacher_policy_sha256"]
            or before_decoder != payload["decoder_sha256"]):
        raise ValueError("Warm start has another teacher or decoder")
    results, initial_encoder, batch_digest, exposures = {}, None, None, None
    for name, coefficient in (("action_only", 0.), ("next_state", settings["auxiliary_weight"])):
        folder = output/name
        folder.mkdir()
        torch.manual_seed(settings["seed"])
        rng = np.random.default_rng(settings["seed"])
        student = current_student(native, payload["current_statistics"]).train()
        student.encoder.load_state_dict(payload["encoder"], strict=True)
        device, present = next(student.parameters()).device, native.present
        encoder_sha = module_sha(student.encoder)
        if initial_encoder is not None and encoder_sha != initial_encoder:
            raise ValueError("Paired arms have different encoder initializations")
        initial_encoder = encoder_sha
        head = TransitionHead(native.policy.cfg.latent).to(device)
        parameters = list(student.trainable_parameters())
        optimizer = torch.optim.Adam(parameters, lr=settings["learning_rate"])
        head_optimizer = torch.optim.Adam(head.parameters(), lr=settings["learning_rate"]) if coefficient else None
        trace, best, updates = [], None, 0
        digest = hashlib.sha256()
        counts = {f"{origin}:{clip}":0 for origin, data in enumerate(origins)
                  for clip in sorted(set(data["train"]["clip_id"]))}
        started = time.monotonic_ns()

        def evaluate(step, loss=None, action_loss=None, state_loss=None, gradient=None):
            nonlocal best
            risks, details, state_risks = [], [], []
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
                            goal, current, following, executed = (torch.as_tensor(
                                data["validation"][key][indices], device=device) for key in
                                ("f0", "current_state", "next_state", "executed_action"))
                            error = (head(student.encoder(goal), current, executed)-following).square().mean()
                            total += float(error)*len(indices)
                        groups.append(total/len(ids))
                state_risks.append(float(np.mean(groups)))
                student.train()
            risk = float(np.mean(risks))
            if not np.isfinite(risk):
                raise ValueError("Nonfinite paired validation action risk")
            if best is None or risk < best["risk"]:
                best = dict(update=step, risk=risk, state=cpu_state(student.encoder))
            trace.append(dict(update=step, loss=loss, action_loss=action_loss,
                auxiliary_state_loss=state_loss, gradient_norm=gradient,
                validation_risk=risk, validation_origin_risks=risks, validation=details,
                validation_state_mse=float(np.mean(state_risks)), validation_state_origin_mse=state_risks,
                batch_identity_sha256=digest.hexdigest(), source_sample_counts=dict(counts),
                elapsed_seconds=(time.monotonic_ns()-started)/1e9))
            save_json(folder/"training_trace.json", trace)
            save_json(output/"progress.json", dict(arm=name, **trace[-1]))

        evaluate(0)
        for update in range(1, settings["maximum_updates"]+1):
            if time.monotonic_ns() >= deadline_ns-30_000_000_000:
                raise RuntimeError("Do not compare arms with unequal optimizer exposures")
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
                    ("f0", "history", "teacher_action", "current_state", "next_state", "executed_action")))
            goal, history, target, current, following, executed = (
                torch.cat([part[i] for part in tensors]) for i in range(6))
            optimizer.zero_grad(set_to_none=True)
            if head_optimizer is not None:
                head_optimizer.zero_grad(set_to_none=True)
            latent = student.encoder(goal)
            action = student.decoder(latent, history)
            action_loss = action_distillation_loss(action, target, present)
            state_loss = (head(latent, current, executed)-following).square().mean()
            loss = action_loss+coefficient*state_loss
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_norm_limit"],
                                                     error_if_nonfinite=True)
            optimizer.step()
            if head_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_norm_limit"],
                                               error_if_nonfinite=True)
                head_optimizer.step()
            updates = update
            if update % settings["validation_every_updates"] == 0 or update == settings["maximum_updates"]:
                evaluate(update, float(loss.detach().cpu()), float(action_loss.detach().cpu()),
                         float(state_loss.detach().cpu()), float(gradient.detach().cpu()))
        counters = [int(row["step"].item()) for row in optimizer.state.values()]
        auxiliary_counters = [] if head_optimizer is None else [
            int(row["step"].item()) for row in head_optimizer.state.values()]
        if (not counters or any(n != updates for n in counters+auxiliary_counters)
                or updates != settings["maximum_updates"]):
            raise ValueError("Paired Adam exposures differ from the registered bound")
        if batch_digest is not None and (digest.hexdigest() != batch_digest or counts != exposures):
            raise ValueError("Paired arms sampled different records or source exposures")
        batch_digest, exposures = digest.hexdigest(), counts
        if module_sha(native.policy) != before_teacher or module_sha(student.decoder) != before_decoder:
            raise ValueError("Frozen teacher or deployment decoder changed")
        from upper_mps_forecast import audit_cpu_tensors, cpu_tree
        common = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
            method="paired_behavior_conditioned_next_state_auxiliary", arm=name, provenance=provenance,
            current_statistics=payload["current_statistics"], human_real_future_frames=0,
            history_frames=10, history_frame_dim=226, decoder_sha256=before_decoder,
            frozen_teacher_policy_sha256=before_teacher, actual_updates=updates,
            actual_adam_steps=counters, auxiliary_adam_steps=auxiliary_counters,
            auxiliary_weight=coefficient, selected_update=best["update"],
            selected_validation_risk=best["risk"], selection_split="validation",
            selection_weighting="equal state origins then equal source clips",
            test_used_for_selection=False, auxiliary_head_deployed=False,
            original_training_restarted=False, user_accepted_checkpoint=None)
        for filename, values in (("selected.pt", dict(common, encoder=best["state"])),
                ("last.pt", dict(common, encoder=cpu_state(student.encoder),
                    optimizer=cpu_tree(torch, optimizer.state_dict()), auxiliary_head=cpu_state(head)))):
            audit_cpu_tensors(torch, values)
            with (folder/filename).open("xb") as stream:
                torch.save(values, stream)
            audit_cpu_tensors(torch, torch.load(folder/filename, map_location="cpu", weights_only=True))
        last_sha = module_sha(student.encoder)
        student.encoder.load_state_dict(best["state"], strict=True)
        result = dict(common, status="TRAINED_PENDING_CLOSED_LOOP", initial_encoder_sha256=initial_encoder,
            selected_encoder_sha256=module_sha(student.encoder), last_encoder_sha256=last_sha,
            batch_identity_sha256=batch_digest, source_sample_counts=counts,
            initial_validation_risk=trace[0]["validation_risk"], validation_improved=best["update"] != 0,
            new_physical_transitions=0, elapsed_fit_seconds=(time.monotonic_ns()-started)/1e9,
            normalization="frozen teacher observation normalizer; no future statistics fitting",
            state_prediction_scope="next observed state under recorded executed behavior, not hypothetical student action",
            hardware_latency_ms=None, natural_teleoperation_conclusion=None)
        save_json(folder/"fit_result.json", result)
        results[name] = result
        del student, head, optimizer, head_optimizer
    paired = dict(status="PAIRED_FIT_PASSED_PENDING_CLOSED_LOOP", arms=results,
        matched_initial_encoder=True, matched_batch_identity_sha256=batch_digest,
        matched_source_exposures=True, actual_optimizer_updates=sum(r["actual_updates"] for r in results.values()),
        auxiliary_property="training-only next proprioceptive state prediction with recorded behavior conditioning",
        deployment_architecture_unchanged=True, new_physical_transitions=0, human_real_future_frames=0,
        test_sources_read=False, user_accepted_checkpoint=None)
    save_json(output/"paired_fit_result.json", paired)
    return paired
