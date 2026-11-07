"""Frozen native BF16 versus identical FP32 shadow on the saved probe rows."""

import argparse
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require, save
from delivery import read
from paused_runtime import assert_owner
from wm_probe_data import nominal_context, physical_errors, summarize
from wm_probe_worker import load_world_model, samples, tensor_state_sha


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu
        and identity(args.precision_plan)["sha256"] == args.precision_plan_sha256,
        "Admitted precision device/plan differs")
    runtime, config, plan, settings, registered = (read(p) for p in
        (args.runtime_config, args.config, args.plan, args.probe_plan, args.precision_plan))
    require(identity(args.parent_root/"probe_result.json") == registered["parent_result"],
        "Use the original settled native diagnostic")
    parent = read(args.parent_root/"probe_result.json")
    sys.path[:0] = [str(Path(plan["source"])/"src"), str(Path(runtime["adapted_source"])/"src"),
        str(HERE.parent/"morphmimic_q1")]
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One admitted CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from bootstrap import install
    spec = install("q1")["robot"]
    from morphmimic.model.state import canonicalize
    from interim_review import load_physical_robot
    from q1_native_review import native_state
    from wm_probe_math import StepCounter, gradient_probe
    from wm_precision_math import anchor_vjp, fp32_shadow, native_reproduction, prediction_preservation

    audit = read(args.output/"checkpoint_audit.json")
    checkpoint = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    baseline = read(Path(config["primebot_review"])/"checkpoint_audit.json")["final"]["checkpoint"]
    require(all(identity(r["path"]) == r for r in (checkpoint, baseline)), "Model input changed")
    native, metadata, frozen, payload = load_world_model(checkpoint["path"], spec, plan)
    require(frozen == parent["frozen_tensor_sha256"] and payload["cfg"]["bf16"] is True,
        "The native WM must retain the saved configuration and teacher tensors")
    shadow = fp32_shadow(native)
    require(tensor_state_sha(shadow.state_dict()) == frozen["wm"], "FP32 shadow must use identical tensors")
    robot, physical_settings = load_physical_robot(baseline["path"])
    require(all(identity(r["path"]) == r for r in settings["context_sources"]), "Context sources differ")
    context, provenance = nominal_context(robot, spec, physical_settings)
    require(context.tolist() == parent["context_values"], "Native context reproduction differs")
    origins, sample_checks, sources = {}, {}, [checkpoint, baseline]+settings["context_sources"]
    for name, root in (("original", args.training_root), ("mixed", args.aggregate_root)):
        origins[name], sample_checks[name], records = samples(root, aggregate=name == "mixed",
            quota=settings["samples_per_group"])
        sources.extend(records)
    require(sources == parent["source_inputs"] and sample_checks == parent["sample_checks"],
        "Reconstruct the same source-owned native rows, without test data")
    native, shadow = native.to("cuda:0"), shadow.to("cuda:0")
    present = native.joint_presence
    ctx = torch.as_tensor(context, device="cuda:0").unsqueeze(0)
    native_step = StepCounter(native, args.deadline_ns, maximum=8)
    shadow_step = StepCounter(shadow, args.deadline_ns, maximum=368)
    metrics = {"native":[], "shadow":[]}
    selected, gradients, reproduced = [], [], []
    for origin in ("original", "mixed"):
        for split in ("train", "validation"):
            batch = origins[origin][split]
            state, following = (native_state(robot, torch.as_tensor(np.stack([r[key] for r in batch]),
                dtype=torch.float32), spec).detach().to("cuda:0") for key in ("current", "following"))
            action = torch.as_tensor(np.stack([r["executed"] for r in batch]), device="cuda:0", dtype=torch.float32)
            require(torch.equal(action[:, ~present], torch.zeros_like(action[:, ~present]))
                and torch.allclose(following[:, 240:], action, atol=3e-5, rtol=0.), "Executed command memory differs")
            keys = [dict(origin=origin, split=split, clip_id=r["clip_id"], frame=r["frame"],
                dataset_row=r["dataset_row"]) for r in batch]
            selected.extend(keys)
            target = following.cpu().numpy()
            hold = physical_errors(state.cpu().numpy(), target, present.cpu().numpy())
            for label, step in (("native", native_step), ("shadow", shadow_step)):
                with torch.no_grad():
                    predicted = step(state, action, ctx.expand(len(batch), -1))
                errors = physical_errors(predicted.cpu().numpy(), target, present.cpu().numpy())
                metrics[label].extend(dict(key, wm={k:float(v[i]) for k, v in errors.items()},
                    identity={k:float(v[i]) for k, v in hold.items()}) for i, key in enumerate(keys))
            anchor = native.frame(state[:1])
            anchor = anchor._replace(pos_xy=anchor.pos_xy.detach(), quat=anchor.quat.detach())
            def physical_features(value):
                return (canonicalize(value, anchor)/native.normalizer.std)[..., native.feature_presence]
            saved = next(g for g in parent["gradients"] if g["origin"] == origin and g["split"] == split)
            require(saved["clip_id"] == batch[0]["clip_id"] and saved["frame"] == batch[0]["frame"],
                "The precision VJP anchor differs from its saved native anchor")
            actual = anchor_vjp(native_step, state[:1], action[:1], ctx, present, physical_features)
            reproduced.append(dict(origin=origin, split=split, actual=actual,
                **native_reproduction(actual, saved)))
            check = gradient_probe(shadow_step, state[:1], action[:1], ctx, present, physical_features,
                epsilons=registered["finite_difference_epsilons"],
                relative_tolerance=registered["gradient_relative_tolerance"],
                absolute_tolerance=registered["gradient_absolute_tolerance"])
            gradients.append(dict(origin=origin, split=split, clip_id=saved["clip_id"], frame=saved["frame"], **check))
    require(selected == parent["selected_rows"] and len(selected) == registered["same_selected_rows"] == 256,
        "Precision comparison must retain the exact original256 rows and order")
    calls = native_step.calls+shadow_step.calls
    require(calls == registered["expected_wm_step_calls"] == 376 and calls <= registered["maximum_wm_step_calls"],
        "The precision forward budget differs")
    summaries = {name:summarize(rows, improvement_ratio=registered["validation_mean_ratio_max_against_identity"],
        maximum_source_ratio=registered["validation_per_source_ratio_max_against_identity"])
        for name, rows in metrics.items()}
    preservation = prediction_preservation(summaries["native"], summaries["shadow"])
    response = {j:max(abs(next(r["autograd"] for r in g["present_joints"] if r["joint"] == j))
        for g in gradients) for j in present.nonzero().flatten().tolist()}
    gradient_passed = all(g["finite_difference_passed"] for g in gradients) and all(
        v > registered["minimum_joint_gradient_magnitude"] for v in response.values())
    native_predictions_identical = metrics["native"] == parent["physical_prediction_metrics"]
    reproduction_passed = all(r["passed"] for r in reproduced) and native_predictions_identical
    for model in (native, shadow):
        require(tensor_state_sha(model.state_dict()) == frozen["wm"] and all(
            p.grad is None and not p.requires_grad for p in model.parameters()), "Frozen model tensors changed")
    require(native.cfg.bf16 is True and shadow.cfg.bf16 is False
        and all(tensor_state_sha(payload[name]) == digest for name, digest in frozen.items())
        and all(identity(r["path"]) == r for r in sources)
        and identity(args.parent_root/"probe_result.json") == registered["parent_result"],
        "Native policy, precision configuration, saved inputs or original failed evidence changed")
    torch.cuda.synchronize()
    result = dict(status="WM_PRECISION_COMPLETED", diagnostic_screen="PASSED" if reproduction_passed
        and gradient_passed and summaries["shadow"]["calibration_screen_passed"] and preservation["passed"] else "FAILED",
        native_reproduction_passed=reproduction_passed, native_predictions_identical_to_saved=native_predictions_identical,
        native_anchor_reproduction=reproduced, native_original_diagnostic_screen=parent["diagnostic_screen"],
        native_saved_finite_differences_reused=True, shadow_physical_gradient_screen_passed=gradient_passed,
        shadow_gradients=gradients, shadow_per_joint_maximum_physical_gradient=response,
        calibration=summaries, shadow_prediction_preservation=preservation,
        physical_prediction_metrics=metrics, selected_rows=selected, sampled_transition_count=len(selected),
        native_cuda_wm_step_forward_calls=native_step.calls, shadow_cuda_wm_step_forward_calls=shadow_step.calls,
        cuda_wm_step_forward_calls=calls, cuda_backward_calls=8, cuda_primitive_kernel_count=None,
        native_bf16=native.cfg.bf16, shadow_bf16=shadow.cfg.bf16, tf32_enabled=False,
        model_metadata=metadata, frozen_tensor_sha256=frozen, source_inputs=sources,
        parent_result=registered["parent_result"], context_provenance=provenance, source_integrity="PASSED",
        test_sources_read=False, new_physical_transitions=0, actual_optimizer_updates=0,
        hardware_latency_ms=None, natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
    save(args.output/"precision_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime-config", "config", "plan", "output", "training-root", "aggregate-root", "probe-plan",
                 "precision-plan", "parent-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("precision-plan-sha256", "expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    run(parser.parse_args())
