"""Calibrate the actual frozen32k WM on saved behavior; no new simulation or fit."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require, save
from delivery import read
from paused_runtime import assert_owner
from wm_probe_data import nominal_context, physical_errors, selected_transitions, summarize


def tensor_state_sha(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_world_model(path, spec, plan):
    import torch
    from inference import validate_metadata
    from morphmimic.model.state import StateNormalizer
    from morphmimic.model.world_model import WorldModel, WorldModelCfg
    from morphmimic.robots.morphology import morphology_vector

    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = validate_metadata(value, "q1", plan["code_commit"])
    require(metadata["iteration"] == 32000 and value["native_q1_contract"]["plan"] == plan,
        "Only the original audited32k native WM is admitted")
    cfg, weights = value["cfg"], value["wm"]
    require(all(torch.isfinite(t).all() for t in weights.values()), "Nonfinite frozen WM tensor")
    expected_presence = torch.tensor([v is not None for v in spec.translator.perm])
    require(torch.equal(weights["joint_presence"], expected_presence)
        and torch.equal(weights["morph"], value["morph"]["wm"])
        and torch.equal(weights["morph"], morphology_vector("q1"))
        and weights["ctx_scale"].shape == (62,) and weights["normalizer.std"].shape == (268,)
        and (weights["normalizer.std"] > 0).all() and (weights["height_std"] > 0).all(),
        "Saved WM morphology, presence, contexts or statistics differ")
    model = WorldModel(WorldModelCfg(hidden=tuple(cfg["wm_hidden"]), act=cfg["act"],
        termination=cfg["termination"], layer_norm=cfg["wm_layer_norm"], bf16=cfg["bf16"],
        frame=cfg["wm_frame"], residual_scale=cfg["residual_scale"], morph_dim=len(weights["morph"])),
        StateNormalizer(), weights["joint_limits"], weights["default_joint_pos"], imu_idx=spec.imu_slot)
    require(type(model).__name__ == "NativeWorldModel", "Native step override is required")
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False).eval()
    require(tensor_state_sha(model.state_dict()) == tensor_state_sha(weights), "Strict WM tensor copy differs")
    frozen = {name:tensor_state_sha(value[name]) for name in
        ("wm", "policy", "obs_norm", "goal_norm", "human_norm", "heading_norm")}
    return model, metadata, frozen, value


def samples(root, *, aggregate, quota):
    collection, receipt = read(root/"collection_result.json"), read(root/"dataset_receipt.json")
    require(collection.get("status") == "PASSED" and collection.get("test_sources_read") is False
        and receipt.get("test_sources_read") is False, "Use only original owned train/validation states")
    traces, terminal = {}, {}
    records = [identity(root/"collection_result.json"), identity(root/"dataset_receipt.json")]
    for clip in collection["clips"]:
        record = clip["trace"]
        require(identity(record["path"]) == record, "Saved physical trace changed")
        with np.load(record["path"], allow_pickle=False) as arrays:
            traces[clip["clip_id"]] = dict(arrays)
        terminal[clip["clip_id"]] = clip["failure_frame"]
        records.append(record)
    output, checks = {}, {}
    for split in ("train", "validation"):
        record = receipt["files"][split]
        require(identity(record["path"]) == record, "Preserved dataset differs")
        with np.load(record["path"], allow_pickle=False) as arrays:
            data = {key:arrays[key] for key in ("frame", "clip_id", "actual_state", "teacher_action")}
            if aggregate:
                data["behavior_action"] = arrays["behavior_action"]
        owned = {row["clip_id"] for row in collection["clips"] if row["split"] == split}
        require(set(map(int, np.unique(data["clip_id"]))) == owned
            and len(owned) == (12 if split == "train" else 6), "Probe source split ownership changed")
        output[split], checks[split] = selected_transitions(data, traces, terminal,
            aggregate=aggregate, quota=quota)
        records.append(record)
    return output, checks, records


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu
        and identity(args.probe_plan)["sha256"] == args.probe_plan_sha256,
        "Admitted device or probe plan differs")
    runtime, config, plan, settings = (read(p) for p in
        (args.runtime_config, args.config, args.plan, args.probe_plan))
    require(args.plan == Path(config["output"])/"e1024_train_plan.json", "Wrong original plan location")
    sys.path[:0] = [str(Path(plan["source"])/"src"), str(Path(runtime["adapted_source"])/"src"),
        str(HERE.parent/"morphmimic_q1")]
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from bootstrap import install
    spec = install("q1")["robot"]
    from morphmimic.model.state import canonicalize
    from interim_review import load_physical_robot
    from q1_native_review import native_state
    from wm_probe_math import StepCounter, gradient_probe

    audit = read(args.output/"checkpoint_audit.json")
    checkpoint = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    baseline = read(Path(config["primebot_review"])/"checkpoint_audit.json")["final"]["checkpoint"]
    require(identity(checkpoint["path"]) == checkpoint and identity(baseline["path"]) == baseline,
        "Saved physical or native model changed")
    model, metadata, frozen, payload = load_world_model(checkpoint["path"], spec, plan)
    robot, physical_settings = load_physical_robot(baseline["path"])
    require(all(identity(r["path"]) == r for r in settings["context_sources"]),
        "Nominal context source identities differ")
    context, provenance = nominal_context(robot, spec, physical_settings)
    origins, checks, sources = {}, {}, [checkpoint, baseline]+settings["context_sources"]
    for name, root in (("original", args.training_root), ("mixed", args.aggregate_root)):
        origins[name], checks[name], records = samples(root, aggregate=name == "mixed", quota=settings["samples_per_group"])
        sources.extend(records)
    present = model.joint_presence.to("cuda:0")
    model = model.to("cuda:0")
    step = StepCounter(model, args.deadline_ns, settings["maximum_forward_calls"])
    metrics, gradients, selected_rows = [], [], []
    ctx = torch.as_tensor(context, device="cuda:0").unsqueeze(0)
    for origin in ("original", "mixed"):
        for split in ("train", "validation"):
            batch = origins[origin][split]
            state, following = (native_state(robot, torch.as_tensor(np.stack([r[key] for r in batch]),
                dtype=torch.float32), spec).detach().to("cuda:0") for key in ("current", "following"))
            action = torch.as_tensor(np.stack([r["executed"] for r in batch]), device="cuda:0", dtype=torch.float32)
            require(torch.equal(action[:, ~present], torch.zeros_like(action[:, ~present]))
                and torch.allclose(following[:, 240:], action, atol=3e-5, rtol=0.),
                "Saved next command memory does not match the actual no-delay executed action")
            with torch.no_grad():
                predicted = step(state, action, ctx.expand(len(batch), -1))
            wm = physical_errors(predicted.detach().cpu().numpy(), following.cpu().numpy(), present.cpu().numpy())
            identity_errors = physical_errors(state.cpu().numpy(), following.cpu().numpy(), present.cpu().numpy())
            for i, row in enumerate(batch):
                key = dict(origin=origin, split=split, clip_id=row["clip_id"], frame=row["frame"], dataset_row=row["dataset_row"])
                metrics.append(dict(key, wm={k:float(v[i]) for k, v in wm.items()},
                    identity={k:float(v[i]) for k, v in identity_errors.items()}))
                selected_rows.append(key)
            anchor = model.frame(state[:1])._replace(pos_xy=model.frame(state[:1]).pos_xy.detach(),
                quat=model.frame(state[:1]).quat.detach())
            def physical_features(value):
                local = canonicalize(value, anchor)/model.normalizer.std
                return local[..., model.feature_presence]
            gradient = gradient_probe(step, state[:1], action[:1], ctx, present, physical_features,
                epsilons=settings["finite_difference_epsilons"],
                relative_tolerance=settings["gradient_relative_tolerance"],
                absolute_tolerance=settings["gradient_absolute_tolerance"])
            gradients.append(dict(origin=origin, split=split, clip_id=batch[0]["clip_id"], frame=batch[0]["frame"], **gradient))
    require(len(metrics) <= 256 and step.calls <= 512 and len(gradients) == 4, "Probe bounds differ")
    joint_response = {int(joint):max(abs(next(r["autograd"] for r in g["present_joints"]
        if r["joint"] == joint)) for g in gradients) for joint in present.nonzero().flatten().tolist()}
    physical_gradient_passed = all(g["finite_difference_passed"] for g in gradients) and all(
        value > settings["minimum_joint_gradient_magnitude"] for value in joint_response.values())
    summary = summarize(metrics, improvement_ratio=settings["validation_mean_ratio_max"],
        maximum_source_ratio=settings["validation_per_source_ratio_max"])
    require(tensor_state_sha(model.state_dict()) == frozen["wm"] and all(
        tensor_state_sha(payload[name]) == digest for name, digest in frozen.items())
        and all(p.grad is None and not p.requires_grad for p in model.parameters()), "Frozen WM/teacher tensors changed")
    require(all(identity(row["path"]) == row for row in sources), "Saved transitions or model changed")
    torch.cuda.synchronize()
    result = dict(status="WM_PROBE_COMPLETED", diagnostic_screen="PASSED" if physical_gradient_passed
        and summary["calibration_screen_passed"] else "FAILED", physical_gradient_screen_passed=physical_gradient_passed,
        calibration=summary, gradients=gradients, sample_checks=checks, selected_rows=selected_rows,
        per_joint_maximum_physical_gradient=joint_response,
        sampled_transition_count=len(metrics), physical_prediction_metrics=metrics,
        cuda_wm_step_forward_calls=step.calls, cuda_backward_calls=4, cuda_primitive_kernel_count=None,
        model_metadata=metadata, context_provenance=provenance, context_values=context.tolist(),
        frozen_tensor_sha256=frozen, source_inputs=sources, source_integrity="PASSED",
        arrays_inspected_for_identity_and_selection=True, test_sources_read=False,
        new_physical_transitions=0, actual_optimizer_updates=0, hardware_latency_ms=None,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
    save(args.output/"probe_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime-config", "config", "plan", "output", "training-root", "aggregate-root", "probe-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("probe-plan-sha256", "expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    run(parser.parse_args())
