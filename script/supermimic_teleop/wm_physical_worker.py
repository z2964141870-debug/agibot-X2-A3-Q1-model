"""Fit one causal encoder using only the preserved data and frozen FP32 WM."""

import argparse
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


def attach_states(values, root, convert, context):
    audits = {}
    for split, records in values.items():
        with np.load(root/(split+"_data.npz"), allow_pickle=False) as arrays:
            data = dict(arrays)
        lookup = {(int(c), int(f)):i for i, (c, f) in enumerate(zip(data["clip_id"], data["frame"]))}
        require(len(lookup) == len(data["frame"]), "Duplicate saved physical source/frame")
        ids = [lookup[(int(c), int(f))] for c, f in zip(records["clip_id"], records["frame"])]
        for key in ("f0", "history", "teacher_action", "clip_id", "frame"):
            require(np.array_equal(records[key], data[key][ids]), "Physical labels differ from the exact control rows")
        state = np.asarray(convert(data["actual_state"][ids]), dtype=np.float32)
        require(state.shape == (len(ids), 269) and np.isfinite(state).all(), "Invalid saved native physical state")
        records["physical_state"] = state.copy()
        records["wm_context"] = np.broadcast_to(context, (len(ids), 62)).copy()
        audits[split] = dict(records=len(ids), source_frame_join_exact=True,
            student_inputs=["f0", "history"], physical_state_is_offline_loss_input_only=True)
    return values, audits


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu
        and identity(args.physical_plan)["sha256"] == args.physical_plan_sha256
        and time.monotonic_ns() < args.deadline_ns-60_000_000_000, "Admitted GPU, plan or deadline differs")
    runtime, config, plan, settings = (read(p) for p in
        (args.runtime_config, args.config, args.plan, args.physical_plan))
    sys.path[:0] = [str(Path(plan["source"])/"src"), str(Path(runtime["adapted_source"])/"src"),
                   str(HERE.parent/"morphmimic_q1")]
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from inference import load_policy
    from interim_review import load_physical_robot
    from q1_native_review import native_state
    from student_auxiliary_worker import load_origin, native_contracts
    from student_fit import module_sha
    from wm_physical_fit import fit_physical
    from wm_physical_loss import PhysicalLoss, matched_settings
    from wm_precision_math import fp32_shadow
    from wm_probe_data import nominal_context
    from wm_probe_worker import load_world_model, tensor_state_sha

    spec, analytic_obs = native_contracts()
    from morphmimic.model.state import canonicalize
    audit = read(args.output/"checkpoint_audit.json")
    teacher = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    native, metadata = load_policy(Path(teacher["path"]), spec, plan["code_commit"], "cuda:0", expected_plan=plan)
    payload = torch.load(args.initial_r2_root/"selected.pt", map_location="cpu", weights_only=True)
    require(metadata.get("iteration") == 32000
        and module_sha(native.policy) == payload["frozen_teacher_policy_sha256"]
        and module_sha(native.policy.decoder) == payload["decoder_sha256"], "Frozen teacher differs")
    wm, _, frozen, wm_payload = load_world_model(teacher["path"], spec, plan)
    shadow = fp32_shadow(wm).to("cuda:0")
    del wm, wm_payload
    precision = read(args.precision_root/"precision_result.json")
    require(frozen == precision["frozen_tensor_sha256"]
        and tensor_state_sha(shadow.state_dict()) == frozen["wm"], "FP32 WM differs from the passed diagnostic")
    baseline = read(Path(config["primebot_review"])/"checkpoint_audit.json")["final"]["checkpoint"]
    require(identity(baseline["path"]) == baseline, "Physical model identity changed")
    robot, physical_settings = load_physical_robot(baseline["path"])
    probe_plan = read(args.probe_plan)
    require(all(identity(r["path"]) == r for r in probe_plan["context_sources"]), "Nominal context sources changed")
    context, context_audit = nominal_context(robot, spec, physical_settings)
    default, present = spec.translator.to_canonical(torch.zeros(22)), native.present.cpu()
    norm = native.norms["obs_norm"]
    mean, scale = norm.mean.cpu(), norm.var.sqrt().cpu()+norm.eps
    def convert(states):
        with torch.inference_mode():
            return native_state(robot, torch.tensor(states, dtype=torch.float32), spec).cpu().numpy()
    def observation(states):
        with torch.inference_mode():
            converted = torch.tensor(convert(states))
            obs = analytic_obs(converted, torch.zeros(29), default, converted[:, 240:], spec.imu_slot)
            mask = torch.cat((present, present, torch.ones(6, dtype=torch.bool), present))
            return ((torch.where(mask, obs, 0.)-mean)/scale).numpy()
    origins, checks, sources = [], {}, []
    prior_labels = read(args.auxiliary_root/"transition_label_audit.json")
    for label, root, aggregate in (("original", args.training_root, False), ("mixed", args.aggregate_root, True)):
        values, row_checks, preserved = load_origin(root, aggregate=aggregate,
            observation=observation, present=present.numpy())
        require(row_checks == prior_labels[label], "Physical fit changes the preserved control transition rows")
        values, checks[label] = attach_states(values, root, convert, context)
        origins.append(values)
        sources.extend(preserved)
    save(args.output/"physical_label_audit.json", dict(status="PASSED", origins=checks,
        transition_masks_match_control=True, context_provenance=context_audit, context_values=context.tolist(),
        targets="same saved state and same FP32 WM under saved teacher action",
        new_physical_transitions=0, test_sources_read=False, sources=sources))
    physical = PhysicalLoss(shadow, canonicalize, args.deadline_ns, settings["maximum_wm_step_calls"])
    provenance = dict(teacher=teacher, warm_start=identity(args.initial_r2_root/"selected.pt"),
        physical_label_audit=identity(args.output/"physical_label_audit.json"),
        physical_plan=identity(args.physical_plan), precision=identity(args.precision_root/"precision_result.json"),
        action_control=identity(args.auxiliary_root/"action_only/fit_result.json"),
        control_trace=identity(args.auxiliary_root/"action_only/training_trace.json"))
    result = fit_physical(native, physical, origins, payload,
        matched_settings(settings, read(HERE/"student_auxiliary_plan_20261107.json")),
        read(args.auxiliary_root/"action_only/fit_result.json"),
        read(args.auxiliary_root/"action_only/training_trace.json"),
        args.output, args.deadline_ns, provenance)
    require(all(identity(row["path"]) == row for row in sources), "Preserved source arrays changed")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime-config", "config", "plan", "output", "training-root", "aggregate-root",
                 "initial-r2-root", "auxiliary-root", "probe-plan", "precision-root", "physical-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("physical-plan-sha256", "expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    run(parser.parse_args())
