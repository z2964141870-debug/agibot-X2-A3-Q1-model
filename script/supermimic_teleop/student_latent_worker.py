"""Label saved states with the frozen future teacher; train only a causal encoder."""

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


def attach_goals(values, root):
    for split, records in values.items():
        with np.load(root/(split+"_data.npz"), allow_pickle=False) as arrays:
            data = dict(arrays)
        lookup = {(int(clip), int(frame)):i for i, (clip, frame) in
                  enumerate(zip(data["clip_id"], data["frame"]))}
        require(len(lookup) == len(data["frame"]), "Duplicated preserved source/frame label")
        ids = [lookup[(int(clip), int(frame))] for clip, frame in zip(records["clip_id"], records["frame"])]
        require(len(ids) == len(data["frame"]), "Control reuse forbids changing the original transition rows")
        for key in ("f0", "history", "teacher_action", "clip_id", "frame"):
            require(np.array_equal(records[key], data[key][ids]), "Teacher labels differ from registered control rows")
        records["full_goal"] = data["full_goal"][ids].copy()
    return values


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu
        and identity(args.latent_plan)["sha256"] == args.latent_plan_sha256
        and time.monotonic_ns() < args.deadline_ns-60_000_000_000, "Admitted GPU, plan or deadline differs")
    runtime, config, plan = read(args.runtime_config), read(args.config), read(args.plan)
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
    from student_latent_fit import fit_latent, latent_records

    audit = read(args.output/"checkpoint_audit.json")
    teacher = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    spec, analytic_obs = native_contracts()
    native, metadata = load_policy(Path(teacher["path"]), spec, plan["code_commit"], "cuda:0", expected_plan=plan)
    payload = torch.load(args.initial_r2_root/"selected.pt", map_location="cpu", weights_only=True)
    require(metadata.get("iteration") == 32000
        and module_sha(native.policy) == payload["frozen_teacher_policy_sha256"]
        and module_sha(native.policy.decoder) == payload["decoder_sha256"], "Frozen teacher differs")
    baseline = read(Path(config["primebot_review"])/"checkpoint_audit.json")["final"]["checkpoint"]
    require(identity(baseline["path"]) == baseline, "Physical model identity changed")
    robot, _ = load_physical_robot(baseline["path"])
    default, present = spec.translator.to_canonical(torch.zeros(22)), native.present.cpu()
    norm = native.norms["obs_norm"]
    mean, scale = norm.mean.cpu(), norm.var.sqrt().cpu()+norm.eps
    def observation(states):
        with torch.inference_mode():
            converted = native_state(robot, torch.tensor(states, dtype=torch.float32), spec)
            obs = analytic_obs(converted, torch.zeros(29), default, converted[:, 240:], spec.imu_slot)
            mask = torch.cat((present, present, torch.ones(6, dtype=torch.bool), present))
            return ((torch.where(mask, obs, 0.)-mean)/scale).numpy()
    origins, checks, sources = [], {}, []
    prior_labels = read(args.auxiliary_root/"transition_label_audit.json")
    for label, root, aggregate in (("original", args.training_root, False), ("mixed", args.aggregate_root, True)):
        values, transition_check, preserved = load_origin(root, aggregate=aggregate,
            observation=observation, present=present.numpy())
        require(transition_check == prior_labels[label], "Reused action control has another transition-row set")
        values = attach_goals(values, root)
        labeled, checks[label] = {}, {}
        for split, data in values.items():
            labeled[split], checks[label][split] = latent_records(data, native)
        origins.append(labeled)
        sources.extend(preserved)
    save(args.output/"teacher_latent_audit.json", dict(status="PASSED", origins=checks,
        source_arrays=sources, source_transitions_reused=True, new_physical_transitions=0,
        teacher_future_is_offline_label_only=True, selected_teacher=teacher))
    provenance = dict(teacher=teacher, warm_start=identity(args.initial_r2_root/"selected.pt"),
        original_r1_fit=identity(args.training_root/"fit_result.json"),
        aggregate_fit=identity(args.aggregate_root/"fit_result.json"),
        teacher_latent_audit=identity(args.output/"teacher_latent_audit.json"),
        latent_plan=identity(args.latent_plan),
        action_control=identity(args.auxiliary_root/"action_only/fit_result.json"),
        control_trace=identity(args.auxiliary_root/"action_only/training_trace.json"))
    result = fit_latent(native, origins, payload, read(args.latent_plan)["fit"],
        read(args.auxiliary_root/"action_only/fit_result.json"),
        read(args.auxiliary_root/"action_only/training_trace.json"),
        args.output, args.deadline_ns, provenance)
    require(all(identity(row["path"]) == row for row in sources), "Preserved data changed during latent fit")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime-config", "config", "plan", "output", "training-root", "aggregate-root",
                 "initial-r2-root", "auxiliary-root", "latent-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("latent-plan-sha256", "expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    run(parser.parse_args())
