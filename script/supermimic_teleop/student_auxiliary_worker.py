"""CUDA fitting and CPU labels from saved physical states; no new simulation."""

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


def load_origin(root, *, aggregate, observation, present):
    from student_auxiliary_data import transition_records
    collection, receipt = read(root/"collection_result.json"), read(root/"dataset_receipt.json")
    traces, terminal, sources = {}, {}, [identity(root/"collection_result.json"), identity(root/"dataset_receipt.json")]
    for row in collection["clips"]:
        record = row["trace"]
        require(identity(record["path"]) == record, "Saved transition trace changed")
        with np.load(record["path"], allow_pickle=False) as arrays:
            traces[row["clip_id"]] = dict(arrays)
        terminal[row["clip_id"]] = row["failure_frame"]
        sources.append(record)
    values, checks = {}, {}
    for split in ("train", "validation"):
        record = receipt["files"][split]
        require(identity(record["path"]) == record, "Preserved transition dataset changed")
        with np.load(record["path"], allow_pickle=False) as arrays:
            data = dict(arrays)
        values[split], checks[split] = transition_records(data, traces, terminal, observation, present, aggregate=aggregate)
        sources.append(record)
    return values, checks, sources


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu
        and identity(args.auxiliary_plan)["sha256"] == args.auxiliary_plan_sha256
        and time.monotonic_ns() < args.deadline_ns-60_000_000_000, "Admitted GPU, plan or deadline differs")
    runtime, config, plan = read(args.runtime_config), read(args.config), read(args.plan)
    sys.path[:0] = [str(Path(plan["source"])/"src"), str(Path(runtime["adapted_source"])/"src"),
                   str(HERE.parent/"morphmimic_q1")]
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from bootstrap import install
    from inference import load_policy
    from interim_review import load_physical_robot
    from morphmimic.model.rewards_torch import analytic_obs
    from q1_native_review import native_state
    from student_auxiliary_fit import fit_pair
    from student_fit import module_sha

    audit = read(args.output/"checkpoint_audit.json")
    teacher = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    spec = install("q1")["robot"]
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
    original, old_check, old_sources = load_origin(args.training_root, aggregate=False, observation=observation, present=present.numpy())
    mixed, mixed_check, mixed_sources = load_origin(args.aggregate_root, aggregate=True, observation=observation, present=present.numpy())
    save(args.output/"transition_label_audit.json", dict(status="PASSED", original=old_check, mixed=mixed_check,
        next_state_target="actual observation at frame+1 under recorded executed action",
        encoder_inputs=["causal f0/history", "unchanged robot history"], new_physical_transitions=0,
        selected_teacher=teacher, physical_model=baseline, sources=old_sources+mixed_sources))
    provenance = dict(teacher=teacher, warm_start=identity(args.initial_r2_root/"selected.pt"),
        original_r1_fit=identity(args.training_root/"fit_result.json"), aggregate_fit=identity(args.aggregate_root/"fit_result.json"),
        transition_label_audit=identity(args.output/"transition_label_audit.json"), auxiliary_plan=identity(args.auxiliary_plan))
    result = fit_pair(native, [original, mixed], payload, read(args.auxiliary_plan)["fit"], args.output, args.deadline_ns, provenance)
    require(all(identity(row["path"]) == row for row in old_sources+mixed_sources), "Training altered preserved transitions")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime-config", "config", "plan", "output", "training-root", "aggregate-root", "initial-r2-root", "auxiliary-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("auxiliary-plan-sha256", "expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    run(parser.parse_args())
