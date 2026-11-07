"""Owned CUDA R2 fit; reuse data and never construct a simulator or Trainer."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require
from delivery import read
from paused_runtime import assert_owner
from student_causal_training import PLAN_SHA, reused_dataset


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu,
        "Fit worker must expose only the admitted original GPU")
    require(identity(args.causal_plan)["sha256"] == PLAN_SHA
        and time.monotonic_ns() < args.deadline_ns-60_000_000_000, "R2 plan or reserved deadline changed")
    plan, settings = read(args.plan), read(args.causal_plan)
    require(plan["source"] == str(args.morph_source), "Policy source differs from frozen original plan")
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path[:0] = [str(args.morph_source/"src"), str(HERE.parent/"morphmimic_q1")]
    from bootstrap import install
    from inference import load_policy
    from student_causal_fit import fit_r2
    from student_fit import module_sha

    native, metadata = load_policy(args.checkpoint, install("q1")["robot"], plan["code_commit"],
        "cuda:0", expected_plan=plan)
    prior = read(args.training_root/"fit_result.json")
    require(metadata.get("iteration") == 32000 and module_sha(native.policy) == prior["frozen_teacher_policy_sha256"]
        and module_sha(native.policy.decoder) == prior["decoder_sha256"],
        "R2 teacher or frozen decoder differs from the actual R1 teacher")
    jobs = read(args.training_root/"collection_plan.json")["jobs"]
    dataset, values = reused_dataset(args.training_root, jobs, load=True)
    require(dataset == read(args.output/"reused_dataset.json"), "Dataset changed between admission and fit")
    result = fit_r2(native, values, settings["fit"], args.output, args.deadline_ns,
        dict(teacher=identity(args.checkpoint), datasets=read(args.training_root/"dataset_receipt.json")["files"],
            causal_plan=identity(args.causal_plan), reused_training_root=str(args.training_root),
            original_r1_fit=identity(args.training_root/"fit_result.json")))
    require(all(identity(r["path"]) == r for r in dataset["records"]), "Preserved data changed during optimization")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "morph-source", "training-root", "output", "causal-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps(dict(status=result["status"], actual_updates=result["actual_updates"], human_real_future_frames=0)))
