"""Owned H4 specialist fitting; no simulator, data collection or teacher trainer."""

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


def run(args):
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu,
            "Expose only the admitted original GPU")
    auth = read(args.output/"authorization.json")
    require(time.monotonic_ns() < args.deadline_ns-60_000_000_000
        and all(identity(r["path"]) == r for r in auth["protected_inputs"]),
        "Admitted H4 sources or reserved deadline changed")
    plan, settings = read(args.plan), read(args.h4_plan)
    require(plan["source"] == str(args.morph_source), "Original policy source changed")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA device unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path[:0] = [str(args.morph_source/"src"), str(HERE.parent/"morphmimic_q1")]
    from bootstrap import install
    from inference import load_policy
    from student_fit import module_sha
    from student_h4_fit import fit
    from upper_mps_forecast import audit_cpu_tensors

    native, metadata = load_policy(args.checkpoint, install("q1")["robot"], plan["code_commit"],
                                  "cuda:0", expected_plan=plan)
    payload = torch.load(args.training_root/"selected.pt", map_location="cpu", weights_only=True)
    audit_cpu_tensors(torch, payload)
    require(metadata.get("iteration") == 32000 and payload["provenance"]["teacher"] == identity(args.checkpoint),
            "H4 teacher or warm-start provenance changed")
    norms = {key:module_sha(value) for key, value in native.norms.items()}
    require(all(not value.training for value in native.norms.values()), "Teacher statistics must remain frozen")
    data = {}
    for split in ("train", "validation"):
        with np.load(args.training_root/(split+"_data.npz"), allow_pickle=False) as arrays:
            data[split] = dict(arrays)
    result = fit(native, data, payload, settings["fit"], args.output, args.deadline_ns,
        dict(teacher=identity(args.checkpoint), initial_student=identity(args.training_root/"selected.pt"),
            datasets=read(args.training_root/"dataset_receipt.json")["files"],
            h4_plan=identity(args.h4_plan), original_r1_fit=identity(args.training_root/"fit_result.json"),
            original_normalizers=norms))
    require(norms == {key:module_sha(value) for key, value in native.norms.items()}
        and all(identity(r["path"]) == r for r in auth["protected_inputs"]),
        "H4 fit changed normalizers or protected inputs")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "morph-source", "training-root", "output", "h4-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-gpu", "gpu-owner-ticks"):
        parser.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        parser.add_argument("--"+name, type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps(dict(status=result["status"], actual_updates=result["actual_updates"],
                          selected_update=result["selected_update"], human_real_future_frames=4)))
