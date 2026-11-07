"""Paired residual fitting in the isolated native CUDA runtime, without physics."""

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
        "Protected residual sources or reserved deadline changed")
    plan, experiment = read(args.plan), read(args.residual_plan)
    require(plan["source"] == str(args.morph_source), "Original policy source changed")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path[:0] = [str(args.morph_source/"src"), str(HERE.parent/"morphmimic_q1")]
    from bootstrap import install
    from inference import load_policy
    from student_causal_fit import INPUT_FIELDS
    from student_fit import module_sha
    from student_proprio_fit import fit_pair
    from upper_mps_forecast import audit_cpu_tensors

    native, metadata = load_policy(args.checkpoint, install("q1")["robot"], plan["code_commit"],
        "cuda:0", expected_plan=plan)
    checkpoint = experiment["initial_checkpoint"]
    require(identity(checkpoint["path"]) == checkpoint, "Original R2 warm start changed")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    audit_cpu_tensors(torch, payload)
    require(metadata.get("iteration") == 32000
        and payload.get("schema") == "q1_low_latency_f0_history_encoder_v1"
        and payload.get("selected_update") == 500 and payload.get("human_real_future_frames") == 0
        and payload["provenance"]["teacher"] == identity(args.checkpoint),
        "Require original R2 selected500 with the audited32k teacher")
    norms = {k:module_sha(v) for k, v in native.norms.items()}
    require(all(not v.training for v in native.norms.values()), "Teacher normalizers must remain frozen")
    origins = []
    for origin in experiment["origins"]:
        value = {}
        for split in ("train", "validation"):
            with np.load(Path(origin["root"])/(split+"_data.npz"), allow_pickle=False) as arrays:
                value[split] = {k:arrays[k] for k in INPUT_FIELDS}
        origins.append(value)
    result = fit_pair(native, origins, payload, read(experiment["candidate_plan"]["path"]),
        experiment["fit"], args.output, args.deadline_ns,
        dict(teacher=identity(args.checkpoint), initial_student=checkpoint,
            datasets=experiment["origins"], residual_plan=identity(args.residual_plan),
            original_normalizers=norms))
    require(norms == {k:module_sha(v) for k, v in native.norms.items()}
        and all(identity(r["path"]) == r for r in auth["protected_inputs"]),
        "Paired residual fit changed frozen statistics or protected inputs")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "morph-source", "output", "residual-plan"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-gpu", "gpu-owner-ticks"):
        p.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        p.add_argument("--"+name, type=int, required=True)
    value = run(p.parse_args())
    print(json.dumps(dict(status=value["status"], new_optimizer_updates=value["new_optimizer_updates"],
        selected=[dict(arm=r["arm"], update=r["selected_update"], zero=r["selected_adapter_is_zero"])
            for r in value["arms"]], human_real_future_frames=0)))
