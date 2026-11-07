"""Fit the two source-matched arms using only already collected causal inputs."""

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
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu, "Use only the admitted original GPU")
    auth, original, experiment = read(args.output/"authorization.json"), read(args.plan), read(args.pair_plan)
    require(time.monotonic_ns() < args.deadline_ns-60_000_000_000
        and all(identity(r["path"]) == r for r in auth["protected_inputs"])
        and original["source"] == str(args.morph_source), "Source identity or reserved deadline changed")
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
    from student_onpolicy_pair_fit import fit_pair
    from student_proprio_models import zero_adapter
    from upper_mps_forecast import audit_cpu_tensors
    native, metadata = load_policy(args.checkpoint, install("q1")["robot"], original["code_commit"],
        "cuda:0", expected_plan=original)
    initial = experiment["initial_checkpoint"]
    require(identity(initial["path"]) == initial, "Actual selected3000 warm start changed")
    payload = torch.load(initial["path"], map_location="cpu", weights_only=True)
    audit_cpu_tensors(torch, payload)
    require(metadata.get("iteration") == 32000 and payload.get("schema") == "q1_f0_proprio_residual_v1"
        and payload.get("arm") == "causal_history" and payload.get("selected_update") == 3000
        and payload.get("actual_updates") == 3000 and payload.get("human_real_future_frames") == 0
        and payload["provenance"]["teacher"] == identity(args.checkpoint) and not zero_adapter(payload["adapter"]),
        "Continue the actual nonzero residual selected3000 with its original32k teacher")
    norms = {k:module_sha(v) for k, v in native.norms.items()}
    require(all(not v.training for v in native.norms.values()), "Teacher statistics must stay frozen")

    def inputs(path):
        with np.load(path, allow_pickle=False) as values:
            return {name:values[name] for name in INPUT_FIELDS}

    origins = [{s:inputs(Path(o["root"])/(s+"_data.npz")) for s in ("train", "validation")}
        for o in experiment["origins"]]
    pure = {s:inputs(experiment["pure_transitions"][s]["path"]) for s in ("train", "validation")}
    result = fit_pair(native, origins, pure, read(experiment["candidate_plan"]["path"]),
        experiment["jobs"], payload, experiment["fit"], args.output, args.deadline_ns,
        dict(teacher=identity(args.checkpoint), initial_student=initial, datasets=experiment["origins"],
            pure_transitions=experiment["pure_transitions"], pair_plan=identity(args.pair_plan),
            original_normalizers=norms))
    require(result["old_records"] == dict(train=6725, validation=2358)
        and result["pure_records"] == dict(train=406, validation=219)
        and norms == {k:module_sha(v) for k, v in native.norms.items()}
        and all(identity(r["path"]) == r for r in auth["protected_inputs"]),
        "Dataset coverage, frozen statistics or protected sources changed")
    torch.cuda.synchronize()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "morph-source", "output", "pair-plan"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-gpu", "gpu-owner-ticks"):
        p.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        p.add_argument("--"+name, type=int, required=True)
    value = run(p.parse_args())
    print(json.dumps(dict(status=value["status"], actual_optimizer_updates=value["actual_optimizer_updates"],
        selected=[dict(arm=r["variant"], update=r["selected_update"], unchanged=r["selected_adapter_unchanged"])
            for r in value["arms"]], human_real_future_frames=0)))
