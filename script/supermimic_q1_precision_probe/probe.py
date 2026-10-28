"""Paired BF16/FP32 inference sensitivity with frozen control and zero optimization."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_activity_probe"),
               str(HERE.parent / "supermimic_q1_pd_probe"), str(HERE.parent / "supermimic_q1_gait_comp"),
               str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "supermimic_q1_ppo"),
               str(HERE.parent / "project_tools")]
from worker import process_identity
from resources import identity, save, stamp
from run_trial import Q1Budget
from runtime_guard import guarded_run
from overnight import OPTIMIZATION_DEADLINE
from precision_control import precision_checkpoint

CANDIDATES = {"bf16_static050": ("bf16", "baseline"), "fp32_static050": ("fp32", "baseline"),
              "bf16_activity050075": ("bf16", "activity050075"),
              "fp32_activity050075": ("fp32", "activity050075")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def first_action_diagnostic(teacher, count, path, precision, scale):
    import torch
    from unisupermimic.rl.algos.supermimic import networks

    def capture(policy, inputs, selected_output):
        handle.remove()
        previous = networks.AMP_DTYPE
        try:
            networks.AMP_DTYPE = None if precision == "bf16" else torch.bfloat16
            alternative = policy(*inputs)
        finally:
            networks.AMP_DTYPE = previous
        difference = (selected_output[:count].float() - alternative[:count].float()).abs()
        target_delta = difference * scale
        save(path, dict(selected_precision=precision, environments=count,
             fixed_forward_block=int(selected_output.shape[0]),
             mean_action_difference=float(difference.mean()), max_action_difference=float(difference.max()),
             mean_joint_target_difference_rad=float(target_delta.mean()),
             max_joint_target_difference_rad=float(target_delta.max()),
             lower_joint_target_rms_rad=float(target_delta[:, :14].square().mean().sqrt()),
             upper_joint_target_rms_rad=float(target_delta[:, 14:].square().mean().sqrt()),
             output_dtype=str(selected_output.dtype), weight_dtype=str(next(policy.parameters()).dtype),
             actual_autocast_dtype=str(previous), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
             physical_steps=0, optimizer_updates=0,
             scope="First real padded4096 teacher call, identical history/goal/weights; unblended nominal targets, not a rollout error or causal proof."))
    handle = teacher.register_forward_hook(capture)
    return handle


def worker(a):
    import numpy as np
    import torch
    pd = module("frozen_precision_pd", HERE.parent / "supermimic_q1_pd_probe/probe.py")
    activity = module("frozen_precision_activity", HERE.parent / "supermimic_q1_activity_probe/probe.py")
    from physical import FIXED_DEMOS, summarize, gait_summary, comparison
    from diagnose_q1_upper_reference import policy_digest
    from overnight import continuing_evidence

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = json.loads(a.library.read_text())
    rows = [dict(row, split=split) for split in ("train", "validation") for row in source["splits"][split]
            if a.scope == "full330" or split == "validation" or row["name"] in FIXED_DEMOS]
    if len(rows) != (330 if a.scope == "full330" else 33):
        raise ValueError("Original fixed library required")
    library = a.output / "evaluation_library.json"
    save(library, {"splits": {"train": rows}, "source": identity(a.library)})
    run = a.output / "run"
    run.mkdir(exist_ok=False)
    status = dict(status="INITIALIZING", process=process_identity(), started_utc=stamp(), optimizer_steps=0)
    save(run / "status.json", status)
    baseline, previous_traces, results = [], {}, []
    thresholds = json.loads(a.calibration.read_text())["thresholds"]
    with torch.no_grad():
        for candidate in a.candidates:
            precision, blend = CANDIDATES[candidate]
            summaries, comparisons = [], []
            for repeat in range(3):
                seed = 32000 + repeat
                case = run / f"{candidate}_repeat{repeat}"
                case.mkdir()
                env = None
                try:
                    env = pd.prepare_scene(a, library, case, a.runtime / "q1_assets/scene.xml", seed, len(rows))
                    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
                    from unisupermimic.rl.algos.supermimic import networks
                    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES, action_scale
                    checkpoint = torch.load(a.checkpoint, map_location=env.device, weights_only=True)
                    teacher = load_policy(env, precision_checkpoint(checkpoint, precision)).requires_grad_(False).eval()
                    expected = torch.bfloat16 if precision == "bf16" else None
                    if networks.AMP_DTYPE != expected or any(p.dtype != torch.float32 for p in teacher.parameters()):
                        raise ValueError("Actual policy inference precision differs from paired contract")
                    del checkpoint
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    digest = policy_digest(teacher)
                    scale = torch.tensor([action_scale(n) for n in Q1_JOINT_NAMES], device=env.device)
                    handle = first_action_diagnostic(teacher, len(rows), case / "initial_precision_delta.json", precision, scale)
                    def progress(frame):
                        status.update(status="RUNNING", candidate=candidate, repeat=repeat, frame=frame, updated_utc=stamp())
                        save(run / "status.json", status)
                    progress(0)
                    trace = activity.rollout(env, teacher, thresholds, blend, progress)
                    handle.remove()
                    path = a.output / f"{candidate}_repeat{repeat}.npz"
                    np.savez_compressed(path, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, seed=seed, candidate=candidate,
                                   inference_precision=precision, upper_composition=blend, trace=identity(path))
                    if candidate == "bf16_static050":
                        baseline.append(path)
                    else:
                        with np.load(baseline[repeat], allow_pickle=False) as saved:
                            paired = comparison(trace, {key: saved[key] for key in saved.files}, rows)
                        summary["initial_static075"] = paired
                        comparisons.append(paired)
                    if precision == "fp32":
                        reference = previous_traces[("bf16_" + candidate.split("_", 1)[1], repeat)]
                        with np.load(reference, allow_pickle=False) as saved:
                            summary["same_control_bf16"] = comparison(trace, {key: saved[key] for key in saved.files}, rows)
                    previous_traces[(candidate, repeat)] = path
                    if policy_digest(teacher) != digest or networks.AMP_DTYPE != expected:
                        raise AssertionError("Teacher weights, normalization or requested precision changed")
                    save(path.with_suffix(".json"), summary)
                    summaries.append(summary)
                    del teacher, trace
                finally:
                    if env is not None:
                        env.close()
                        env.backend.close()
                    del env
                    gc.collect()
                    torch.cuda.empty_cache()
            results.append(dict(candidate=candidate, summaries=summaries,
                                physical_continuation_diagnostic=continuing_evidence(comparisons) if comparisons else None))
            save(a.output / "partial_results.json", dict(candidates=results, optimizer_steps=0))
    save(a.output / "results.json", dict(status="COMPLETED", optimizer_steps=0, rows=rows, candidates=results,
        checkpoint=identity(a.checkpoint), scope=a.scope, user_accepted_checkpoint=None,
        limitations="Same frozen weights; explicit FP32/none autocast and TF32 disabled in both. Three configured evaluation seeds, not training seeds or bitwise native equality. Validation searched, not held-out."))
    status.update(status="COMPLETED", ended_utc=stamp(), frozen_teacher_unchanged=True)
    save(run / "status.json", status)


def main(a):
    if a.worker:
        try:
            worker(a)
        except BaseException as error:
            path = a.output / "run/status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
            save(path, status)
            raise
        return
    if (a.output / "status.json").exists():
        raise FileExistsError(a.output / "status.json")
    if not a.controller:
        a.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(a.frozen_plan.read_text())["source"]
    sources = {item["path"]: item for item in previous}
    sources.update({str(p): identity(p) for p in [*HERE.glob("*.py"), a.checkpoint, a.library, a.calibration]})
    source = list(sources.values())
    for item in source:
        if identity(item["path"]) != item:
            raise RuntimeError("Frozen dependency changed: " + item["path"])
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != a.expected_boot:
        raise RuntimeError("Unexpected boot")
    save(a.output / "plan.json", dict(source=source, started_utc=stamp(), process=process_identity(),
        optimizer_steps=0, maximum_wall_seconds=2400, scope=a.scope, candidates=a.candidates,
        code_commit=a.code_commit, optimization_deadline_unix=OPTIMIZATION_DEADLINE,
        expected_boot=a.expected_boot, expected_gpu=a.expected_gpu))
    class Budget(Q1Budget):
        def check(self):
            value = super().check()
            if value["gpu"]["uuid"] != a.expected_gpu:
                raise RuntimeError("Unexpected GPU")
            if time.monotonic()-self.started >= 2400 or time.time() >= OPTIMIZATION_DEADLINE-55:
                raise RuntimeError("Bounded precision probe deadline reached")
            return value
    budget = Budget(SimpleNamespace(output=a.output, runtime=a.runtime, mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    status = dict(status="RUNNING", process=process_identity(), started_utc=stamp(), optimizer_steps=0, no_automatic_retry=True)
    save(a.output / "status.json", status)
    try:
        guard = guarded_run(command, ROOT, budget, a.output / "guard.json", ROOT / "logs" / (a.output.name+".log"),
            progress_path=a.output / "run/status.json", maximum_idle_s=900, grace_s=45)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Precision probe failed; no automatic restart")
        for item in source:
            if identity(item["path"]) != item:
                raise RuntimeError("Frozen dependency changed: " + item["path"])
        status.update(status="COMPLETED", ended_utc=stamp(), frozen_source_count=len(source))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "frozen-plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--scope", choices=("quick33", "full330"), default="quick33")
    p.add_argument("--candidates", choices=tuple(CANDIDATES), nargs="+", default=list(CANDIDATES))
    p.add_argument("--expected-boot", required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--detach", action="store_true")
    p.add_argument("--controller", action="store_true")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT / "data/supermimic_for_q1"):
        p.error("Use sole experiment root")
    if (len(a.code_commit) != 40 or len(set(a.candidates)) != len(a.candidates)
            or a.candidates[0] != "bf16_static050"
            or any(name.startswith("fp32_") and "bf16_" + name.split("_", 1)[1] not in a.candidates[:a.candidates.index(name)]
                   for name in a.candidates)):
        p.error("Require published40-digit source and baseline-first matched BF16 before FP32 candidates")
    if time.time() >= OPTIMIZATION_DEADLINE-2400:
        p.error("Insufficient bounded exploration window")
    if a.detach:
        a.output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__)), *[v for v in sys.argv[1:] if v != "--detach"], "--controller"]
        with (ROOT / "logs" / (a.output.name+"_controller.log")).open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        save(a.output / "launch.json", dict(controller_pid=process.pid, started_utc=stamp(), optimizer_steps=0))
        print(json.dumps(dict(controller_pid=process.pid, output=str(a.output), optimizer_steps=0)), flush=True)
    else:
        main(a)
