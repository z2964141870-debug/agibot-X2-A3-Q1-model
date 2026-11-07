"""One bounded physical-loss experiment, preserving every original run."""

import argparse
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import ROOT, identity, now, require, save
from delivery import read
from paused_runtime import own_gpu
from preview_review import environment
from runtime_guard import guarded_run
from student_auxiliary_training import aggregate_evidence
from student_causal_review import trained_causal_evidence
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle
from student_latent_training import auxiliary_evidence

NEW_CODE = ("wm_physical_training.py", "wm_physical_worker.py", "wm_physical_fit.py", "wm_physical_loss.py")


def precision_evidence(root, expected):
    paths = [root/n for n in ("result.json", "precision_result.json", "precision.guard.json", "authorization.json")]
    result, precision, guard, authorization = map(read, paths)
    require(identity(root/"precision_result.json") == {k:expected[k] for k in ("path", "bytes", "sha256")}
        and result.get("status") == precision.get("status") == "WM_PRECISION_COMPLETED"
        and result.get("source_integrity") == precision.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and precision.get("diagnostic_screen") == "PASSED"
        and precision.get("native_reproduction_passed") is True
        and precision.get("shadow_prediction_preservation", {}).get("passed") is True
        and all(g["finite_difference_passed"] for g in precision["shadow_gradients"])
        and guard.get("status") == "WORKER_COMPLETED"
        and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Use only the actual passed, settled and reaped FP32 diagnostic")
    records = [identity(p) for p in paths]+[r for key in
        ("evaluation_sources", "training_source_inputs", "controller_inputs") for r in authorization[key]]
    require(all(identity(r["path"]) == r for r in records), "Passed precision sources changed")
    return records


def physical_preflight(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = train_preflight(args)
    settings = read(args.physical_plan)
    base = ROOT/"data/supermimic_for_q1"
    require(args.initial_r2_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107"
        and args.aggregate_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107"
        and args.auxiliary_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_AUXILIARY_PAIR02_20261107"
        and args.precision_root == base/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PRECISION01_20261107"
        and args.physical_plan == HERE/"wm_physical_distill_plan_20261107.json"
        and identity(args.physical_plan)["sha256"] == args.physical_plan_sha256
        and settings.get("schema") == "q1_frozen_fp32_wm_physical_distillation_v1"
        and settings.get("revision") == 2 and settings.get("human_real_future_frames") == 0
        and settings.get("maximum_updates") == 3000 and settings.get("learning_rate") == .0001
        and settings.get("seed") == 2026110703 and settings.get("maximum_wm_step_calls") == 4096
        and settings.get("fit_phase_seconds") == 1740 and settings.get("test_sources_read") is False
        and settings.get("new_collection_allowed") is False, "Use the source-audited registered physical-only fit")
    require({str(HERE/n) for n in NEW_CODE} <= {r["path"] for r in code}
        and not {str(HERE/n) for n in NEW_CODE} & {r["path"] for r in sources},
        "New physical sources are missing or overlap original frozen inputs")
    require(args.probe_plan == HERE/"wm_probe_plan_20261107.json", "Use the verified nominal context plan")
    controls += trained_causal_evidence(args.initial_r2_root, args.training_root)
    controls += aggregate_evidence(args.aggregate_root)+auxiliary_evidence(args.auxiliary_root)
    controls += precision_evidence(args.precision_root, settings["required_precision_result"])
    controls += [identity(args.physical_plan), identity(args.probe_plan),
                 identity(HERE/"student_auxiliary_plan_20261107.json")]+read(args.probe_plan)["context_sources"]
    unique = {}
    for row in controls:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting physical input identities")
        unique[row["path"]] = row
    require(all(identity(r["path"]) == r for r in unique.values()), "Physical fit admission sources changed")
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = physical_preflight(args)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"reused_dataset.json", dataset)
        save(args.output/"cache_boundary.json", dict(stage="REUSE_ONCE_BEFORE_R2_BOUNDARY",
            prior=identity(args.initial_r2_root/"cache_boundary.json"), new_reclaim_attempted=False))
        budget = CausalBudget(config, receipt, args.output, owner)
        budget.check()
        started = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r3", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started)
        budget.maximum_wall = min(1740, max(0, reservation["reserved_seconds"]-5))
        deadline = started+max(0, budget.maximum_wall-5)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R3_FROZEN_FP32_WM_PHYSICAL_DISTILLATION",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, gpu_lock_owner=owner,
            cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            original_phase_protection_seconds=1800, maximum_output_bytes=1073741824,
            maximum_wm_step_calls=4096, new_physical_transitions=0, human_real_future_frames=0,
            control_retrained=False, world_model_deployed=False, original_training_restarted=False,
            final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_FROZEN_FP32_WM_PHYSICAL_DISTILLATION", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {n:signal.signal(n, stop) for n in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["morph_python"], "-u", str(HERE/"wm_physical_worker.py")]
            values = {"runtime-config":args.runtime_config, "config":args.config,
                "plan":Path(config["output"])/"e1024_train_plan.json", "output":args.output,
                "training-root":args.training_root, "aggregate-root":args.aggregate_root,
                "initial-r2-root":args.initial_r2_root, "auxiliary-root":args.auxiliary_root,
                "probe-plan":args.probe_plan, "precision-root":args.precision_root,
                "physical-plan":args.physical_plan, "physical-plan-sha256":args.physical_plan_sha256,
                "expected-gpu":config["expected_gpu"], "gpu-owner-pid":owner["pid"],
                "gpu-owner-ticks":owner["start_ticks"], "deadline-ns":deadline}
            for key, value in values.items():
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded physical worker did not complete")
            require(all(identity(r["path"]) == r for r in controls+sources+code), "Physical protected inputs changed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP" and fit.get("actual_updates") == 3000
                and fit.get("control_retrained") is False and fit.get("wm_step_calls", 4097) <= 4096,
                "No complete matched physical fit")
            result.update(status=fit["status"], source_integrity="PASSED", fit_result=identity(args.output/"fit_result.json"),
                actual_optimizer_updates=3000, human_real_future_frames=0, natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                    result["cost_settlement"] = settle(args.cost_ledger, run_id=str(args.output), owner=owner,
                        cleanup=read(guard_path)["cleanup"])
                result["ended_utc"] = now()
                save(args.output/"result.json", result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for n, handler in handlers.items():
                    signal.signal(n, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "pause-receipt", "audit-root", "code-manifest",
            "training-root", "causal-plan", "source-split", "cost-ledger", "r1-analysis-root",
            "initial-r2-root", "aggregate-root", "auxiliary-root", "probe-plan", "precision-root", "physical-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256", "physical-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
