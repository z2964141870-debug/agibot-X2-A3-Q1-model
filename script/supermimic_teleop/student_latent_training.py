"""Independent latent-label ablation reusing the completed exact action control."""

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
from interim_review import TRAINING
from paused_runtime import own_gpu
from preview_review import environment
from runtime_guard import guarded_run
from student_auxiliary_training import aggregate_evidence
from student_causal_review import trained_causal_evidence
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle

NEW_CODE = ("student_latent_training.py", "student_latent_worker.py", "student_latent_fit.py")


def auxiliary_evidence(root):
    result, fit, guard, auth = (read(root/name) for name in
        ("result.json", "paired_fit_result.json", "fit.guard.json", "authorization.json"))
    require(result.get("status") == fit.get("status") == "PAIRED_FIT_PASSED_PENDING_CLOSED_LOOP"
        and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and guard.get("status") == "WORKER_COMPLETED"
        and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Reuse only the actual completed, settled, reaped auxiliary control")
    records = [identity(root/name) for name in ("result.json", "paired_fit_result.json", "fit.guard.json",
        "authorization.json", "transition_label_audit.json", "action_only/fit_result.json",
        "action_only/training_trace.json", "action_only/selected.pt")]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    require(all(identity(row["path"]) == row for row in records), "Reused auxiliary control sources changed")
    return records


def latent_preflight(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = train_preflight(args)
    expected = ROOT/"data/supermimic_for_q1"
    plan = read(args.latent_plan)
    prior = read(HERE/"student_auxiliary_plan_20261107.json")
    require(args.initial_r2_root == expected/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107"
        and args.aggregate_root == expected/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107"
        and args.auxiliary_root == expected/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_AUXILIARY_PAIR02_20261107"
        and args.latent_plan == HERE/"student_latent_plan_20261107.json"
        and identity(args.latent_plan)["sha256"] == args.latent_plan_sha256
        and plan.get("schema") == "q1_offline_teacher_latent_distillation_v1"
        and plan.get("human_real_future_frames") == plan.get("new_collection_steps") == 0
        and plan.get("control_retrained") is False and plan.get("test_sources_read") is False
        and plan.get("cumulative_cost_stage") == "r3" and plan.get("maximum_wall_seconds") == 1740
        and plan.get("fit", {}).get("latent_weight") == .1
        and all(value == prior["fit"].get(key) for key, value in plan["fit"].items() if key != "latent_weight"),
        "Use the registered latent-only change and exact prior fitting settings")
    require({str(HERE/name) for name in NEW_CODE} <= {row["path"] for row in code}
        and not {str(HERE/name) for name in NEW_CODE} & {row["path"] for row in sources},
        "Independent latent code is missing or entered frozen sources")
    controls += trained_causal_evidence(args.initial_r2_root, args.training_root)
    controls += aggregate_evidence(args.aggregate_root)+auxiliary_evidence(args.auxiliary_root)
    controls += [identity(args.latent_plan), identity(HERE/"student_auxiliary_plan_20261107.json")]
    unique = {}
    for row in controls:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting latent source identities")
        unique[row["path"]] = row
    require(all(identity(row["path"]) == row for row in unique.values()), "Latent admission sources changed")
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = latent_preflight(args)
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
        save(args.output/"authorization.json", dict(stage="R3_OFFLINE_TEACHER_LATENT_DISTILLATION",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, gpu_lock_owner=owner,
            cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            original_phase_protection_seconds=1800, maximum_output_bytes=1073741824,
            new_physical_transitions=0, human_real_future_frames=0, teacher_latent_labels_are_offline_only=True,
            control_retrained=False, original_training_restarted=False,
            final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_OFFLINE_TEACHER_LATENT_DISTILLATION", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, closed_loop_acceptance="PENDING",
            user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {number:signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
        path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["morph_python"], "-u", str(HERE/"student_latent_worker.py")]
            values = {"runtime-config":args.runtime_config, "config":args.config,
                "plan":TRAINING/"e1024_train_plan.json", "output":args.output,
                "training-root":args.training_root, "aggregate-root":args.aggregate_root,
                "initial-r2-root":args.initial_r2_root, "auxiliary-root":args.auxiliary_root,
                "latent-plan":args.latent_plan, "latent-plan-sha256":args.latent_plan_sha256,
                "expected-gpu":config["expected_gpu"], "gpu-owner-pid":owner["pid"],
                "gpu-owner-ticks":owner["start_ticks"], "deadline-ns":deadline}
            for key, value in values.items():
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded latent worker did not complete")
            require(all(identity(row["path"]) == row for row in controls+sources+code), "Latent protected sources changed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP" and fit.get("actual_updates") == 3000
                and fit.get("control_retrained") is False, "No complete matched latent fit")
            result.update(status=fit["status"], source_integrity="PASSED", fit_result=identity(args.output/"fit_result.json"),
                actual_optimizer_updates=3000, human_real_future_frames=0, natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if path.exists() and read(path).get("cleanup", {}).get("worker_reaped") is True:
                    result["cost_settlement"] = settle(args.cost_ledger, run_id=str(args.output), owner=owner,
                                                       cleanup=read(path)["cleanup"])
                result["ended_utc"] = now()
                save(args.output/"result.json", result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for number, handler in handlers.items():
                    signal.signal(number, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "pause-receipt", "audit-root", "code-manifest",
            "training-root", "causal-plan", "source-split", "cost-ledger", "r1-analysis-root",
            "initial-r2-root", "aggregate-root", "auxiliary-root", "latent-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256", "latent-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
