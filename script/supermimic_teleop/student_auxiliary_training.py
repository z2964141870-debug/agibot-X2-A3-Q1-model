"""One independent paired GigaBrain-property ablation; original runs stay frozen."""

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
from student_causal_review import trained_causal_evidence
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle


NEW_CODE = ("student_auxiliary_training.py", "student_auxiliary_worker.py",
            "student_auxiliary_fit.py", "student_auxiliary_data.py")


def aggregate_evidence(root):
    result, fit, guard, auth, collection, receipt = (read(root/name) for name in
        ("result.json", "fit_result.json", "fit.guard.json", "authorization.json",
         "collection_result.json", "dataset_receipt.json"))
    require(result.get("method") == fit.get("method") == "one_round_student_state_aggregation"
        and result.get("source_integrity") == "PASSED"
        and result.get("status") == fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and guard.get("status") == "WORKER_COMPLETED"
        and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists()
        and collection.get("status") == "PASSED" and collection.get("test_sources_read") is False
        and receipt.get("status") == "MIXED_STUDENT_STATE_LABELS_SAVED",
        "Need actual source-gated, settled and reaped aggregation")
    records = [identity(root/name) for name in ("result.json", "fit_result.json", "fit.guard.json",
        "authorization.json", "collection_result.json", "dataset_receipt.json", "collection_plan.json")]
    records += list(receipt["files"].values())+[row["trace"] for row in collection["clips"]]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    unique = {}
    for row in records:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting prior source identities")
        unique[row["path"]] = row
    require(all(identity(row["path"]) == row for row in unique.values()), "Prior aggregation source changed")
    return list(unique.values())


def auxiliary_preflight(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = train_preflight(args)
    from student_aggregate import PLAN_SHA
    from student_causal_training import PLAN_SHA as CAUSAL_SHA
    plan = read(args.auxiliary_plan)
    require(args.aggregate_root == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107"
        and args.initial_r2_root == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107"
        and args.auxiliary_plan == HERE/"student_auxiliary_plan_20261107.json"
        and identity(args.auxiliary_plan)["sha256"] == args.auxiliary_plan_sha256
        and plan.get("schema") == "q1_behavior_conditioned_auxiliary_pair_v1"
        and plan.get("original_r2_plan_sha256") == CAUSAL_SHA
        and plan.get("original_aggregate_plan_sha256") == PLAN_SHA
        and plan.get("human_real_future_frames") == 0
        and plan.get("new_collection_steps") == 0
        and plan.get("cumulative_cost_stage") == "r3"
        and plan.get("maximum_wall_seconds") == 1740
        and plan.get("fit", {}).get("maximum_updates") == 3000
        and plan.get("fit", {}).get("auxiliary_weight") == .1,
        "Use the registered single-property paired experiment")
    require({str(HERE/name) for name in NEW_CODE} <= {row["path"] for row in code}
        and not {str(HERE/name) for name in NEW_CODE} & {row["path"] for row in sources},
        "Independent auxiliary code is missing or entered frozen sources")
    controls += trained_causal_evidence(args.initial_r2_root, args.training_root)
    controls += aggregate_evidence(args.aggregate_root)+[identity(args.auxiliary_plan)]
    unique = {row["path"]:row for row in controls}
    require(all(identity(row["path"]) == row for row in unique.values()), "Auxiliary admission sources changed")
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = auxiliary_preflight(args)
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
        save(args.output/"authorization.json", dict(stage="R3_BEHAVIOR_CONDITIONED_NEXT_STATE_PAIR",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, gpu_lock_owner=owner,
            cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            original_phase_protection_seconds=1800, maximum_output_bytes=1073741824,
            new_physical_transitions=0, human_real_future_frames=0,
            next_state_labels_are_offline_only=True, auxiliary_head_deployed=False,
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_BEHAVIOR_CONDITIONED_NEXT_STATE_PAIR", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, closed_loop_acceptance="PENDING",
            user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {n:signal.signal(n, stop) for n in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["morph_python"], "-u", str(HERE/"student_auxiliary_worker.py")]
            values = {"runtime-config":args.runtime_config, "config":args.config,
                "plan":TRAINING/"e1024_train_plan.json", "output":args.output,
                "training-root":args.training_root, "aggregate-root":args.aggregate_root,
                "initial-r2-root":args.initial_r2_root, "auxiliary-plan":args.auxiliary_plan,
                "auxiliary-plan-sha256":args.auxiliary_plan_sha256,
                "expected-gpu":config["expected_gpu"], "gpu-owner-pid":owner["pid"],
                "gpu-owner-ticks":owner["start_ticks"], "deadline-ns":deadline}
            for key, value in values.items():
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded paired worker did not complete")
            require(all(identity(row["path"]) == row for row in controls+sources+code), "Paired protected sources changed")
            fit = read(args.output/"paired_fit_result.json")
            require(fit.get("status") == "PAIRED_FIT_PASSED_PENDING_CLOSED_LOOP"
                and fit.get("actual_optimizer_updates") == 6000, "No complete matched paired fit")
            result.update(status=fit["status"], source_integrity="PASSED", fit_result=identity(args.output/"paired_fit_result.json"),
                actual_optimizer_updates=6000, human_real_future_frames=0, natural_teleoperation_qualified=False)
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
            "initial-r2-root", "aggregate-root", "auxiliary-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256", "auxiliary-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
