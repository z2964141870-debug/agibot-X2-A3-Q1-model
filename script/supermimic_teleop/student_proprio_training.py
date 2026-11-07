"""Single-use paired f0 residual fitting under the original cost and source gates."""

import argparse
import json
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
from paused_runtime import own_gpu, paused_state
from preview_review import environment
from runtime_guard import allocated_tree, guarded_run
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_same_state import unique
from student_training import TrainingBudget

DATA = ROOT/"data/supermimic_for_q1"
PRIOR = DATA/"Q1_LOW_LATENCY_CKPT32000_R1_H4_CPU_PAIR01_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_20261108"
PLAN = HERE/"student_proprio_plan_20261108.json"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
FILES = ("student_proprio_training.py", "student_proprio_worker.py", "student_proprio_models.py",
    "student_proprio_fit.py", "student_proprio_plan_20261108.json", "test_student_proprio.py")
WALL = 600
OUTPUT_BYTES = 128*(1 << 20)


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), "Use the single-use F0_PROPRIO_PAIR01 root")
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set("0123456789abcdef"),
        "Publish the independent residual workflow before admission")
    sample = resources(True)
    config, receipt = read(args.config), read(args.pause_receipt)
    paused_state(config, receipt)
    require(identity(args.manifest)["sha256"] == args.manifest_sha256, "Residual source manifest changed")
    manifest, plan = read(args.manifest), read(PLAN)
    require(manifest.get("schema") == "q1_f0_proprio_residual_code_v1"
        and manifest.get("evaluation_commit") == args.evaluation_commit
        and len(manifest["files"]) == len(FILES)
        and {r["path"] for r in manifest["files"]} == {str(HERE/n) for n in FILES},
        "Unexpected independent residual code scope")
    require(plan.get("schema") == "q1_f0_proprio_residual_plan_v1"
        and plan["arms"] == ["zero_history", "causal_history"]
        and plan["fit"] == dict(seed=2026110802, maximum_updates=3000, batch_size=128,
            learning_rate=.0001, gradient_norm_limit=1., validation_every_updates=500)
        and plan["human_real_future_frames"] == 0 and plan["resource"]["ledger_stage"] == "r2"
        and plan["resource"]["maximum_wall_seconds"] == WALL
        and plan["resource"]["maximum_output_bytes"] == OUTPUT_BYTES,
        "Use the registered equal-exposure residual plan")
    result, auth = read(PRIOR/"result.json"), read(PRIOR/"authorization.json")
    require(result.get("status") == "CPU_H4_PAIRED_ACTIONS_PASSED"
        and result.get("source_integrity") == "PASSED" and result.get("new_policy_calls") == 512
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Prior paired test must have completed with actual cleanup and settlement")
    prior = [r for r in auth["protected_inputs"] if r["path"] != str(LEDGER)]
    require(not {str(HERE/n) for n in FILES} & {r["path"] for r in prior},
        "New residual files overlap earlier execution inputs")
    guards = []
    for phase in result["phases"]:
        record = phase["guard"]
        guard = read(record["path"])
        require(guard.get("status") == "WORKER_COMPLETED"
            and guard.get("cleanup", {}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == 0, "Prior paired worker remains unconfirmed")
        guards.append(record)
    require(len(guards) == 4, "Require all four prior condition guards")
    dataset = [r for origin in plan["origins"] for r in origin["records"]]
    candidate = Path(plan["candidate_plan"]["path"])
    require(candidate == ROOT/"data/manifests/q1_student_coverage_candidate_20261107.json"
        and identity(candidate)["sha256"] == plan["candidate_plan"]["sha256"], "Coverage mapping changed")
    require(len(plan["origins"]) == 3 and len(dataset) == 15
        and all(read(Path(o["root"])/"result.json").get("source_integrity") == "PASSED"
            and read(Path(o["root"])/"collection_result.json").get("status") == "PASSED"
            for o in plan["origins"]), "Use the three completed preserved train/validation origins")
    controls = [identity(p) for p in (PRIOR/"result.json", PRIOR/"authorization.json",
        PRIOR/"checkpoint_audit.json", args.config, args.runtime_config, args.pause_receipt,
        args.manifest, candidate, TRAINING/"e1024_train_plan.json")]
    protected = unique(prior+guards+dataset+[plan["initial_checkpoint"], plan["initial_fit"]]
        +controls+manifest["files"])
    require(str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Protected residual source identity changed")
    initial = read(plan["initial_fit"]["path"])
    require(initial.get("selected_update") == 500 and initial.get("actual_updates") == 3500,
        "Warm start the original R2 selected500, not coverage or a diagnostic last checkpoint")
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == s])) for s in STAGES}
    require(charged(ledger) == plan["resource"]["expected_cumulative_seconds"] == 3557
        and by_stage == plan["resource"]["expected_by_stage"]
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r2"]-by_stage["r2"] >= WALL and charged(ledger)+WALL <= LIMIT,
        "Retain the settled3557 seconds and original stage limits")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+OUTPUT_BYTES < 500_000_000_000,
        "Private project500GB bound reached")
    with own_gpu() as owner:
        require(owner is not None, "Idle original GPU lock admission failed")
    return sample, protected, plan


class ProprioBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < OUTPUT_BYTES,
            "Residual output128MiB bound reached")
        return super().check()


def execute(args):
    sample, protected, plan = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", read(PRIOR/"checkpoint_audit.json"))
        save(args.output/"reused_dataset.json", dict(origins=plan["origins"],
            coverage=plan["data_coverage"], actual_states_and_labels_reused=True,
            test_sources_read=False, new_physical_steps=0))
        budget = ProprioBudget(config, receipt, args.output, owner)
        budget.maximum_wall = WALL
        budget.check()
        previous, reservation, requested = dict(os.environ), None, False
        result = dict(stage="F0_PROPRIO_RESIDUAL_PAIR", status="RUNNING", controller=owner,
            started_utc=now(), admission=sample, new_physical_steps=0, new_rendered_frames=0,
            user_accepted_checkpoint=None)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r2", owner=owner,
                gpu=config["expected_gpu"], commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            budget.maximum_wall = min(WALL, reservation["reserved_seconds"]-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/"authorization.json", dict(stage="F0_PROPRIO_RESIDUAL_PAIR",
                evaluation_commit=args.evaluation_commit, protected_inputs=protected, gpu_lock_owner=owner,
                residual_plan=identity(PLAN), maximum_wall_seconds=budget.maximum_wall,
                maximum_total_optimizer_updates=6000, maximum_output_bytes=OUTPUT_BYTES,
                maximum_human_future_frames=0, original_training_restarted=False,
                cache_reclaim_attempted=False, normalization_refitted=False,
                final_acceptance=False, user_accepted_checkpoint=None))
            audit = read(PRIOR/"checkpoint_audit.json")
            selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
            original = read(TRAINING/"e1024_train_plan.json")
            values = dict(plan=TRAINING/"e1024_train_plan.json", checkpoint=selected["path"],
                morph_source=original["source"], output=args.output, residual_plan=PLAN,
                expected_gpu=config["expected_gpu"], gpu_owner_pid=owner["pid"],
                gpu_owner_ticks=owner["start_ticks"], deadline_ns=deadline)
            command = [runtime["morph_python"], "-u", str(HERE/"student_proprio_worker.py")]
            for key, value in values.items():
                command.extend(("--"+key.replace("_", "-"), str(value)))
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded paired residual fit failed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "PAIRED_RESIDUAL_TRAINED_PENDING_CLOSED_LOOP"
                and fit.get("new_optimizer_updates") == 6000
                and [r["arm"] for r in fit["arms"]] == plan["arms"]
                and fit.get("data_coverage") == plan["data_coverage"]
                and fit.get("matched_batch_order_and_exposure") is True
                and all(identity(r["path"]) == r for r in protected), "Paired result or protected sources changed")
            result.update(status=fit["status"], source_integrity="PASSED", actual_optimizer_updates=6000,
                arms=[dict(arm=r["arm"], selected_update=r["selected_update"],
                    selected_adapter_is_zero=r["selected_adapter_is_zero"],
                    initial_validation_risk=r["initial_validation_risk"],
                    selected_validation_risk=r["selected_validation_risk"]) for r in fit["arms"]],
                fit_result=identity(args.output/"fit_result.json"), natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if reservation:
                cleanup = read(guard_path).get("cleanup", {}) if guard_path.is_file() else {}
                if cleanup.get("worker_reaped") is True:
                    result["cost_settlement"] = settle(LEDGER, run_id=str(args.output), owner=owner, cleanup=cleanup)
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json", result)
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, default in dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
            runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
            pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json").items():
        p.add_argument("--"+name.replace("_", "-"), type=Path, default=default)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--evaluation-commit", required=True)
    p.add_argument("--admit-only", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.admit_only:
        sample, protected, plan = admission(args)
        print(json.dumps(dict(status="F0_PROPRIO_PAIR_ADMITTED", protected_files=len(protected),
            resources=sample, coverage=plan["data_coverage"], new_optimizer_updates=0)))
    else:
        execute(args)
