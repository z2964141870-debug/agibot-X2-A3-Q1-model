"""Single-use source-matched continuation after the completed pure-state collection."""

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
PRIOR = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_LABELS01_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_PAIR01_20261108"
PLAN = HERE/"student_onpolicy_pair_plan_20261108.json"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
FILES = ("student_onpolicy_pair_training.py", "student_onpolicy_pair_worker.py", "student_onpolicy_pair_data.py",
    "student_onpolicy_pair_fit.py", "student_onpolicy_pair_plan_20261108.json", "test_student_onpolicy_pair.py")
WALL = 600
OUTPUT_BYTES = 128*(1 << 20)


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), "Use the single-use F0_ONPOLICY_PAIR01 root")
    sample = resources(True)
    config, receipt = read(args.config), read(args.pause_receipt)
    paused_state(config, receipt)
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set("0123456789abcdef")
        and identity(args.manifest)["sha256"] == args.manifest_sha256, "Require the published code and manifest")
    manifest, plan = read(args.manifest), read(PLAN)
    require(manifest.get("schema") == "q1_f0_onpolicy_pair_code_v1"
        and manifest.get("evaluation_commit") == args.evaluation_commit
        and len(manifest["files"]) == len(FILES)
        and {r["path"] for r in manifest["files"]} == {str(HERE/n) for n in FILES}, "Unexpected new code scope")
    require(plan.get("schema") == "q1_f0_onpolicy_pair_plan_v1" and plan["arms"] == ["saved_only", "onpolicy_mix"]
        and plan["fit"] == dict(seed=2026110804, maximum_updates=3000, batch_size=128,
            learning_rate=.0001, gradient_norm_limit=1., validation_every_updates=500)
        and plan["human_real_future_frames"] == 0 and plan["resource"]["ledger_stage"] == "r2"
        and plan["resource"]["maximum_wall_seconds"] == WALL
        and plan["resource"]["maximum_output_bytes"] == OUTPUT_BYTES, "Registered paired plan changed")
    prior, auth = read(PRIOR/"result.json"), read(PRIOR/"authorization.json")
    guard, collection = read(PRIOR/"collection.guard.json"), read(PRIOR/"collection_result.json")
    require(prior.get("status") == "COMPLETED" and prior.get("source_integrity") == "PASSED"
        and prior.get("cost_settlement", {}).get("state") == "SETTLED"
        and not Path("/proc", str(prior["controller"]["pid"])).exists()
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(guard["cleanup"]["worker_pid"])).exists()
        and collection.get("status") == "PURE_STUDENT_SHADOW_LABELS_PASSED"
        and collection.get("teacher_behavior_fraction") == 0. and collection.get("new_optimizer_updates") == 0
        and collection.get("actual_physical_steps") == 1259 and collection.get("teacher_queries") == 625,
        "Require the actual cleanly settled pure-student collection")
    require(collection["transitions"] == plan["pure_transitions"]
        and {s:collection["data_validation"][s]["records"] for s in ("train", "validation")} == dict(train=406, validation=219),
        "Pure-state label identity or count changed")
    collection_plan = read(HERE/"student_onpolicy_plan_20261108.json")
    require(plan["jobs"] == collection_plan["jobs"]
        and plan["initial_checkpoint"] == collection_plan["selected_checkpoint"], "Collected policy or matched jobs changed")
    prior_inputs = [r for r in auth["protected_inputs"] if r["path"] != str(LEDGER)]
    require(not {str(HERE/n) for n in FILES}&{r["path"] for r in prior_inputs}, "New files overlap frozen inputs")
    old_plan = read(HERE/"student_proprio_plan_20261108.json")
    require(plan["origins"] == old_plan["origins"] and plan["candidate_plan"] == old_plan["candidate_plan"],
        "Preserve the three old origins and their source mapping")
    generated = [identity(PRIOR/n) for n in ("result.json", "authorization.json", "checkpoint_audit.json",
        "collection.guard.json", "collection_result.json", "collection_plan.json")]
    generated += [r[k] for r in collection["clips"] for k in ("trace", "timing")]
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt,
        args.manifest, TRAINING/"e1024_train_plan.json")]
    protected = unique(prior_inputs+generated+controls+manifest["files"]
        +list(plan["pure_transitions"].values())+[plan["initial_checkpoint"], plan["initial_fit"]]
        +[r for origin in plan["origins"] for r in origin["records"]])
    require(str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Protected paired sources changed")
    initial = read(plan["initial_fit"]["path"])
    require(initial.get("arm") == "causal_history" and initial.get("selected_update") == 3000
        and initial.get("actual_updates") == 3000 and initial.get("selected_adapter_is_zero") is False,
        "Continue the actual nonzero history residual, without reinitialization")
    ledger = checked(read(LEDGER))
    by_stage = {s:charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == s])) for s in STAGES}
    require(charged(ledger) == plan["resource"]["expected_cumulative_seconds"] == 4380
        and by_stage == plan["resource"]["expected_by_stage"]
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r2"]-by_stage["r2"] >= WALL and charged(ledger)+WALL <= LIMIT, "Preserve the settled4380 ledger")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+OUTPUT_BYTES < 500_000_000_000, "Project500GB bound")
    with own_gpu() as owner:
        require(owner is not None, "Original nonblocking GPU lock unavailable")
    return sample, protected, plan


class PairBudget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < OUTPUT_BYTES, "Output128MiB bound")
        return super().check()


def execute(args):
    sample, protected, plan = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as lock:
        owner = dict(lock, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", read(PRIOR/"checkpoint_audit.json"))
        budget = PairBudget(config, receipt, args.output, owner); budget.maximum_wall = WALL
        budget.check()
        previous, requested, reservation = dict(os.environ), False, None
        result = dict(stage="F0_ONPOLICY_MATCHED_PAIR", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_steps=0, new_rendered_frames=0, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r2", owner=owner,
                gpu=config["expected_gpu"], commit=args.evaluation_commit)
            budget.maximum_wall = min(WALL, reservation["reserved_seconds"]-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/"authorization.json", dict(stage=result["stage"], evaluation_commit=args.evaluation_commit,
                protected_inputs=protected, gpu_lock_owner=owner, pair_plan=identity(PLAN),
                maximum_wall_seconds=budget.maximum_wall, maximum_total_optimizer_updates=6000,
                maximum_output_bytes=OUTPUT_BYTES, maximum_human_future_frames=0,
                original_training_restarted=False, cache_reclaim_attempted=False,
                normalization_refitted=False, final_acceptance=False, user_accepted_checkpoint=None))
            selected = next(r["checkpoint"] for r in read(PRIOR/"checkpoint_audit.json")["records"] if r["robot"] == "q1")
            original = read(TRAINING/"e1024_train_plan.json")
            values = dict(plan=TRAINING/"e1024_train_plan.json", checkpoint=selected["path"],
                morph_source=original["source"], output=args.output, pair_plan=PLAN,
                expected_gpu=config["expected_gpu"], gpu_owner_pid=owner["pid"],
                gpu_owner_ticks=owner["start_ticks"], deadline_ns=deadline)
            command = [runtime["morph_python"], "-u", str(HERE/"student_onpolicy_pair_worker.py")]
            for key, value in values.items():
                command.extend(("--"+key.replace("_", "-"), str(value)))
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded source-matched fit failed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "ONPOLICY_PAIR_TRAINED_PENDING_CLOSED_LOOP"
                and fit.get("actual_optimizer_updates") == 6000
                and [r["variant"] for r in fit["arms"]] == plan["arms"]
                and fit.get("matched_initialization_capacity_and_source_slots") is True
                and all(identity(r["path"]) == r for r in protected), "Paired result or source terminal gate failed")
            result.update(status=fit["status"], source_integrity="PASSED", actual_optimizer_updates=6000,
                arms=[{k:r[k] for k in ("variant", "selected_update", "selected_adapter_unchanged",
                    "initial_validation_risk", "selected_validation_risk")} for r in fit["arms"]],
                fit_result=identity(args.output/"fit_result.json"), natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}"); raise
        finally:
            if reservation and guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                result["cost_settlement"] = settle(LEDGER, run_id=str(args.output), owner=owner,
                    cleanup=read(guard_path)["cleanup"])
            result.update(ended_utc=now(), cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json", result)
            os.environ.clear(); os.environ.update(previous)
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
        print(json.dumps(dict(status="F0_ONPOLICY_PAIR_ADMITTED", protected_files=len(protected), resources=sample)))
    else:
        execute(args)
