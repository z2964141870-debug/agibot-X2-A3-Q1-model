"""One independent H4-only continuation under the original paused identity and ledger."""

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
from student_causal_training import reused_dataset
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_review import training_evidence
from student_rollouts import collection_jobs
from student_same_state import unique
from student_training import TrainingBudget

DATA = ROOT/"data/supermimic_for_q1"
PARENT = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R1_20261107"
ACTION = DATA/"Q1_LOW_LATENCY_CKPT32000_CPU_ACTION01_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_20261108"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
PLAN = HERE/"student_h4_plan_20261108.json"
FILES = ("student_h4_training.py", "student_h4_worker.py", "student_h4_fit.py",
         "student_h4_plan_20261108.json", "test_student_h4.py")


def admission(args):
    require(args.output == OUTPUT and not OUTPUT.exists(), "Use the fresh H4_SPECIALIST01 output only")
    require(len(args.evaluation_commit) == 40 and set(args.evaluation_commit) <= set("0123456789abcdef"),
            "Publish the independent H4 workflow before admission")
    sample = resources(True)
    config, receipt = read(args.config), read(args.pause_receipt)
    paused_state(config, receipt)
    require(identity(args.manifest)["sha256"] == args.manifest_sha256, "H4 manifest changed")
    manifest = read(args.manifest)
    require(manifest.get("schema") == "q1_h4_specialist_code_v1"
        and len(manifest["files"]) == len(FILES)
        and {r["path"] for r in manifest["files"]} == {str(HERE/name) for name in FILES},
        "Unexpected independent H4 source scope")
    plan = read(PLAN)
    require(plan.get("schema") == "q1_h4_specialist_plan_v1" and plan["request_horizon"] == 4
        and plan["fit"] == dict(seed=2026110801, learning_rate=.0001, maximum_updates=4000,
            batch_size=128, gradient_norm_limit=1., validation_every_updates=250,
            early_stop_stale_validations=6)
        and plan["resource"]["maximum_wall_seconds"] == 600
        and plan["resource"]["ledger_stage"] == "r1", "Use the preregistered H4 plan")
    result, auth = read(ACTION/"result.json"), read(ACTION/"authorization.json")
    require(result.get("status") == "CPU_STATE_ACTION_PARITY_PASSED"
        and result.get("source_integrity") == "PASSED" and result.get("new_policy_calls") == 64
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Prior CPU action gate must have completed and released its owned workers")
    prior = [r for r in auth["protected_inputs"] if r["path"] != str(LEDGER)]
    require(not {str(HERE/name) for name in FILES} & {r["path"] for r in prior},
            "H4 additions overlap protected earlier execution inputs")
    jobs = collection_jobs(read(ROOT/"data/manifests/q1_teleop_forecast_split_20261106.json"))
    dataset, _ = reused_dataset(PARENT, jobs)
    for filename, key in (("fit_result.json", "r1_fit_sha256"), ("selected.pt", "initial_checkpoint_sha256"),
            ("train_data.npz", "train_data_sha256"), ("validation_data.npz", "validation_data_sha256")):
        require(identity(PARENT/filename)["sha256"] == plan[key], "Original R1 identity differs: "+filename)
    parent = training_evidence(PARENT)
    controls = [identity(p) for p in (ACTION/"result.json", ACTION/"authorization.json",
        ACTION/"checkpoint_audit.json", args.config, args.runtime_config, args.pause_receipt, args.manifest,
        TRAINING/"e1024_train_plan.json")]
    protected = unique(prior+parent+controls+dataset["records"]+manifest["files"])
    require(all(identity(r["path"]) == r for r in protected), "A protected H4 input changed")
    ledger = checked(read(LEDGER))
    r1 = charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == "r1"]))
    require(charged(ledger) == plan["resource"]["expected_cumulative_seconds"] == 2492
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r1"]-r1 >= 600 and charged(ledger)+600 <= LIMIT,
        "Retain the original2492 seconds and resolve stage/cumulative reservations")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+128*(1 << 20) < 500_000_000_000,
            "Private project500GB bound reached")
    with own_gpu() as owner:
        require(owner is not None, "GPU idle-lock admission failed")
    return sample, protected, dataset


class H4Budget(TrainingBudget):
    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 128*(1 << 20),
                "H4 output128MiB bound reached")
        return super().check()


def execute(args):
    sample, protected, dataset = admission(args)
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", read(ACTION/"checkpoint_audit.json"))
        save(args.output/"reused_dataset.json", dataset)
        budget = H4Budget(config, receipt, args.output, owner)
        budget.maximum_wall = 600
        budget.check()
        reservation, previous, requested = None, dict(os.environ), False
        result = dict(stage="H4_SPECIALIST_CONTINUATION", status="RUNNING", controller=owner,
            started_utc=now(), admission=sample, new_physical_steps=0, user_accepted_checkpoint=None)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            reservation = reserve(LEDGER, run_id=str(args.output), stage="r1", owner=owner,
                                  gpu=config["expected_gpu"], commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            budget.maximum_wall = min(600, reservation["reserved_seconds"]-5)
            deadline = time.monotonic_ns()+int(budget.maximum_wall-5)*1_000_000_000
            save(args.output/"authorization.json", dict(stage="H4_SPECIALIST_CONTINUATION",
                evaluation_commit=args.evaluation_commit, protected_inputs=protected,
                gpu_lock_owner=owner, maximum_wall_seconds=budget.maximum_wall,
                maximum_optimizer_updates=4000, maximum_output_bytes=128*(1 << 20),
                maximum_human_future_frames=4, original_training_restarted=False,
                normalization_refitted=False, cache_reclaim_attempted=False,
                new_physical_steps=0, final_acceptance=False, user_accepted_checkpoint=None))
            audit = read(ACTION/"checkpoint_audit.json")
            selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
            original = read(TRAINING/"e1024_train_plan.json")
            values = dict(plan=TRAINING/"e1024_train_plan.json", checkpoint=selected["path"],
                morph_source=original["source"], training_root=PARENT, output=args.output, h4_plan=PLAN,
                expected_gpu=config["expected_gpu"], gpu_owner_pid=owner["pid"],
                gpu_owner_ticks=owner["start_ticks"], deadline_ns=deadline)
            command = [runtime["morph_python"], "-u", str(HERE/"student_h4_worker.py")]
            for key, value in values.items():
                command.extend(("--"+key.replace("_", "-"), str(value)))
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded H4 fit failed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
                and 0 < fit.get("actual_updates", 0) <= 4000
                and fit.get("allowed_request_horizons") == [4]
                and all(identity(r["path"]) == r for r in protected), "H4 result or sources changed")
            result.update(status="TRAINED_PENDING_CLOSED_LOOP", source_integrity="PASSED",
                actual_optimizer_updates=fit["actual_updates"], selected_update=fit["selected_update"],
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
        sample, protected, dataset = admission(args)
        print(json.dumps(dict(status="H4_SPECIALIST_ADMITTED", protected_files=len(protected),
            resources=sample, dataset=dataset["validation"], new_optimizer_updates=0)))
    else:
        execute(args)
