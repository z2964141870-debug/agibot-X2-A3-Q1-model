"""Single-use R1 collection and encoder fit after real R0 and paused admission."""

import argparse
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import ROOT, identity, now, require, save
from delivery import read
from interim_review import GIB, TRAINING
from memory_watch import memory
from paused_preview import Budget, preflight
from paused_runtime import assert_owner, compute_pids, own_gpu, paused_state
from preview_review import environment, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run
from student_cost import reserve, settle
from student_rollouts import SPLIT_SHA, collection_jobs, collect


PLAN_SHA = "a44441c977e1afb6dfc49f1f1ccfba8bcbd2cb97eb377b35a709e0b9c87f79d6"
PROFILE_SHA = "1ee3c84482c3d0f8731648046991c7cb5cc0fb6d2f4cbc50e2bee34f90aa5385"
CACHE_HELPER_SHA = "698d277663209489fd0045762f121320a442817888687520a724fec9fb51327c"
WALL_SECONDS = 1740
OUTPUT_BYTES = GIB
PROJECT_BYTES = 500_000_000_000
NEW_CODE = ("student_training.py", "student_training_worker.py", "student_rollouts.py",
            "student_cost.py", "student_fit.py", "student_models.py", "student_inputs.py", "student_data.py")


def profile_gate(root):
    result, auth = read(root/"result.json"), read(root/"authorization.json")
    require(result.get("stage") == "SAVED32K_EVENT_PROFILE" and result.get("status") == "COMPLETED"
            and result.get("source_integrity") == "PASSED" and result.get("new_policy_calls") == 512
            and result.get("new_physical_transitions") == result.get("new_training_updates") == 0,
            "R0 profiler must actually complete before new training")
    cases = result.get("cases", [])
    expected = {(event, method, horizon) for event in ("stop", "late_burst")
                for method, horizon in (("masked", 10), ("damped_cv", 4))}
    require(len(cases) == 4 and {(row["event"], row["method"], row["horizon_frames"]) for row in cases} == expected
            and all(row.get("status") == "PROFILE_ACTION_PARITY_PASSED" and row.get("calls") == 128
                    and row.get("maximum_absolute_action_difference", float("inf")) <= 2e-5
                    for row in cases), "R0 saved-state packing/action parity is incomplete")
    records = [identity(root/name) for name in ("result.json", "authorization.json")]
    for event, method, horizon in sorted(expected):
        path = root/f"{event}_{method}_h{horizon}.guard.json"
        value = read(path)
        require(value.get("status") == "WORKER_COMPLETED" and value.get("cleanup", {}).get("worker_reaped") is True
                and value["cleanup"].get("worker_return_code") == 0, "R0 owned worker was not cleanly reaped")
        records.append(identity(path))
    records += auth["saved_trace_inputs"]+auth["controller_inputs"]
    require(all(identity(row["path"]) == row for row in records), "A protected R0 input or receipt changed")
    return records


def cache_boundary(config, receipt, owner, output):
    paused_state(config, receipt)
    assert_owner(owner["pid"], owner["start_ticks"])
    require(not compute_pids(), "Cache boundary requires an idle admitted GPU")
    before = memory(Path("/sys/fs/cgroup/memory"))
    value = dict(stage="ONCE_BEFORE_R1", before=before, attempted=False, files_advised=0,
        advice_range_bytes=0, observed_resident_bytes=0, file_contents_read=False,
        files_deleted=0, global_drop_caches=False, maximum_seconds=15, maximum_advice_bytes=8*GIB)
    if before["total_cache_bytes"] <= 4*GIB:
        value.update(status="BELOW_4GIB_BOUNDARY", after=before)
    else:
        helper_path = ROOT/"script/project_tools/reclaim_q1_file_cache.py"
        require(identity(helper_path)["sha256"] == CACHE_HELPER_SHA, "Audited cache primitives changed")
        spec = importlib.util.spec_from_file_location("q1_student_cache_primitives", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        started = time.monotonic()
        deadline = started+15
        protected, busy, inaccessible = helper.active_files(ROOT, deadline)
        require(not busy, "Other project work prevents boundary page advice")
        value.update(attempted=True, protected_inodes=len(protected), unreadable_init=inaccessible)
        files = set()
        reason = "PROJECT_SCAN_COMPLETED"
        for path, info in helper.iter_candidates(ROOT, protected, deadline):
            offset = 0
            while offset < info.st_size:
                if time.monotonic() >= deadline or value["advice_range_bytes"] >= 8*GIB:
                    reason = "BOUNDED_WALL_OR_ADVICE_LIMIT"
                    break
                length = min(helper.CHUNK, info.st_size-offset, 8*GIB-value["advice_range_bytes"])
                resident, advised = helper.release(path, info, protected, offset, length, ROOT)
                if advised:
                    value["advice_range_bytes"] += length
                    value["observed_resident_bytes"] += resident
                    files.add(str(path))
                offset += length
            if reason != "PROJECT_SCAN_COMPLETED":
                break
        value.update(status=reason, elapsed_seconds=time.monotonic()-started, files_advised=len(files),
            after=memory(Path("/sys/fs/cgroup/memory")),
            attribution="Observed before/after deltas are not exclusively attributable to advice")
    save(output/"cache_boundary.json", value)
    require(value["after"]["oom_kill"] == before["oom_kill"] == 0 and not value["after"]["under_oom"],
            "OOM changed at the independent task boundary")
    return value


class TrainingBudget(Budget):
    def __init__(self, *args):
        super().__init__(*args)
        self.project_time, self.project = 0., None
        self.maximum_wall = WALL_SECONDS

    def check(self):
        require(time.monotonic()-self.started < self.maximum_wall, "R1 single phase1740s/remaining budget bound reached")
        require(allocated_tree(self.output)["unique_allocated_bytes"] < OUTPUT_BYTES,
                "R1 output1GiB bound reached")
        if self.project is None or time.monotonic()-self.project_time >= 60:
            self.project, self.project_time = allocated_tree(ROOT), time.monotonic()
        require(self.project["unique_allocated_bytes"] < PROJECT_BYTES, "Project500GB bound reached")
        return dict(super().check(), project_storage=self.project)


def train_preflight(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require(identity(args.student_plan)["sha256"] == PLAN_SHA and identity(args.source_split)["sha256"] == SPLIT_SHA,
            "Published student plan or source split changed")
    require(args.student_plan == HERE/"student_training_plan_20261107.json"
            and args.source_split == ROOT/"data/manifests/q1_teleop_forecast_split_20261106.json",
            "Use the original independently published plan and split")
    require({str(HERE/name) for name in NEW_CODE} <= {row["path"] for row in code},
            "Executable student workflow is absent from its new code manifest")
    require(not {str(HERE/name) for name in NEW_CODE} & {row["path"] for row in sources},
            "New student code overlaps frozen training sources")
    require(args.cost_ledger == ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json",
            "Do not reset or choose another cumulative GPU cost ledger")
    profile_inputs = profile_gate(args.profile_root)
    protected_manifest = ROOT/"data/manifests/q1_event_profile_code_20261107.json"
    require(identity(protected_manifest)["sha256"] == PROFILE_SHA, "Original226-file profile manifest changed")
    profiles = read(protected_manifest)
    profile_code = validate_code_manifest(protected_manifest, PROFILE_SHA, profiles["evaluation_commit"])
    require(not {str(HERE/name) for name in NEW_CODE} & {row["path"] for row in profile_code},
            "Student workflow entered the original profile inputs")
    helper = identity(ROOT/"script/project_tools/reclaim_q1_file_cache.py")
    require(helper["sha256"] == CACHE_HELPER_SHA, "Audited cache helper changed")
    controls = [identity(path) for path in (args.config, args.runtime_config, args.pause_receipt,
        args.code_manifest, args.student_plan, args.source_split, protected_manifest)]
    controls += profile_inputs+[helper]
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+OUTPUT_BYTES < PROJECT_BYTES,
            "New task would exceed project500GB storage budget")
    jobs = collection_jobs(read(args.source_split))
    return config, runtime, plan, frozen, receipt, sources, audit, code, controls, jobs


def execute(args):
    if args.child:
        assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
        config, runtime = read(args.config), read(args.runtime_config)
        plan, frozen = read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"])
        return collect(args, config, runtime, plan, frozen)
    config, runtime, plan, frozen, receipt, sources, audit, code, controls, jobs = train_preflight(args)
    with own_gpu() as owner:
        require(owner is not None, "Cannot identify the new student GPU owner")
        owner = dict(owner, boot=config["expected_boot"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"collection_plan.json", dict(jobs=jobs, test_sources_read=False))
        cache_boundary(config, receipt, owner, args.output)
        budget = TrainingBudget(config, receipt, args.output, owner)
        budget.check()
        started_ns = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r1", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started_ns)
        budget.maximum_wall = min(WALL_SECONDS, max(0, reservation["reserved_seconds"]-5))
        args.deadline_ns = started_ns+max(0, budget.maximum_wall-5)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R1_SHORT_PREVIEW_ENCODER_DISTILLATION",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code, training_source_inputs=sources,
            controller_inputs=controls, gpu_lock_owner=owner, cost_reservation=reservation,
            maximum_wall_seconds=budget.maximum_wall, original_phase_protection_seconds=1800,
            maximum_physical_steps=sum(row["maximum_steps"] for row in jobs), maximum_output_bytes=OUTPUT_BYTES,
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R1_SHORT_PREVIEW_ENCODER_DISTILLATION", status="RUNNING", controller=owner,
            started_utc=now(), closed_loop_acceptance="PENDING", user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {value:signal.signal(value, stop) for value in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"collection_fit.guard.json"
        try:
            os.environ.update(environment(runtime))
            command = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child"]
            for key in ("config", "runtime_config", "output", "student_plan", "evaluation_commit"):
                command.extend(("--"+key.replace("_", "-"), str(getattr(args, key))))
            for key, value in (("gpu-owner-pid", owner["pid"]), ("gpu-owner-ticks", owner["start_ticks"]),
                               ("deadline-ns", args.deadline_ns)):
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            phase = guarded_run(command, ROOT, budget, guard_path, args.output/"collection_fit.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(phase["status"] == "WORKER_COMPLETED" and phase["cleanup"]["worker_reaped"]
                    and phase["cleanup"]["worker_return_code"] == 0, "Bounded R1 worker did not complete")
            require(all(identity(row["path"]) == row for row in controls+sources+code),
                    "Student sources, frozen teacher sources or control receipts changed")
            value = read(args.output/"fit_result.json")
            require(value["status"] == "TRAINED_PENDING_CLOSED_LOOP" and value["actual_updates"] > 0,
                    "No actual completed student fit")
            result.update(status="TRAINED_PENDING_CLOSED_LOOP", source_integrity="PASSED",
                actual_optimizer_updates=value["actual_updates"], fit_result=identity(args.output/"fit_result.json"),
                original_training_restarted=False, natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if guard_path.exists():
                    terminal = read(guard_path)
                    cleanup = terminal.get("cleanup", {})
                    if cleanup.get("worker_reaped") is True:
                        result["cost_settlement"] = settle(args.cost_ledger, run_id=str(args.output),
                            owner=owner, cleanup=cleanup)
                result["ended_utc"] = now()
                save(args.output/"result.json", result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "student-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("pause-receipt", "audit-root", "code-manifest", "source-split", "profile-root", "cost-ledger"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.add_argument("--deadline-ns", type=int)
    args = parser.parse_args(argv)
    if not args.child and not all(getattr(args, key) for key in ("pause_receipt", "audit_root", "code_manifest",
            "source_split", "profile_root", "cost_ledger", "manifest_sha256")):
        parser.error("Controller needs published sources, completed R0, paused identity and the original cost ledger")
    if args.child and not all((args.gpu_owner_pid, args.gpu_owner_ticks, args.deadline_ns)):
        parser.error("Owned child needs PID/start ticks and the original monotonic deadline")
    execute(args)


if __name__ == "__main__":
    main()
