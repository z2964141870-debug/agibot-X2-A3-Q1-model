"""Independent R2 fit using preserved R1 states after its completed review."""

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
from interim_review import GIB, TRAINING
from paused_preview import Budget, preflight
from paused_runtime import own_gpu
from preview_review import environment
from runtime_guard import allocated_tree, guarded_run
from student_cost import reserve, settle
from student_review import training_evidence
from student_rollouts import SPLIT_SHA, collection_jobs, validate_records
from student_training import cache_boundary


PLAN_SHA = "b73365397cabb29968ab244f3be546ef809b78258973843f2c7245d1fc1fe2c0"
WALL_SECONDS = 1740
NEW_CODE = ("student_causal_training.py", "student_causal_worker.py", "student_causal_fit.py")


def saved_analysis_gate(root, parent):
    summary, audit = read(root/"summary.json"), read(root/"audit.json")
    require(summary.get("stage") == "R1_MATCHED_SAVED_REVIEW_ANALYSIS" and summary.get("total_cases") == 23
        and len(summary.get("rows", [])) == 23 and audit.get("status") == "PASSED"
        and audit.get("source_inputs_unchanged") is True and summary.get("new_inference_calls") == 0
        and summary.get("new_physical_transitions") == summary.get("new_training_updates") == 0,
        "Wait for real completed R1 matched saved analysis, not fitting loss or a plan")
    inputs = summary.get("inputs", [])
    require(inputs and audit.get("source_files") == len(inputs)
        and len({r["path"] for r in inputs}) == len(inputs)
        and all(identity(r["path"]) == r for r in inputs), "Saved R1 analysis sources changed")
    recorded = {r["path"]: r for r in inputs}
    require(all(recorded.get(r["path"]) == r for r in parent), "Analysis belongs to another trained student")
    source = Path(inputs[0]["path"])
    result = read(source)
    require(source.name == "result.json" and result.get("status") == "COMPLETED"
        and result.get("source_integrity") == "PASSED" and result.get("physical_cases") == 23
        and result.get("stage") == "R1_MATCHED_CLOSED_LOOP_CONTINUATION"
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "R1 matched review must complete and release its controller before R2")
    return [identity(root/name) for name in ("summary.json", "audit.json")]+inputs


def reused_dataset(root, jobs, *, load=False):
    import numpy as np

    plan, receipt, collection = (read(root/name) for name in
        ("collection_plan.json", "dataset_receipt.json", "collection_result.json"))
    require(plan.get("jobs") == jobs and receipt.get("status") == "PACKING_ACTION_PARITY_PASSED"
        and receipt.get("test_sources_read") is False and collection.get("status") == "PASSED"
        and collection.get("test_sources_read") is False and len(collection.get("clips", [])) == len(jobs),
        "Preserve original owned collection jobs, labels and split provenance")
    records = [identity(root/name) for name in
        ("collection_plan.json", "dataset_receipt.json", "collection_result.json")]
    traces = {}
    for expected, row in enumerate(collection["clips"]):
        record = row["trace"]
        require(row.get("clip_id") == expected and row.get("clip") == jobs[expected]["clip"]
            and row.get("split") == jobs[expected]["split"]
            and Path(record["path"]) == root/f"collection_{expected:02d}.npz"
            and identity(record["path"]) == record, "Preserved physical trace changed its identity")
        with np.load(record["path"], allow_pickle=False) as values:
            traces[expected] = dict(values)
        trace = traces[expected]
        require(set(trace) == {"actual", "canonical_action"}
            and trace["actual"].shape == (len(trace["canonical_action"])+1, 79)
            and trace["canonical_action"].ndim == 2 and trace["canonical_action"].shape[1] == 29
            and all(np.isfinite(value).all() for value in trace.values()),
            "Preserved physical trace has invalid state/action arrays")
        records.append(record)
    values, checked = {}, {}
    for split, expected_sources in (("train", 12), ("validation", 6)):
        record = receipt["files"][split]
        require(Path(record["path"]) == root/(split+"_data.npz") and identity(record["path"]) == record,
            "Use the preserved actual-state dataset, not new reference-state labels")
        with np.load(record["path"], allow_pickle=False) as arrays:
            data = dict(arrays)
        checked[split] = validate_records(data, split, jobs, traces)
        require(checked[split]["source_clips"] == expected_sources and checked[split]["records"] >= 32,
            "Preserve all original training or validation source groups")
        if load:
            values[split] = {key: data[key] for key in ("f0", "history", "teacher_action", "clip_id")}
        records.append(record)
    require(all(identity(r["path"]) == r for r in records), "R1 datasets changed during reuse checks")
    return dict(records=records, validation=checked, actual_states_and_labels_reused=True,
        new_physical_transitions=0, test_sources_read=False), values


class CausalBudget(Budget):
    def __init__(self, *args):
        super().__init__(*args)
        self.maximum_wall, self.project_time, self.project = WALL_SECONDS, 0., None

    def check(self):
        require(time.monotonic()-self.started < self.maximum_wall, "R2 fit phase1740s/remaining budget reached")
        require(allocated_tree(self.output)["unique_allocated_bytes"] < GIB, "R2 output1GiB bound reached")
        if self.project is None or time.monotonic()-self.project_time >= 60:
            self.project, self.project_time = allocated_tree(ROOT), time.monotonic()
        require(self.project["unique_allocated_bytes"] < 500_000_000_000, "Project500GB bound reached")
        return dict(super().check(), project_storage=self.project)


def train_preflight(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require(args.training_root == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R1_20261107"
        and args.causal_plan == HERE/"student_causal_plan_20261107.json"
        and identity(args.causal_plan)["sha256"] == PLAN_SHA
        and identity(args.source_split)["sha256"] == SPLIT_SHA
        and args.source_split == ROOT/"data/manifests/q1_teleop_forecast_split_20261106.json",
        "Use the published R2 plan and original R1 data/split")
    require(args.cost_ledger == ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json",
        "Retain the original cumulative cost ledger")
    require({str(HERE/name) for name in NEW_CODE} <= {r["path"] for r in code}
        and not {str(HERE/name) for name in NEW_CODE} & {r["path"] for r in sources},
        "New causal workflow is missing or overlaps frozen training sources")
    parent = training_evidence(args.training_root)
    analysis = saved_analysis_gate(args.r1_analysis_root, parent)
    jobs = collection_jobs(read(args.source_split))
    dataset, _ = reused_dataset(args.training_root, jobs)
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+GIB < 500_000_000_000,
        "New task would exceed project500GB storage budget")
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt,
        args.code_manifest, args.causal_plan, args.source_split, TRAINING/"e1024_train_plan.json")]
    controls += parent+analysis+dataset["records"]
    unique = {}
    for record in controls:
        require(record["path"] not in unique or unique[record["path"]] == record,
            "Conflicting protected input identities")
        unique[record["path"]] = record
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def worker_command(args, runtime, audit, owner, deadline_ns):
    selected = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    plan = read(TRAINING/"e1024_train_plan.json")
    values = {"plan": TRAINING/"e1024_train_plan.json", "checkpoint": selected["path"],
        "morph-source": plan["source"], "training-root": args.training_root,
        "output": args.output, "causal-plan": args.causal_plan,
        "expected-gpu": owner["gpu"], "gpu-owner-pid": owner["pid"],
        "gpu-owner-ticks": owner["start_ticks"], "deadline-ns": deadline_ns}
    command = [runtime["morph_python"], "-u", str(HERE/"student_causal_worker.py")]
    for key, value in values.items():
        command.extend(("--"+key, str(value)))
    return command


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = train_preflight(args)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"reused_dataset.json", dataset)
        boundary = cache_boundary(config, receipt, owner, args.output)
        boundary["stage"] = "ONCE_BEFORE_R2"
        save(args.output/"cache_boundary.json", boundary)
        budget = CausalBudget(config, receipt, args.output, owner)
        budget.check()
        started_ns = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r2", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started_ns)
        budget.maximum_wall = min(WALL_SECONDS, max(0, reservation["reserved_seconds"]-5))
        deadline_ns = started_ns+max(0, budget.maximum_wall-5)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R2_F0_HISTORY_ENCODER_DISTILLATION",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code, training_source_inputs=sources,
            controller_inputs=controls, gpu_lock_owner=owner, cost_reservation=reservation,
            maximum_wall_seconds=budget.maximum_wall, original_phase_protection_seconds=1800,
            maximum_output_bytes=GIB, new_physical_transitions=0, human_real_future_frames=0,
            teacher_role="privileged labels on preserved real robot states", task_scope="full_body_precise_imitation",
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R2_F0_HISTORY_ENCODER_DISTILLATION", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, closed_loop_acceptance="PENDING",
            user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {value:signal.signal(value, stop) for value in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"fit.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            save(args.output/"result.json", result)
            guard = guarded_run(worker_command(args, runtime, audit, owner, deadline_ns), ROOT, budget,
                guard_path, args.output/"fit.log", check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded R2 fit did not complete")
            require(all(identity(r["path"]) == r for r in controls+sources+code),
                "R2 sources, R1 evidence or frozen original inputs changed")
            fit = read(args.output/"fit_result.json")
            require(fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP" and fit.get("actual_updates", 0) > 0
                and fit.get("human_real_future_frames") == fit.get("new_physical_transitions") == 0,
                "No completed causal student fit")
            result.update(status="TRAINED_PENDING_CLOSED_LOOP", source_integrity="PASSED",
                actual_optimizer_updates=fit["actual_updates"], fit_result=identity(args.output/"fit_result.json"),
                human_real_future_frames=0, natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if guard_path.exists():
                    cleanup = read(guard_path).get("cleanup", {})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "pause-receipt", "audit-root", "code-manifest",
            "training-root", "causal-plan", "source-split", "cost-ledger", "r1-analysis-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
