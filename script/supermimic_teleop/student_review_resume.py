"""Continue the R1 review after its media-interpreter error, preserving its replay."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from interim_review import GIB, fixed_rows, media
from paused_preview import preflight
from paused_runtime import own_gpu
from preview_review import environment, validate_code_manifest
from runtime_guard import guarded_run
from stream_events import EventSpec, EventTimeline
from student_review import ReviewBudget, folder, replay, review_cases, training_evidence, validate_timing


def phases_for(case):
    event, variant, horizon, clip = case
    video = clip == FIXED[0] and ((event == "nominal" and (variant, horizon) in
        (("teacher", 10), ("teacher", 4), ("student", 4)))
        or (event in ("stop", "reverse", "dropout", "late_burst") and variant == "student"))
    return ("replay", "media") if video else ("replay",)


def child_command(args, runtime, owner, case, phase):
    require(phase in ("replay", "media"), "Unknown review phase")
    event, variant, horizon, clip = case
    return [runtime["primebot_python"], "-u", str(Path(__file__).resolve()), "--child", phase,
        "--config", str(args.config), "--runtime-config", str(args.runtime_config),
        "--output", str(args.output), "--training-root", str(args.training_root),
        "--evaluation-commit", args.evaluation_commit, "--event-kind", event,
        "--variant", variant, "--horizon", str(horizon), "--clip", clip,
        "--gpu-owner-pid", str(owner["pid"]), "--gpu-owner-ticks", owner["start_ticks"]]


def spent_seconds(result):
    require(result.get("status") == "FAILED" and result.get("stage") == "R1_MATCHED_CLOSED_LOOP"
        and result.get("error") == "KeyError: 'python'", "Reuse only the diagnosed media-interpreter exit")
    elapsed = (datetime.fromisoformat(result["ended_utc"])-datetime.fromisoformat(result["started_utc"])).total_seconds()
    require(0 < elapsed < 2700, "Prior review exhausted or changed its original wall budget")
    return elapsed


def previous_evidence(args, parent, sources, audit, runtime, frozen):
    root = args.previous_review
    require(root.is_absolute() and root.resolve() == root and root != args.output
        and root.is_relative_to(ROOT/"data/supermimic_for_q1"), "Require the preserved independent failed review")
    result, auth = read(root/"result.json"), read(root/"authorization.json")
    spent = spent_seconds(result)
    require(not Path("/proc", str(result["controller"]["pid"])).exists()
        and auth.get("stage") == "R1_MATCHED_CLOSED_LOOP" and auth.get("maximum_wall_seconds") == 2700
        and auth.get("phase_wall_seconds") == 1800 and auth.get("new_optimizer_updates") == 0
        and auth.get("trained_student_inputs") == parent and auth.get("training_source_inputs") == sources
        and read(root/"checkpoint_audit.json") == audit, "Prior review provenance, cleanup or budget differs")
    protected = auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]+parent
    require(all(identity(r["path"]) == r for r in protected), "Prior review source or trained student changed")
    first = "nominal_teacher_h10_"+FIXED[0]
    guard_path = root/(first+"_replay.guard.json")
    guard = read(guard_path)
    require(result.get("phases") == [dict(name=first+"_replay", guard=identity(guard_path))]
        and result.get("cases") == [] and result.get("phase") == first+"_media"
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(guard["worker_pid"])).exists(), "Reuse only the completed and reaped first replay")
    directory = root/first
    review = read(directory/"review/result.json")
    require(review.get("status") == "PHYSICS_PASSED" and review.get("policy_worker_reaped") is True
        and review.get("policy_worker_return_code") == 0 and review.get("cuda_initialized") is False
        and review.get("render_xml") == identity(runtime["xml"]) and len(review.get("cases", [])) == 1,
        "Saved first replay lacks clean runtime and scene provenance")
    row = review["cases"][0]
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == FIXED[0])
    selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    require(review.get("checkpoint") == selected and row.get("clip") == FIXED[0] and row.get("seed") == 0
        and row.get("event") == "nominal" and row.get("variant") == "teacher" and row.get("horizon_frames") == 10
        and row.get("interface") == "human" and row.get("evaluation_randomization") is False
        and row.get("student_checkpoint") == identity(args.training_root/"selected.pt")
        and row.get("complete") is True and row.get("failure_frame") is None
        and row.get("valid_steps") == row.get("attempted_steps") == row.get("reference_steps") > 0
        and Path(row["trace"]) == directory/"review"/(FIXED[0]+"_s0")/"rollout.npz"
        and all(row.get(k) == item[s]["sha256"] for k, s in
            (("reference_sha256", "q1_primebot"), ("native_reference_sha256", "q1_morph"), ("human_sha256", "human"))),
        "First physical replay has another policy, reference or incomplete prefix")
    import numpy as np
    with np.load(row["trace"], allow_pickle=False) as trace:
        require(len(trace["actual"]) == len(trace["reference"]) == row["reference_steps"]+1
            and trace["canonical_action"].shape == (row["reference_steps"], 29)
            and all(np.isfinite(trace[k]).all() for k in trace.files), "Saved first replay has invalid arrays")
    timing = directory/"stream_timing.jsonl"
    require(review.get("stream_telemetry") == identity(timing), "First replay timing changed")
    values = [json.loads(line) for line in timing.read_text().splitlines()]
    require(len(values) == row["attempted_steps"], "First replay timing is truncated")
    timeline = EventTimeline(row["original_paired_reference"]["frames"], EventSpec("nominal", 50, 10, 0))
    for frame, value in enumerate(values):
        validate_timing(value, frame, 10, "teacher", timeline)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    require(files and all(not p.is_symlink() and p.stat().st_size < 8*1024**2 for p in files)
        and sum(p.stat().st_size for p in files) < 3*GIB, "Prior evidence exceeds its original file bounds")
    records = [identity(p) for p in files]+protected
    return spent, directory, records


def copy_first(source, output):
    destination = output/source.name
    originals = [identity(p) for p in source.rglob("*") if p.is_file()]
    shutil.copytree(source, destination)
    for before in originals:
        copied = identity(destination/Path(before["path"]).relative_to(source))
        require((before["bytes"], before["sha256"]) == (copied["bytes"], copied["sha256"]), "Replay copy changed")
    review = read(destination/"review/result.json")
    for row in review["cases"]:
        row["trace"] = str(destination/Path(row["trace"]).relative_to(source))
        row["reused_physical_source"] = originals
        save(Path(row["trace"]).parent/"result.json", row)
    review["stream_telemetry"] = identity(destination/"stream_timing.jsonl")
    review["reused_physical_source"] = originals
    save(destination/"review/result.json", review)
    require(all(identity(r["path"]) == r for r in originals), "Original replay was modified by copying")


class ResumeBudget(ReviewBudget):
    def __init__(self, config, receipt, output, owner, spent):
        super().__init__(config, receipt, output, owner)
        self.spent = spent

    def check(self):
        require(self.spent+time.monotonic()-self.started < 2700, "Cumulative R1 review2700s bound reached")
        return super().check()


def execute(args):
    if args.child:
        config, runtime = read(args.config), read(args.runtime_config)
        if args.child == "media":
            return media(SimpleNamespace(output=folder(args)), config, runtime)
        from interim_review import TRAINING
        return replay(args, config, runtime, read(TRAINING/"e1024_train_plan.json"), read(config["frozen_data"]))
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require({str(HERE/n) for n in ("student_review_resume.py", "student_review_worker.py")} <= {r["path"] for r in code},
        "Published review continuation is missing")
    parent = training_evidence(args.training_root)
    spent, first, preserved = previous_evidence(args, parent, sources, audit, runtime, frozen)
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt, args.code_manifest)]
    with own_gpu() as owner:
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        copy_first(first, args.output)
        save(args.output/"authorization.json", dict(stage="R1_MATCHED_CLOSED_LOOP_CONTINUATION",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code, training_source_inputs=sources,
            controller_inputs=controls, trained_student_inputs=parent, preserved_failed_review_inputs=preserved,
            maximum_cumulative_review_wall_seconds=2700, prior_review_wall_seconds=spent,
            phase_wall_seconds=1800, maximum_output_bytes=3*GIB, cases=review_cases(), gpu_lock_owner=owner,
            reused_physical_cases=1, new_physical_cases=22, new_optimizer_updates=0, accepted_checkpoint=None,
            hardware_end_to_end_measured=False))
        result = dict(status="RUNNING", stage="R1_MATCHED_CLOSED_LOOP_CONTINUATION", controller=owner,
            started_utc=now(), phases=[], cases=[], reused_physical_cases=1,
            natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
        budget, previous, requested = ResumeBudget(config, receipt, args.output, owner, spent), dict(os.environ), False

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            for index, case in enumerate(review_cases()):
                args.event_kind, args.variant, args.horizon, args.clip = case
                for phase in phases_for(case):
                    if index == 0 and phase == "replay":
                        continue
                    require(all(identity(r["path"]) == r for r in parent+controls+preserved), "Preserved R1 evidence changed")
                    validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                    name = f"{args.event_kind}_{args.variant}_h{args.horizon}_{args.clip}_{phase}"
                    result["phase"] = name
                    save(args.output/"result.json", result)
                    budget.phase_started = time.monotonic()
                    guard = guarded_run(child_command(args, runtime, owner, case, phase), ROOT, budget,
                        args.output/(name+".guard.json"), args.output/(name+".log"), check_interval_s=.5,
                        grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name, guard=identity(args.output/(name+".guard.json"))))
                    require(guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"] is True
                        and guard["cleanup"]["worker_return_code"] == 0, "R1 continuation phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in parent+controls+preserved+sources+code), "Review source changed")
            result.update(status="COMPLETED", source_integrity="PASSED", physical_cases=23, new_physical_cases=22,
                new_optimizer_updates=0, quality_conclusion="PENDING_SAVED_ANALYSIS",
                cumulative_review_wall_seconds=spent+time.monotonic()-budget.started)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["ended_utc"] = now()
            save(args.output/"result.json", result)
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "training-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("pause-receipt", "audit-root", "code-manifest", "previous-review"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--child", choices=("replay", "media"))
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"))
    parser.add_argument("--variant", choices=("teacher", "student"))
    parser.add_argument("--horizon", type=int, choices=(1, 4, 10))
    parser.add_argument("--clip", choices=FIXED)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    args = parser.parse_args()
    if not args.child and not all((args.pause_receipt, args.audit_root, args.code_manifest,
                                  args.manifest_sha256, args.previous_review)):
        parser.error("Controller needs the preserved failed review and published paused32k admission")
    if args.child and not all((args.event_kind, args.variant, args.horizon, args.clip,
                              args.gpu_owner_pid, args.gpu_owner_ticks)):
        parser.error("Child needs the exact case and GPU owner identity")
    execute(args)
