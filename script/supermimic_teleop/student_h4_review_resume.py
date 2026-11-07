"""Preserve the first H4 replay and continue after the isolated media-path error."""

import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from datetime import datetime

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent/"q1_five_stage"),str(HERE.parent/"primebot_q1")]
from common import FIXED,ROOT,identity,now,require,save
from delivery import read
from interim_review import GIB,TRAINING,fixed_rows,media
from paused_preview import preflight
from paused_runtime import compute_pids,own_gpu
from preview_review import environment,runtime_paths
from runtime_guard import guarded_run
from stream_events import EventSpec,EventTimeline
from student_causal_review import CausalReviewBudget,review_cases
from student_cost import LIMIT,STAGES,charged,checked,reserve,settle
from student_h4_review import (LEDGER,OUTPUT as FAILED,PLAN,TRAINED,child_command as original_child,
    matched_baselines,parser)
from student_review import folder,replay,validate_timing
from student_review_resume import copy_first
from student_same_state import unique

OUTPUT = ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_REVIEW_RESUME1_20261108"
STAGE = "H4_SPECIALIST01_MATCHED_CLOSED_LOOP_CONTINUATION"
FAILURE_RECORD = ROOT/"logs/session_records/q1_h4_specialist_review_media_path_failed_20261108.json"
FAILURE_SHA = "3a0a7b2da48bbcd35fd840f0ac3253b2fce33977f05ffd220a30e56aaee79416"


def remaining_phases():
    return [(case,phase) for index,case in enumerate(review_cases()) for phase in ("replay","media")
            if not (index == 0 and phase == "replay")]


def spent_seconds(result,log):
    first = "nominal_student_h4_"+FIXED[0]
    require(result.get("status") == "FAILED" and result.get("stage") == "H4_SPECIALIST01_MATCHED_CLOSED_LOOP"
        and result.get("phase") == first+"_media" and result.get("cases") == []
        and result.get("error") == "RuntimeError: H4 review phase failed: "+first+"_media"
        and "ModuleNotFoundError: No module named 'video'" in log
        and result.get("cost_settlement",{}).get("state") == "SETTLED",
        "Continue only the diagnosed, settled H4 media-path failure")
    elapsed = (datetime.fromisoformat(result["ended_utc"])-datetime.fromisoformat(result["started_utc"])).total_seconds()
    spent = math.ceil(max(elapsed,result["cost_settlement"]["charged_seconds"]))
    require(0 < spent < 1800,"Original H4 review budget is exhausted")
    return spent


def previous_evidence(args,runtime,audit,frozen):
    require(identity(FAILURE_RECORD)["sha256"] == FAILURE_SHA,"The original H4 failure receipt changed")
    failure = read(FAILURE_RECORD)
    require(failure.get("stage") == "H4_REVIEW_MEDIA_PATH_FAILURE"
        and failure.get("status") == "FAILED_PRESERVED"
        and all(identity(r["path"]) == r for r in failure["files"]),"Original failed H4 evidence changed")
    result,auth = (read(FAILED/n) for n in ("result.json","authorization.json"))
    name = "nominal_student_h4_"+FIXED[0]
    spent = spent_seconds(result,(FAILED/(name+"_media.log")).read_text())
    require(spent == failure["conservative_spent_seconds"] == 101,"Keep the original settled H4 review charge")
    require(not Path("/proc",str(result["controller"]["pid"])).exists()
        and auth.get("stage") == result["stage"] and auth.get("maximum_wall_seconds") == 1800
        and auth.get("new_optimizer_updates") == 0 and auth.get("human_real_future_frames") == 4
        and auth.get("selected_update") == 1500 and read(FAILED/"checkpoint_audit.json") == audit,
        "Prior H4 source, cleanup or original budget differs")
    expected = [dict(name=name+"_"+phase,guard=identity(FAILED/(name+"_"+phase+".guard.json")))
                for phase in ("replay","media")]
    require(result.get("phases") == expected,"Unexpected physical work in the failed root")
    for item,status,code in zip(expected,("WORKER_COMPLETED","WORKER_FAILED"),(0,1)):
        guard = read(item["guard"]["path"])
        require(guard.get("status") == status and guard.get("cleanup",{}).get("worker_reaped") is True
            and guard["cleanup"].get("worker_return_code") == code
            and not Path("/proc",str(guard["worker_pid"])).exists(),"Previous H4 workers are not cleanly reaped")
    first = FAILED/name
    review = read(first/"review/result.json")
    require(review.get("status") == "PHYSICS_PASSED" and review.get("policy_worker_reaped") is True
        and review.get("policy_worker_return_code") == 0 and review.get("cuda_initialized") is False
        and review.get("render_xml") == identity(runtime["xml"]) and len(review.get("cases",[])) == 1,
        "First H4 physical replay has no clean scene/runtime provenance")
    row = review["cases"][0]
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == FIXED[0])
    selected = next(r["checkpoint"] for r in audit["records"] if r["robot"] == "q1")
    require(review.get("checkpoint") == selected and row.get("clip") == FIXED[0] and row.get("seed") == 0
        and row.get("event") == "nominal" and row.get("variant") == "student" and row.get("horizon_frames") == 4
        and row.get("interface") == "human" and row.get("evaluation_randomization") is False
        and row.get("student_checkpoint") == identity(TRAINED/"selected.pt")
        and row.get("complete") is True and row.get("failure_frame") is None
        and row.get("valid_steps") == row.get("attempted_steps") == row.get("reference_steps") == 1031
        and Path(row["trace"]) == first/"review"/(FIXED[0]+"_s0")/"rollout.npz"
        and all(row.get(k) == item[s]["sha256"] for k,s in
            (("reference_sha256","q1_primebot"),("native_reference_sha256","q1_morph"),("human_sha256","human"))),
        "The reusable first H4 replay has another policy, target or incomplete trajectory")
    import numpy as np
    with np.load(row["trace"],allow_pickle=False) as arrays:
        require(arrays["actual"].shape == arrays["reference"].shape == (1032,79)
            and arrays["canonical_action"].shape == (1031,29)
            and all(np.isfinite(arrays[k]).all() for k in arrays.files),"Invalid reusable H4 arrays")
    timing = first/"stream_timing.jsonl"
    values = [json.loads(line) for line in timing.read_text().splitlines()]
    require(review.get("stream_telemetry") == identity(timing) and len(values) == 1031,"Reusable H4 timing changed")
    timeline = EventTimeline(row["original_paired_reference"]["frames"],EventSpec("nominal",50,10,0))
    for frame,value in enumerate(values):
        validate_timing(value,frame,4,"student",timeline)
    files = sorted(p for p in FAILED.rglob("*") if p.is_file())
    require(files and all(not p.is_symlink() and p.stat().st_size < 8*(1 << 20) for p in files),
        "Unexpected previous H4 evidence file")
    require([identity(p) for p in files] == failure["files"],"The preserved failed H4 root gained or lost evidence")
    protected = auth["protected_inputs"]+failure["files"]+[identity(FAILURE_RECORD)]
    require(all(identity(r["path"]) == r for r in protected),"Preserved failed H4 sources changed")
    return spent,first,protected


class ResumeBudget(CausalReviewBudget):
    def __init__(self,*args,spent):
        super().__init__(*args)
        self.spent = spent

    def check(self):
        require(self.spent+time.monotonic()-self.started < 1800,"Cumulative H4 review1800s bound reached")
        return super().check()


def child_command(args,runtime,owner,case,phase):
    command = original_child(args,runtime,owner,case,phase)
    command[2] = str(Path(__file__).resolve())
    return command


def admission(args):
    config,runtime,plan,frozen,receipt,sources,audit,code = preflight(args)
    require(args.output == OUTPUT and args.training_root == TRAINED and args.review_plan == PLAN
        and identity(PLAN)["sha256"] == args.review_plan_sha256,"Use the fixed H4 continuation paths")
    spent,first,protected = previous_evidence(args,runtime,audit,frozen)
    baselines = matched_baselines(read(args.r1_analysis_root/"summary.json"))
    require(read(FAILED/"baseline_cases.json") == baselines,"Preserved H4 controls differ")
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,args.code_manifest,PLAN)]
    protected = unique(protected+sources+code+controls)
    ledger = checked(read(LEDGER))
    used = charged(dict(ledger,runs=[r for r in ledger["runs"] if r["stage"] == "r1"]))
    require(charged(ledger) == 2670 and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r1"]-used >= 1800-spent and charged(ledger)+1800-spent <= LIMIT,
        "Keep the settled failure charge and original r1/four-hour budget")
    require(not compute_pids(),"External GPU compute is present")
    with own_gpu() as owner:
        require(owner is not None,"H4 continuation lock admission failed")
    return config,runtime,receipt,audit,protected,baselines,spent,first


def execute(args):
    if args.child:
        config,runtime = read(args.config),read(args.runtime_config)
        if args.child == "media":
            runtime_paths(runtime)
            return media(SimpleNamespace(output=folder(args)),config,runtime)
        return replay(args,config,runtime,read(TRAINING/"e1024_train_plan.json"),read(config["frozen_data"]))
    config,runtime,receipt,audit,protected,baselines,spent,first = admission(args)
    if args.admit_only:
        print(json.dumps(dict(status="H4_SPECIALIST_CONTINUATION_ADMITTED",protected_files=len(protected),
            prior_review_seconds=spent,remaining_wall_seconds=1800-spent,reused_physical_cases=1,new_physical_cases=6)))
        return
    with own_gpu() as lock:
        owner = dict(lock,boot=config["expected_boot"],gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        copy_first(first,args.output)
        for name,value in (("checkpoint_audit.json",audit),("baseline_cases.json",baselines),
                ("selected_encoder_audit.json",read(FAILED/"selected_encoder_audit.json"))):
            save(args.output/name,value)
        protected += [identity(args.output/n) for n in ("baseline_cases.json","selected_encoder_audit.json")]
        save(args.output/"authorization.json",dict(stage=STAGE,evaluation_commit=args.evaluation_commit,
            protected_inputs=protected,cases=review_cases(),physical_cases=7,new_physical_cases=6,reused_physical_cases=1,
            reused_baseline_cases=17,maximum_wall_seconds=1800-spent,maximum_cumulative_review_wall_seconds=1800,
            prior_review_wall_seconds=spent,phase_wall_seconds=1800,maximum_output_bytes=3*GIB,
            gpu_lock_owner=owner,selected_update=1500,human_real_future_frames=4,new_optimizer_updates=0,
            original_worker_reused=True,user_accepted_checkpoint=None))
        budget = ResumeBudget(config,receipt,args.output,owner,spent=spent)
        previous,requested,reservation,last_guard = dict(os.environ),False,None,None
        result = dict(stage=STAGE,status="RUNNING",controller=owner,phases=[],cases=[],started_utc=now(),
            reused_baseline_cases=17,reused_physical_cases=1,user_accepted_checkpoint=None,natural_teleoperation_qualified=False)
        def stop(signum,frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            reservation = reserve(LEDGER,run_id=str(args.output),stage="r1",owner=owner,
                gpu=config["expected_gpu"],commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            os.environ.update(environment(runtime))
            for index,case in enumerate(review_cases()):
                args.event_kind,args.clip = case
                for phase in ("replay","media"):
                    if index == 0 and phase == "replay":
                        continue
                    require(all(identity(r["path"]) == r for r in protected),"Preserved H4 evidence changed")
                    name = folder(args).name+"_"+phase
                    result["phase"] = name
                    save(args.output/"result.json",result)
                    budget.phase_started = time.monotonic()
                    last_guard = args.output/(name+".guard.json")
                    guard = guarded_run(child_command(args,runtime,owner,case,phase),ROOT,budget,last_guard,
                        args.output/(name+".log"),check_interval_s=.5,grace_s=5,stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name,guard=identity(last_guard)))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup",{}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0,"H4 continuation phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in protected),"H4 continuation final source gate failed")
            result.update(status="COMPLETED",source_integrity="PASSED",physical_cases=7,new_physical_cases=6,
                total_comparison_cases=24,new_optimizer_updates=0,quality_conclusion="PENDING_SAVED_ANALYSIS",
                cumulative_review_wall_seconds=spent+time.monotonic()-budget.started)
        except BaseException as error:
            result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if reservation:
                cleanup = (read(last_guard).get("cleanup",{}) if last_guard and last_guard.is_file()
                    else dict(worker_started=False,worker_reaped=True))
                if cleanup.get("worker_reaped") is True:
                    result["cost_settlement"] = settle(LEDGER,run_id=str(args.output),owner=owner,cleanup=cleanup)
            result.update(ended_utc=now(),cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json",result)
            os.environ.clear()
            os.environ.update(previous)
            for signum,handler in handlers.items():
                signal.signal(signum,handler)


if __name__ == "__main__":
    execute(parser().parse_args())
