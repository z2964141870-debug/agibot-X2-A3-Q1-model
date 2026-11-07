"""Seven matched replays of the validation-selected H4 specialist only."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from interim_review import GIB, TRAINING, media
from paused_preview import preflight
from paused_runtime import own_gpu
from preview_review import environment, validate_code_manifest
from runtime_guard import guarded_run
from student_causal_review import CausalReviewBudget, baseline_cases, review_cases
from student_causal_training import saved_analysis_gate
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_h4_training import LEDGER, OUTPUT as TRAINED, PARENT
from student_input_audit import resources
from student_review import folder, replay, training_evidence
from student_same_state import unique

OUTPUT = ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_REVIEW_20261108"
PLAN = HERE/"student_h4_review_plan_20261108.json"
STAGE = "H4_SPECIALIST01_MATCHED_CLOSED_LOOP"


def selection_gate(fit, trace):
    require(fit.get("method") == "h4_only_continuation" and fit.get("actual_updates") == 3000
        and fit.get("actual_adam_steps") == [3000]*10
        and fit.get("interface") == "human_prefix_adapter"
        and fit.get("allowed_request_horizons") == [4] and fit.get("human_real_future_frames") == 4
        and fit.get("maximum_preview_ns") == 80_000_000
        and fit.get("selection_split") == "validation" and fit.get("test_used_for_selection") is False
        and fit.get("test_sources_read") is False and fit.get("normalization_refitted") is False,
        "Require the actual H4-only fit with unchanged validation selection")
    require(trace and trace[0]["update"] == 0 and trace[-1]["update"] == 3000
        and all(a["update"] < b["update"] for a,b in zip(trace,trace[1:])), "Incomplete actual H4 trace")
    selected = min(trace, key=lambda r:r["validation_risk"])
    require(selected["update"] == fit.get("selected_update") == 1500
        and selected["validation_risk"] == fit.get("selected_validation_risk") < trace[0]["validation_risk"],
        "Only validation-selected1500 may enter the new H4 review")
    return selected


def matched_baselines(summary):
    rows = baseline_cases(summary)
    context = [r for r in summary["rows"] if r.get("event") == "nominal"
        and r.get("variant") == "teacher" and r.get("horizon_frames") == 10]
    require(len(context) == 3 and {r["clip"] for r in context} == set(FIXED),
            "Preserve all three support-aware H10 nominal context cases")
    for row in context:
        matched = next(r for r in rows if (r["event"],r["clip"],r["variant"]) ==
                       ("nominal",row["clip"],"teacher"))
        require(row["derived_reference_sha256"] == matched["derived_reference_sha256"],
                "H10 and H4 saved targets differ")
    return rows+context


def trained_evidence(root, settings):
    result, fit, guard, auth, trace = (read(root/n) for n in
        ("result.json","fit_result.json","fit.guard.json","authorization.json","training_trace.json"))
    selection_gate(fit, trace)
    require(result.get("stage") == auth.get("stage") == "H4_SPECIALIST_CONTINUATION"
        and result.get("status") == fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
        and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement",{}).get("state") == "SETTLED"
        and result.get("fit_result") == identity(root/"fit_result.json")
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup",{}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc",str(result["controller"]["pid"])).exists(),
        "Wait for real H4 completion, settled cost, source gate and owned cleanup")
    require(settings.get("schema") == "q1_h4_specialist_review_v1"
        and settings.get("training_root") == str(root)
        and settings.get("fit_result") == identity(root/"fit_result.json")
        and settings.get("selected_checkpoint") == identity(root/"selected.pt")
        and settings.get("selected_update") == 1500
        and settings.get("maximum_new_cases") == 7 and settings.get("maximum_wall_seconds") == 1800
        and settings.get("new_optimizer_updates") == 0, "H4 selected checkpoint or bounded review plan changed")
    records = [identity(root/n) for n in ("result.json","fit_result.json","fit.guard.json",
        "authorization.json","training_trace.json","selected.pt","checkpoint_audit.json","reused_dataset.json")]
    records += auth["protected_inputs"]
    require(not any(r["path"] == str(LEDGER) for r in records), "A changing ledger entered immutable sources")
    require(all(identity(r["path"]) == r for r in records), "Completed H4 artifacts or protected sources changed")
    return records


def changed_encoder(root, original):
    import torch
    from upper_mps_forecast import audit_cpu_tensors
    current, previous = [torch.load(p/"selected.pt",map_location="cpu",weights_only=True) for p in (root,original)]
    for value in (current,previous):
        audit_cpu_tensors(torch,value)
    keys = ("schema","interface","decoder_sha256","frozen_teacher_policy_sha256")
    require(all(current[k] == previous[k] for k in keys)
        and current["provenance"]["teacher"] == previous["provenance"]["teacher"]
        and current.get("method") == "h4_only_continuation" and current.get("selected_update") == 1500
        and current.get("allowed_request_horizons") == [4]
        and current["provenance"]["initial_student"] == identity(original/"selected.pt")
        and current["encoder"].keys() == previous["encoder"].keys(), "Selected H4 metadata or decoder differs")
    changes = {key:float((value-previous["encoder"][key]).abs().max()) for key,value in current["encoder"].items()}
    require(any(changes.values()) and not torch.cuda.is_initialized(), "Expected changed finite H4 encoder on CPU")
    return dict(status="CHANGED_SELECTED_H4_ENCODER",selected_update=1500,
        tensor_maximum_abs_difference=changes,frozen_policy_metadata_equal=True,cuda_initialized=False,
        files=[identity(p/"selected.pt") for p in (root,original)])


def admission(args):
    sample = resources(True)
    config,runtime,plan,frozen,receipt,sources,audit,code = preflight(args)
    require(args.output == OUTPUT and args.training_root == TRAINED and args.review_plan == PLAN
        and identity(PLAN)["sha256"] == args.review_plan_sha256, "Use the unique published H4 review paths")
    settings = read(PLAN)
    parent = training_evidence(PARENT)
    protected = parent+saved_analysis_gate(args.r1_analysis_root,parent)+trained_evidence(TRAINED,settings)
    changed = changed_encoder(TRAINED,PARENT)
    baselines = matched_baselines(read(args.r1_analysis_root/"summary.json"))
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,
        args.code_manifest,PLAN,TRAINING/"e1024_train_plan.json")]
    required = {str(HERE/n) for n in ("student_h4_review.py","student_review.py","student_review_worker.py")}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in sources},
            "Missing independent H4 review code or frozen input overlap")
    protected = unique(protected+changed["files"]+controls+sources+code)
    ledger = checked(read(LEDGER))
    used = charged(dict(ledger,runs=[r for r in ledger["runs"] if r["stage"] == "r1"]))
    require(charged(ledger) == settings["expected_cumulative_seconds"] == 2569
        and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r1"]-used >= 1800 and charged(ledger)+1800 <= LIMIT,
        "Keep the existing settled ledger and original r1/four-hour budget")
    with own_gpu() as owner:
        require(owner is not None, "H4 review idle-lock admission failed")
    return config,runtime,receipt,audit,protected,baselines,changed,sample


def child_command(args,runtime,owner,case,phase):
    event,clip = case
    require(phase in ("replay","media") and (event,clip) in review_cases(), "Unexpected H4 case or phase")
    return [runtime["primebot_python"],"-u",str(Path(__file__).resolve()),"--child",phase,
        "--config",str(args.config),"--runtime-config",str(args.runtime_config),"--output",str(args.output),
        "--training-root",str(args.training_root),"--event-kind",event,"--clip",clip,
        "--gpu-owner-pid",str(owner["pid"]),"--gpu-owner-ticks",owner["start_ticks"]]


def execute(args):
    if args.child:
        config,runtime = read(args.config),read(args.runtime_config)
        if args.child == "media":
            media(SimpleNamespace(output=folder(args)),config,runtime)
        else:
            replay(args,config,runtime,read(TRAINING/"e1024_train_plan.json"),read(config["frozen_data"]))
        return
    config,runtime,receipt,audit,protected,baselines,changed,sample = admission(args)
    if args.admit_only:
        print(json.dumps(dict(status="H4_SPECIALIST_REVIEW_ADMITTED",resources=sample,
            protected_files=len(protected),reused_cases=len(baselines),selected_encoder_audit=changed)))
        return
    with own_gpu() as lock:
        owner = dict(lock,boot=config["expected_boot"],gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        for name,value in (("checkpoint_audit.json",audit),("baseline_cases.json",baselines),
                           ("selected_encoder_audit.json",changed)):
            save(args.output/name,value)
        protected += [identity(args.output/n) for n in ("baseline_cases.json","selected_encoder_audit.json")]
        save(args.output/"authorization.json",dict(stage=STAGE,evaluation_commit=args.evaluation_commit,
            protected_inputs=protected,cases=review_cases(),new_physical_cases=7,reused_baseline_cases=17,
            maximum_wall_seconds=1800,phase_wall_seconds=1800,maximum_output_bytes=3*GIB,
            gpu_lock_owner=owner,selected_update=1500,human_real_future_frames=4,new_optimizer_updates=0,
            original_worker_reused=True,user_accepted_checkpoint=None))
        budget = CausalReviewBudget(config,receipt,args.output,owner)
        previous,requested,reservation,last_guard = dict(os.environ),False,None,None
        result = dict(stage=STAGE,status="RUNNING",controller=owner,phases=[],cases=[],started_utc=now(),
            reused_baseline_cases=17,user_accepted_checkpoint=None,natural_teleoperation_qualified=False)
        def stop(signum,frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            reservation = reserve(LEDGER,run_id=str(args.output),stage="r1",owner=owner,
                                  gpu=config["expected_gpu"],commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            os.environ.update(environment(runtime))
            for case in review_cases():
                args.event_kind,args.clip = case
                for phase in ("replay","media"):
                    require(all(identity(r["path"]) == r for r in protected), "H4 review source changed")
                    name = folder(args).name+"_"+phase
                    result["phase"] = name
                    save(args.output/"result.json",result)
                    budget.phase_started = time.monotonic()
                    last_guard = args.output/(name+".guard.json")
                    guard = guarded_run(child_command(args,runtime,owner,case,phase),ROOT,budget,last_guard,
                        args.output/(name+".log"),check_interval_s=.5,grace_s=5,stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name,guard=identity(last_guard)))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup",{}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0, "H4 review phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in protected), "H4 review final source gate failed")
            result.update(status="COMPLETED",source_integrity="PASSED",new_physical_cases=7,
                total_comparison_cases=24,new_optimizer_updates=0,quality_conclusion="PENDING_SAVED_ANALYSIS")
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


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("config","runtime-config","output","training-root","r1-analysis-root", "pause-receipt",
                 "audit-root","code-manifest","review-plan"):
        p.add_argument("--"+name,type=Path)
    for name in ("evaluation-commit","manifest-sha256","review-plan-sha256","event-kind","clip","gpu-owner-ticks"):
        p.add_argument("--"+name)
    p.add_argument("--gpu-owner-pid",type=int)
    p.add_argument("--child",choices=("replay","media"))
    p.add_argument("--admit-only",action="store_true")
    p.set_defaults(horizon=4,tail_method="masked",variant="student")
    return p


if __name__ == "__main__":
    execute(parser().parse_args())
