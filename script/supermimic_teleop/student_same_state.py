"""Bounded four-policy diagnostic on two already saved failed f0 prefixes."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1"),
                str(HERE.parent/"q1_checkpoint_review")]
from common import ROOT, identity, now, require, save
from delivery import read
from event_inputs import validate_event_timing
from event_profile import statistics
from gpu_preview_worker import GpuPolicyProcess
from interim_review import TRAINING, fixed_rows
from paused_preview import checkpoint, gpu_command, preflight
from paused_runtime import assert_owner, own_gpu
from preview_review import environment, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline
from student_causal_review import validate_causal_timing
from student_cost import LIMIT, charged, checked, reserve, settle
from student_coverage_summary import completed_gate
from student_same_state_math import CLIPS, MAX_FRAMES, SELECTED_SHA, VARIANTS, cleanup_confirmed, compare, parity, prefix
from student_training import TrainingBudget

STAGE = "COVERAGE01_SAME_STATE_DIAGNOSTIC"
DATA = ROOT/"data/supermimic_for_q1"
REVIEW = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_REVIEW_20261107"
ANALYSIS = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_REVIEW_ANALYSIS_20261107"
SELECTED = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_20261107/selected.pt"
ORIGINAL = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107/selected.pt"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
NEW_FILES = ("student_same_state.py", "student_same_state_worker.py", "student_same_state_math.py")


def unique(records):
    values = {}
    for r in records:
        require(r["path"] not in values or values[r["path"]] == r, "Conflicting source identities")
        values[r["path"]] = r
    return list(values.values())


def admission(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    result, auth = read(REVIEW/"result.json"), read(REVIEW/"authorization.json")
    completed_gate(result, auth)
    summary, gate = read(ANALYSIS/"summary.json"), read(ANALYSIS/"audit.json")
    require(summary.get("stage") == "COVERAGE01_MATCHED_SAVED_ANALYSIS"
        and summary.get("complete_new_cases") == 0 and summary.get("total_comparison_cases") == 28
        and gate.get("status") == "PASSED" and gate.get("source_inputs_unchanged") is True
        and summary.get("new_inference_calls") == summary.get("new_physical_transitions")
        == summary.get("new_training_updates") == 0, "Require original failed seven-case saved analysis")
    protected = summary["inputs"]+[identity(ANALYSIS/n) for n in ("summary.json", "audit.json")]
    protected += [identity(SELECTED), identity(ORIGINAL)]
    require(identity(SELECTED)["sha256"] == SELECTED_SHA, "Do not replace validation-selected update500")
    traces = []
    known = {r["path"]:r for r in protected}
    for clip in CLIPS:
        directory = REVIEW/f"nominal_f0_student_h0_{clip}"
        p = directory/"review/result.json"
        require(identity(p) in result["cases"] and read(p).get("status") == "PASSED", "Changed saved case")
        case = read(p)["cases"][0]
        trace, timing = directory/"review"/(clip+"_s0")/"rollout.npz", directory/"stream_timing.jsonl"
        require(case["trace"] == str(trace) and case["student_checkpoint"] == identity(SELECTED)
            and all(str(f) in known and identity(f) == known[str(f)] for f in (trace,timing,p)),
            "Require actual saved source identities, clocks and selected encoder")
        traces.append(dict(case=case, trace=known[str(trace)], timing=known[str(timing)]))
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,args.code_manifest)]
    protected = unique(protected+controls+sources+code)
    required = {str(HERE/n) for n in NEW_FILES}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in summary["inputs"]},
            "Diagnostic code must be independent of all frozen and prior execution sources")
    require(all(identity(r["path"]) == r for r in protected), "Saved source or executable changed")
    ledger = checked(read(LEDGER))
    require(not any(r["state"] == "RESERVED" for r in ledger["runs"]) and charged(ledger)+900 <= LIMIT,
            "Unresolved reservation or insufficient original cumulative budget")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+64*1024**2 < 500_000_000_000,
            "Project500GB bound reached")
    return config,runtime,receipt,audit,protected,traces


class DiagnosticBudget(TrainingBudget):
    def check(self):
        require(time.monotonic()-self.phase_started < 300, "Diagnostic phase300s bound reached")
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 64*1024**2, "Output64MiB bound reached")
        return super().check()


def profile_summary(rows):
    result = {}
    for name, subset in (("all",rows), ("without_first_two",rows[2:])):
        p = [r["profile"] for r in subset if r.get("profile") is not None]
        value = dict(local_pipe_rpc=statistics([r["local_pipe_rpc_ns"] for r in subset]),
            original_target_scope=statistics([r["stream"]["gpu_policy_wall_compute_ns"] for r in subset]))
        if p:
            value["worker_action"] = statistics([r["worker_action_wall_ns"] for r in p])
            value["host_stages"] = {k:statistics([r["stage_wall_ns"][k] for r in p]) for k in p[0]["stage_wall_ns"]}
            cuda = [r["cuda_stream_interval_ms"] for r in p if r["cuda_stream_interval_ms"] is not None]
            value["cuda_stream_intervals"] = {k:statistics([r[k]*1e6 for r in cuda]) for k in cuda[0]} if cuda else None
        result[name] = value
    return result


def diagnose(args):
    import numpy as np
    assert_owner(args.gpu_owner_pid,args.gpu_owner_ticks)
    auth = read(args.output/"authorization.json")
    source = next(r for r in auth["saved_prefixes"] if r["case"]["clip"] == args.clip)
    case = source["case"]
    require(all(identity(source[k]["path"]) == source[k] for k in ("trace","timing")), "Saved prefix changed")
    with np.load(source["trace"]["path"], allow_pickle=False) as values:
        actual, expected = prefix(values["actual"],values["canonical_action"],case)
        reference_frames = len(values["reference"])
    saved_times = [json.loads(s) for s in Path(source["timing"]["path"]).read_text().splitlines()]
    require(len(saved_times) == case["attempted_steps"], "Saved calls or attempted failure missing")
    config, runtime = read(args.config),read(args.runtime_config)
    selected, baseline = checkpoint(args,config)
    plan = read(TRAINING/"e1024_train_plan.json")
    item = next(r for r in fixed_rows(read(config["frozen_data"])) if Path(r["q1_primebot"]["path"]).stem == args.clip)
    f0 = args.variant in VARIANTS[:2]
    horizon,method = (0,"masked") if f0 else ((4,"damped_cv") if args.variant == "teacher_h4" else (10,"masked"))
    argv = gpu_command(args,config,runtime,plan,selected,baseline,horizon,method)
    argv[2] = str(HERE/("student_same_state_worker.py" if f0 else "event_gpu_worker.py"))
    argv += ["--event-kind","nominal"]
    chosen = SELECTED if args.variant == "coverage_f0" else ORIGINAL
    if f0:argv += ["--student-checkpoint",str(chosen)]
    name = f"{args.clip}_{args.variant}"
    start = time.perf_counter_ns()
    actor = GpuPolicyProcess(argv,args.output/(name+".policy.log"),config["expected_gpu"])
    spawn_ns = time.perf_counter_ns()-start
    actions,rows,maximum = [],[],0.
    try:
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000,
                "Wrong frozen teacher checkpoint")
        if f0:
            require(actor.ready.get("same_state_profile") is True and actor.ready["student_checkpoint"] == identity(chosen),
                    "Wrong instrumented student checkpoint")
        started = actor.start(item,0,"human")
        require(started["paired_reference"] == case["original_paired_reference"], "Reference anchor or pairing changed")
        timeline = EventTimeline(reference_frames,EventSpec("nominal",50,10,0))
        with (args.output/(name+".frames.jsonl")).open("x") as stream:
            for frame,state in enumerate(actual):
                before = time.perf_counter_ns()
                reply = actor.action(frame,state)
                rpc_ns = time.perf_counter_ns()-before
                action = np.asarray(reply["canonical_action"],dtype=np.float32)
                require(action.shape == (29,) and np.isfinite(action).all(), "Invalid same-state action")
                error = None
                if f0:
                    validate_causal_timing(reply["stream"],frame,timeline)
                    validate_causal_timing(saved_times[frame],frame,timeline)
                    prof = reply["profile"]
                    require(sum(prof["stage_wall_ns"].values()) == prof["worker_action_wall_ns"], "Host timing partition mismatch")
                    if args.variant == "coverage_f0":
                        error = parity(action,expected[frame])
                        maximum = max(maximum,error)
                else:
                    validate_event_timing(reply["stream"],frame,horizon,method,timeline)
                record = dict(frame=frame,canonical_action=action.tolist(),local_pipe_rpc_ns=rpc_ns,
                    saved_action_maximum_difference=error,stream=reply["stream"],profile=reply.get("profile"))
                stream.write(json.dumps(record,allow_nan=False)+"\n")
                rows.append(record)
                actions.append(action)
        actor.close()
        require(actor.child.returncode == 0,"Same-state policy did not cleanly exit")
    finally:
        actor.close(graceful=False)
    with (args.output/(name+".actions.npz")).open("xb") as stream:
        np.savez(stream,canonical_action=np.asarray(actions,dtype=np.float32))
    save(args.output/(name+".summary.json"),dict(status="SAME_STATE_CALLS_PASSED",clip=args.clip,
        variant=args.variant,calls=len(actions),ready=actor.ready,source_trace=source["trace"],
        maximum_saved_action_difference=maximum if args.variant == "coverage_f0" else None,
        new_checkpoint_parity_checked=args.variant == "coverage_f0",timing=profile_summary(rows),
        fresh_worker_spawn_to_ready_ns=spawn_ns,physical_steps=0,optimizer_updates=0))


def execute(args):
    config,runtime,receipt,audit,protected,traces = admission(args)
    with own_gpu() as owner:
        owner = dict(owner,boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json",audit)
        save(args.output/"authorization.json",dict(stage=STAGE,evaluation_commit=args.evaluation_commit,
            protected_inputs=protected,saved_prefixes=traces,maximum_policy_calls=1024,
            maximum_frames_per_clip=MAX_FRAMES,maximum_wall_seconds=900,phase_wall_seconds=300,
            maximum_output_bytes=64*1024**2,privileged_references=["teacher_h4","teacher_h10"],
            new_physical_steps=0,new_optimizer_updates=0,user_accepted_checkpoint=None))
        result = dict(stage=STAGE,status="RUNNING",controller=owner,phases=[],started_utc=now(),
            new_physical_steps=0,new_optimizer_updates=0,user_accepted_checkpoint=None)
        budget,previous,requested = DiagnosticBudget(config,receipt,args.output,owner),dict(os.environ),False
        budget.maximum_wall=900
        reservation = None
        guard_paths = []
        def stop(signum,frame):
            nonlocal requested
            requested=True
        handlers={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            reservation=reserve(LEDGER,run_id=str(args.output),stage="r3",owner=owner,
                gpu=config["expected_gpu"],commit=args.evaluation_commit)
            result["cost_reservation"]=reservation
            budget.maximum_wall=min(900,reservation["reserved_seconds"])
            os.environ.update(environment(runtime))
            for clip in CLIPS:
                for variant in VARIANTS:
                    validate_code_manifest(args.code_manifest,args.manifest_sha256,args.evaluation_commit)
                    name=f"{clip}_{variant}"
                    result["phase"]=name
                    save(args.output/"result.json",result)
                    budget.phase_started=time.monotonic()
                    cmd=[runtime["primebot_python"],"-u",str(HERE/"student_same_state.py"),"--child",
                        "--output",str(args.output),"--config",str(args.config),"--runtime-config",str(args.runtime_config),
                        "--variant",variant,"--clip",clip,"--gpu-owner-pid",str(owner["pid"]),
                        "--gpu-owner-ticks",owner["start_ticks"]]
                    guard_path=args.output/(name+".guard.json")
                    guard_paths.append(guard_path)
                    guard=guarded_run(cmd,ROOT,budget,guard_path,args.output/(name+".log"),
                        check_interval_s=.5,grace_s=5,stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name,guard=identity(args.output/(name+".guard.json"))))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup",{}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0,"Same-state diagnostic phase failed: "+name)
            import numpy as np
            comparisons=[]
            total=0
            for source in traces:
                clip=source["case"]["clip"]
                values={}
                summaries={v:read(args.output/f"{clip}_{v}.summary.json") for v in VARIANTS}
                for variant in VARIANTS:
                    with np.load(args.output/f"{clip}_{variant}.actions.npz",allow_pickle=False) as arrays:
                        values[variant]=arrays["canonical_action"].copy()
                present=summaries["coverage_f0"]["ready"]["canonical_present"]
                require(present == summaries["original_f0"]["ready"]["canonical_present"],"Canonical presence differs")
                comparisons.append(dict(clip=clip,**compare(values,present,source["case"]["upper_joint_indices"],source["case"])))
                total+=sum(r["calls"] for r in summaries.values())
            require(total<=1024 and all(identity(r["path"]) == r for r in protected),"Call limit or final source integrity failed")
            save(args.output/"comparison.json",dict(stage=STAGE,comparisons=comparisons,
                timing_scope="Instrumented software only; CUDA/host intervals overlap; no hardware end-to-end claim",
                privileged_future_for_offline_reference_only=True,natural_teleoperation_qualified=False))
            result.update(status="COMPLETED",source_integrity="PASSED",new_policy_calls=total,
                saved_action_parity="PASSED",same_state_comparisons=2)
        except BaseException as error:
            result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            guards=[read(p) for p in guard_paths if p.is_file()]
            if reservation and cleanup_confirmed(guards,len(guard_paths)):
                result["cost_settlement"]=settle(LEDGER,run_id=str(args.output),owner=owner,
                    cleanup=dict(worker_reaped=True,worker_return_code=0 if result["status"] == "COMPLETED" else -1,
                        worker_started=bool(guards),owned_phase_workers_reaped=len(guards)))
            result["ended_utc"]=now()
            save(args.output/"result.json",result)
            os.environ.clear()
            os.environ.update(previous)
            for signum,handler in handlers.items():signal.signal(signum,handler)


def parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    defaults=dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
        runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
        pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json",
        audit_root=DATA/"Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106")
    for name,value in defaults.items():p.add_argument("--"+name.replace("_","-"),type=Path,default=value)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--code-manifest",type=Path)
    p.add_argument("--manifest-sha256")
    p.add_argument("--evaluation-commit")
    p.add_argument("--admit-only",action="store_true")
    p.add_argument("--child",action="store_true")
    p.add_argument("--gpu-owner-pid",type=int)
    p.add_argument("--gpu-owner-ticks")
    p.add_argument("--clip",choices=CLIPS)
    p.add_argument("--variant",choices=VARIANTS)
    return p


if __name__ == "__main__":
    args=parser().parse_args()
    if args.child:diagnose(args)
    elif args.admit_only:
        c,r,p,a,s,t=admission(args)
        with own_gpu() as owner:
            print(json.dumps(dict(status="SAME_STATE_ADMITTED",sources=len(s),clips=len(t),
                cumulative_charged_seconds=charged(read(LEDGER)),gpu_owner=owner,new_policy_calls=0)))
    else:execute(args)
