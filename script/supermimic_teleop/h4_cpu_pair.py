"""Counterbalanced original/kinematic R1 H4 calls on two saved state prefixes."""

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
from event_profile import statistics
from gpu_preview_worker import GpuPolicyProcess
from interim_review import TRAINING, fixed_rows
from native_state_action_math import action_parity
from paused_preview import checkpoint, gpu_command, preflight
from paused_runtime import assert_owner, own_gpu
from preview_review import environment
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline
from student_cost import LIMIT, STAGES, charged, checked, reserve, settle
from student_input_audit import resources
from student_review import validate_timing
from student_same_state import DiagnosticBudget, unique
from student_same_state_math import cleanup_confirmed

DATA = ROOT/"data/supermimic_for_q1"
ANALYSIS = DATA/"Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_REVIEW_ANALYSIS_20261108"
OUTPUT = DATA/"Q1_LOW_LATENCY_CKPT32000_R1_H4_CPU_PAIR01_20261108"
SELECTED = DATA/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R1_20261107/selected.pt"
LEDGER = ROOT/"logs/session_records/q1_low_latency_gpu_cost_20261107.json"
CLIP = "walk_forward_loop_001__A022"
FRAMES = 128
CASES = (("nominal", "original"), ("nominal", "kinematic"),
         ("dropout", "kinematic"), ("dropout", "original"))
FILES = ("h4_cpu_pair.py", "h4_cpu_pair_worker.py", "test_h4_cpu_pair.py",
         "native_state_cpu.py", "native_state_action_math.py")
VISIBILITY = ("used_sequences", "input_ready", "fallback", "max_used_dependency_ns",
              "observed_prefix_frames", "policy_prefix_frames", "requested_preview_ns")


def saved_prefix(actual, actions):
    import numpy as np
    require(actual.ndim == 2 and actual.shape[1] == 79 and actions.shape == (len(actual)-1, 29)
        and len(actions) >= FRAMES and np.isfinite(actual[:FRAMES]).all()
        and np.isfinite(actions[:FRAMES]).all(), "Require an entire finite frame-zero128 prefix")
    return actual[:FRAMES].copy(), actions[:FRAMES].copy()


def compare_pair(original, candidate, present):
    require(len(original) == len(candidate) == FRAMES, "Both original128 and accelerated128 calls are required")
    maximum = 0.
    for frame, (a, b) in enumerate(zip(original, candidate)):
        require(a["frame"] == b["frame"] == frame and all(a["stream"].get(k) == b["stream"].get(k)
            for k in VISIBILITY), "Do not shift frames or change the synthetic receiver visibility")
        maximum = max(maximum, action_parity(b["canonical_action"], a["canonical_action"], present))
    return dict(calls_per_converter=FRAMES, maximum_absolute_action_difference=maximum,
        receiver_visibility_equal=True, action_atol=2e-5, action_rtol=2e-5)


def timing(rows):
    require(len(rows) == FRAMES and [r["frame"] for r in rows] == list(range(FRAMES)), "Missing saved calls")
    fields = ("gpu_policy_wall_compute_ns", "worker_action_wall_ns", "isolated_request_worker_wall_ns", "local_pipe_rpc_ns")
    result = {}
    for name, values in (("all", rows), ("without_first_two", rows[2:]),
            ("synthetic_capture_50_60", [r for r in rows if 50 <= r["frame"]+4 < 60]),
            ("after_synthetic_capture_60", [r for r in rows if r["frame"]+4 >= 60])):
        result[name] = {f:statistics([r["stream"][f] for r in values]) for f in fields}
    result["first_two"] = [{f:r["stream"][f] for f in fields} for r in rows[:2]]
    result["fallback_frames"] = [r["frame"] for r in rows if not r["stream"]["input_ready"]]
    return result


def admission(args):
    config,runtime,plan,frozen,receipt,sources,audit,code = preflight(args)
    sample = resources(True)
    require(args.output == OUTPUT and not OUTPUT.exists(), "Use the independent single-use R1_H4_CPU_PAIR01 root")
    summary, gate = read(ANALYSIS/"summary.json"), read(ANALYSIS/"audit.json")
    require(identity(ANALYSIS/"summary.json")["sha256"] ==
        "f2e1e7e95784f4304740d2c7e15227cfacff7e69a0c0d4063887de97fa1e6eee"
        and summary.get("complete_new_cases") == 7 and summary.get("total_comparison_cases") == 24
        and gate.get("status") == "PASSED" and gate.get("source_inputs_unchanged") is True,
        "Use the actual completed H4 analysis, including all preserved R1 controls")
    protected = summary["inputs"] + [identity(ANALYSIS/n) for n in ("summary.json", "audit.json")]
    known = {r["path"]:r for r in protected}
    saved = []
    for event in ("nominal", "dropout"):
        suffix = f"/{event}_student_h4_{CLIP}/review/result.json"
        paths = [Path(p) for p in known if p.endswith(suffix)
            and "/Q1_LOW_LATENCY_CKPT32000_STUDENT_R1_REVIEW_RESUME2_20261107/" in p]
        require(len(paths) == 1, "Require the exact preserved R1 H4 physical case once")
        review = read(paths[0])
        require(review.get("status") == "PASSED" and len(review.get("cases", [])) == 1
            and review.get("policy_worker_reaped") is True and review.get("policy_worker_return_code") == 0,
            "Original R1 case was not complete and cleanly reaped")
        case = review["cases"][0]
        require(case.get("complete") is True and case.get("event") == event and case.get("clip") == CLIP
            and case.get("horizon_frames") == 4 and case.get("variant") == "student"
            and case.get("student_checkpoint") == identity(SELECTED), "Do not substitute the H4 specialist for R1")
        trace, clock = Path(case["trace"]), paths[0].parents[1]/"stream_timing.jsonl"
        require(all(str(p) in known and identity(p) == known[str(p)] for p in (trace,clock)), "Missing original trace or clock")
        saved.append(dict(event=event, case=case, trace=known[str(trace)], clock=known[str(clock)]))
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,args.code_manifest,SELECTED)]
    protected = unique(protected+sources+code+controls)
    required = {str(HERE/n) for n in FILES}
    require(required <= {r["path"] for r in code} and str(LEDGER) not in {r["path"] for r in protected}
        and all(identity(r["path"]) == r for r in protected), "Published CPU/worker sources or prior identities differ")
    ledger = checked(read(LEDGER))
    used = charged(dict(ledger, runs=[r for r in ledger["runs"] if r["stage"] == "r1"]))
    require(charged(ledger) == 3403 and used == 1460 and not any(r["state"] == "RESERVED" for r in ledger["runs"])
        and STAGES["r1"]-used >= 300 and charged(ledger)+300 <= LIMIT, "Keep the original settled cost and stage budget")
    require(allocated_tree(ROOT)["unique_allocated_bytes"]+64*(1 << 20) < 500_000_000_000, "Private500GB bound")
    with own_gpu():
        pass
    return config,runtime,receipt,audit,protected,saved,sample


def child(args):
    import numpy as np
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    auth = read(args.output/"authorization.json")
    require((args.event_kind,args.cpu_converter) in CASES, "Unexpected comparison condition")
    source = next(r for r in auth["saved_streams"] if r["event"] == args.event_kind)
    require(all(identity(source[k]["path"]) == source[k] for k in ("trace", "clock")), "Saved source changed")
    with np.load(source["trace"]["path"], allow_pickle=False) as values:
        actual, expected = saved_prefix(values["actual"], values["canonical_action"])
        reference_frames = source["case"]["original_paired_reference"]["frames"]
    old_times = [json.loads(s) for s in Path(source["clock"]["path"]).read_text().splitlines()]
    require(len(old_times) == source["case"]["attempted_steps"], "Original clock is truncated")
    config,runtime = read(args.config),read(args.runtime_config)
    selected,baseline = checkpoint(args, config)
    argv = gpu_command(args,config,runtime,read(TRAINING/"e1024_train_plan.json"),selected,baseline,4,"masked")
    argv[2] = str(HERE/"h4_cpu_pair_worker.py")
    argv += ["--student-checkpoint",str(SELECTED),"--variant","student","--event-kind",args.event_kind,
             "--cpu-converter",args.cpu_converter]
    name = args.event_kind+"_"+args.cpu_converter
    started = time.perf_counter_ns()
    actor = GpuPolicyProcess(argv,args.output/(name+".policy.log"),config["expected_gpu"])
    spawn_ns, rows = time.perf_counter_ns()-started, []
    maximum = 0.
    try:
        require(actor.ready.get("original_action_path") is True and actor.ready.get("cpu_converter") == args.cpu_converter
            and actor.ready["student_checkpoint"] == identity(SELECTED) and actor.ready["checkpoint"] == selected
            and actor.ready["metadata"]["iteration"] == 32000, "Wrong frozen R1 or32k policy")
        item = next(r for r in fixed_rows(read(config["frozen_data"])) if Path(r["q1_primebot"]["path"]).stem == CLIP)
        require(actor.start(item,0,"human")["paired_reference"] == source["case"]["original_paired_reference"],
            "Saved reference pairing or reset anchor differs")
        timeline = EventTimeline(reference_frames, EventSpec(args.event_kind,50,10,0))
        present = np.asarray(actor.ready["canonical_present"]).reshape(29)
        with (args.output/(name+".frames.jsonl")).open("x") as stream:
            for frame,state in enumerate(actual):
                before = time.perf_counter_ns()
                reply = actor.action(frame,state)
                rpc_ns = time.perf_counter_ns()-before
                value = validate_timing(reply["stream"],frame,4,"student",timeline)
                validate_timing(old_times[frame],frame,4,"student",timeline)
                require(all(value.get(k) == old_times[frame].get(k) for k in VISIBILITY)
                    and value["cpu_converter"] == args.cpu_converter, "Original saved visibility or CPU condition changed")
                error = action_parity(reply["canonical_action"],expected[frame],present)
                require(len(reply["native_action"]) == 22 and np.isfinite(reply["native_action"]).all(), "Invalid native action")
                maximum = max(maximum,error)
                row = dict(frame=frame,canonical_action=reply["canonical_action"],saved_action_maximum_difference=error,
                    stream=dict(value,local_pipe_rpc_ns=rpc_ns))
                stream.write(json.dumps(row,allow_nan=False)+"\n")
                rows.append(row)
        actor.close()
        require(actor.child.returncode == 0, "R1 H4 comparison policy did not exit cleanly")
    finally:
        actor.close(graceful=False)
    save(args.output/(name+".summary.json"),dict(status="SAVED_ACTION_PARITY_PASSED",event=args.event_kind,
        cpu_converter=args.cpu_converter,calls=len(rows),maximum_absolute_action_difference=maximum,
        ready=actor.ready,fresh_worker_spawn_to_ready_ns=spawn_ns,timing=timing(rows),
        new_physical_steps=0,new_training_updates=0,hardware_end_to_end_latency_ms=None))


def execute(args):
    config,runtime,receipt,audit,protected,saved,sample = admission(args)
    with own_gpu() as owner:
        owner = dict(owner,boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json",audit)
        save(args.output/"authorization.json",dict(stage="R1_H4_CPU_PAIRED_SAVED_ACTIONS",evaluation_commit=args.evaluation_commit,
            protected_inputs=protected,saved_streams=saved,conditions=CASES,frames_per_condition=FRAMES,
            maximum_policy_calls=512,maximum_wall_seconds=300,maximum_output_bytes=64*(1 << 20),
            new_physical_steps=0,new_training_updates=0,user_accepted_checkpoint=None))
        result = dict(stage="R1_H4_CPU_PAIRED_SAVED_ACTIONS",status="RUNNING",controller=owner,started_utc=now(),
            phases=[],admission=sample,new_physical_steps=0,new_training_updates=0,user_accepted_checkpoint=None)
        budget = DiagnosticBudget(config,receipt,args.output,owner)
        budget.maximum_wall = 300
        requested = False

        def stop(signum,frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
        previous,reservation,guards = dict(os.environ),None,[]
        try:
            reservation = reserve(LEDGER,run_id=str(args.output),stage="r1",owner=owner,
                gpu=config["expected_gpu"],commit=args.evaluation_commit)
            result["cost_reservation"] = reservation
            budget.maximum_wall = min(300,reservation["reserved_seconds"])
            os.environ.update(environment(runtime))
            for event,converter in CASES:
                require(all(identity(r["path"]) == r for r in protected), "Protected input changed before a phase")
                name = event+"_"+converter
                result["phase"] = name
                save(args.output/"result.json",result)
                budget.phase_started = time.monotonic()
                cmd = [runtime["primebot_python"],"-u",str(HERE/"h4_cpu_pair.py"),"--child",
                    "--output",str(args.output),"--config",str(args.config),"--runtime-config",str(args.runtime_config),
                    "--event-kind",event,"--cpu-converter",converter,"--gpu-owner-pid",str(owner["pid"]),
                    "--gpu-owner-ticks",owner["start_ticks"]]
                path = args.output/(name+".guard.json")
                guards.append(path)
                guard = guarded_run(cmd,ROOT,budget,path,args.output/(name+".log"),check_interval_s=.5,
                    grace_s=5,stop_requested=lambda:requested)
                result["phases"].append(dict(name=name,guard=identity(path)))
                require(guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"] is True
                    and guard["cleanup"]["worker_return_code"] == 0, "H4 comparison phase failed: "+name)
            comparisons = []
            for event in ("nominal", "dropout"):
                paths = [args.output/(event+"_"+c+".frames.jsonl") for c in ("original", "kinematic")]
                rows = [[json.loads(s) for s in p.read_text().splitlines()] for p in paths]
                a,b = [read(args.output/(event+"_"+c+".summary.json")) for c in ("original", "kinematic")]
                require(a["ready"]["canonical_present"] == b["ready"]["canonical_present"], "Canonical mask changed")
                comparisons.append(dict(event=event,**compare_pair(*rows,a["ready"]["canonical_present"]),
                    original_timing=a["timing"],kinematic_timing=b["timing"]))
            require(all(identity(r["path"]) == r for r in protected), "Final H4 paired source integrity failed")
            result.update(status="CPU_H4_PAIRED_ACTIONS_PASSED",source_integrity="PASSED",new_policy_calls=512,
                comparisons=comparisons,natural_teleoperation_qualified=False,hardware_end_to_end_latency_ms=None,
                timing_note="ABBA serial workers; descriptive same-state software comparison, no concurrent baseline or hardware claim")
        except BaseException as error:
            result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            values = [read(p) for p in guards if p.is_file()]
            if reservation and cleanup_confirmed(values,len(guards)):
                result["cost_settlement"] = settle(LEDGER,run_id=str(args.output),owner=owner,
                    cleanup=dict(worker_reaped=True,worker_return_code=0 if result["status"] == "CPU_H4_PAIRED_ACTIONS_PASSED" else -1,
                        worker_started=bool(values),owned_phase_workers_reaped=len(values)))
            result.update(ended_utc=now(),cumulative_gpu_charged_seconds=charged(read(LEDGER)))
            save(args.output/"result.json",result)
            os.environ.clear()
            os.environ.update(previous)
            for signum,handler in handlers.items():
                signal.signal(signum,handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name,value in dict(config=ROOT/"script/q1_five_stage/conditional_shared_v04_20261104.json",
            runtime_config=ROOT/"data/manifests/primebot_q1_monitored_config_20261103.json",
            pause_receipt=ROOT/"logs/session_records/q1_shared_user_pause_20261106.json",
            audit_root=DATA/"Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106").items():
        p.add_argument("--"+name.replace("_","-"),type=Path,default=value)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--code-manifest",type=Path)
    p.add_argument("--manifest-sha256")
    p.add_argument("--evaluation-commit")
    p.add_argument("--admit-only",action="store_true")
    p.add_argument("--child",action="store_true")
    p.add_argument("--event-kind",choices=("nominal","dropout"))
    p.add_argument("--cpu-converter",choices=("original","kinematic"))
    p.add_argument("--gpu-owner-pid",type=int)
    p.add_argument("--gpu-owner-ticks")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.child:
        child(args)
    elif args.admit_only:
        c,r,p,a,s,t,sample = admission(args)
        print(json.dumps(dict(status="R1_H4_CPU_PAIR_ADMITTED",protected_files=len(s),resources=sample,new_policy_calls=0)))
    else:
        execute(args)
