"""Bounded four-stream saved-state profiling, without new physical replays."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import ROOT, identity, now, require, save
from delivery import read
from event_inputs import validate_event_timing
from event_review import command
from gpu_preview_worker import GpuPolicyProcess
from interim_review import TRAINING, fixed_rows
from paused_preview import Budget, checkpoint, preflight
from paused_runtime import assert_owner, own_gpu
from preview_review import environment, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run
from stream_events import EventSpec, EventTimeline

FRAMES = 128
CLIP = "walk_forward_loop_001__A022"
CASES = tuple((event, method, horizon) for event in ("stop", "late_burst")
              for method, horizon in (("masked", 10), ("damped_cv", 4)))
SUMMARY_SHA256 = "74e6399abfe510021bc6b0c2682d960ee4da8af7565ebe0fcf2a31d6a9a68734"


def selected_rows(summary):
    require(summary.get("stage") == "JOINED84_SAVED_EVENT_ANALYSIS"
            and summary.get("status") == "PASSED" and summary.get("joined84_sources_verified") is True
            and summary.get("total_cases") == summary.get("complete_cases") == 84
            and summary.get("original_campaign_status") == "FAILED"
            and summary.get("original_full84_campaign_gate_passed") is False,
            "Require the verified saved84 comparison, retaining its original failure")
    rows = []
    for event, method, horizon in CASES:
        found = [row for row in summary["rows"] if
                 (row["event"], row["method"], row["horizon_frames"], row["clip"])
                 == (event, method, horizon, CLIP)]
        require(len(found) == 1 and found[0].get("complete") is True,
                "A required complete saved stream is missing or duplicated")
        rows.append(found[0])
    return rows


def statistics(values):
    import numpy as np
    require(values and all(math.isfinite(value) and value >= 0 for value in values), "Invalid timing sample")
    return dict(samples=len(values), p50_ms=float(np.percentile(values, 50))/1e6,
        p95_ms=float(np.percentile(values, 95))/1e6, maximum_ms=max(values)/1e6,
        calls_over_20ms=sum(value > 20_000_000 for value in values))


def summarize(frames):
    require(len(frames) == FRAMES and [row["frame"] for row in frames] == list(range(FRAMES)),
            "Require all consecutive saved-state calls")
    result = {}
    for name, subset in (("all", frames), ("without_first_two", frames[2:]),
                         ("capture_event_window", frames[50:60]), ("after_event", frames[60:])):
        keys = frames[0]["profile"]["stage_wall_ns"].keys()
        result[name] = dict(local_pipe_rpc=statistics([r["local_pipe_rpc_ns"] for r in subset]),
            worker_action=statistics([r["profile"]["worker_action_wall_ns"] for r in subset]),
            original_goal_scope=statistics([r["original_goal_scope_ns"] for r in subset]),
            host_stages={key:statistics([r["profile"]["stage_wall_ns"][key] for r in subset]) for key in keys},
            cuda_stream_intervals={key:statistics([r["profile"]["cuda_stream_interval_ms"][key]*1e6
                for r in subset]) for key in frames[0]["profile"]["cuda_stream_interval_ms"]})
    result["slowest_worker_call"] = max(frames, key=lambda row:row["profile"]["worker_action_wall_ns"])
    return result


class ProfileBudget(Budget):
    def check(self):
        require(time.monotonic()-self.started < 1200 and time.monotonic()-self.phase_started < 300,
                "Saved-state profiling whole1200s/phase300s bound reached")
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 64*1024**2,
                "Saved-state profiling output64MiB bound reached")
        return super().check()


def profile(args):
    import numpy as np
    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
    config, runtime = read(args.config), read(args.runtime_config)
    plan = read(TRAINING/"e1024_train_plan.json")
    selected, baseline = checkpoint(args, config)
    parent = read(args.output/"authorization.json")
    row = next(r for r in parent["saved_streams"] if
               (r["event"], r["method"], r["horizon_frames"]) == (args.event_kind, args.tail_method, args.horizon))
    source = next(r for r in parent["saved_trace_inputs"]
                  if r["path"] == str(Path(row["video"]["path"]).with_name("rollout.npz")))
    require(identity(source["path"]) == source, "Saved physical input changed before profiling")
    with np.load(source["path"], allow_pickle=False) as trace:
        actual, expected = trace["actual"][:FRAMES].copy(), trace["canonical_action"][:FRAMES].copy()
        reference_frames = len(trace["reference"])
    require(actual.shape == (FRAMES, 79) and expected.shape == (FRAMES, 29)
            and np.isfinite(actual).all() and np.isfinite(expected).all(), "Invalid saved input prefix")
    frozen = read(config["frozen_data"])
    item = next(r for r in fixed_rows(frozen) if Path(r["q1_primebot"]["path"]).stem == CLIP)
    argv = command(args, config, runtime, plan, selected, baseline)
    argv[2] = str(HERE/"event_profile_worker.py")
    label = f"{args.event_kind}_{args.tail_method}_h{args.horizon}"
    started = time.perf_counter_ns()
    actor = GpuPolicyProcess(argv, args.output/(label+".policy.log"), config["expected_gpu"])
    spawn_ns = time.perf_counter_ns()-started
    frames, maximum = [], 0.
    try:
        require(actor.ready.get("profiling") is True and actor.ready["checkpoint"] == selected,
                "Profiling worker loaded another checkpoint or action path")
        started = time.perf_counter_ns()
        actor.start(item, 0, "human")
        start_ns = time.perf_counter_ns()-started
        timeline = EventTimeline(reference_frames, EventSpec(args.event_kind, 50, 10, 0))
        with (args.output/(label+".frames.jsonl")).open("x") as stream:
            for frame in range(FRAMES):
                before = time.perf_counter_ns()
                reply = actor.action(frame, actual[frame])
                rpc_ns = time.perf_counter_ns()-before
                action = np.asarray(reply["canonical_action"], dtype=np.float32)
                require(action.shape == (29,) and np.isfinite(action).all()
                        and np.allclose(action, expected[frame], atol=2e-5, rtol=2e-5),
                        "Instrumented action differs from the saved same-state CUDA action")
                maximum = max(maximum, float(np.max(np.abs(action-expected[frame]))))
                validate_event_timing(reply["stream"], frame, args.horizon, args.tail_method, timeline)
                record = dict(frame=frame, local_pipe_rpc_ns=rpc_ns, profile=reply["profile"],
                    original_goal_scope_ns=reply["stream"]["gpu_policy_wall_compute_ns"],
                    canonical_action_max_error=float(np.max(np.abs(action-expected[frame]))))
                require(sum(record["profile"]["stage_wall_ns"].values())
                        == record["profile"]["worker_action_wall_ns"], "Host partitions do not sum to the measured worker scope")
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                frames.append(record)
        actor.close()
        require(actor.child.returncode == 0 and identity(source["path"]) == source,
                "Profile worker cleanup failed or its saved source changed")
    finally:
        actor.close(graceful=False)
    result = dict(status="PROFILE_ACTION_PARITY_PASSED", event=args.event_kind, method=args.tail_method,
        horizon_frames=args.horizon, clip=CLIP, calls=FRAMES, maximum_absolute_action_difference=maximum,
        atol=2e-5, rtol=2e-5, source_trace=source, fresh_worker_spawn_to_ready_ns=spawn_ns,
        data_reference_start_ns=start_ns, timing=summarize(frames),
        instrumented_timing=True, physical_transitions=0, optimizer_updates=0,
        measured_end_to_end_latency_ns=None,
        note="Saved-state prefix replay only; timers add instrumentation and do not reproduce the original physical or hardware latency. CUDA stream intervals include host gaps and overlap wall partitions.")
    save(args.output/(label+".summary.json"), result)


def execute(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    summary_path = args.joined_root/"summary.json"
    require(identity(summary_path)["sha256"] == SUMMARY_SHA256
            and read(args.joined_root/"audit.json")["status"] == "JOINED84_ANALYSIS_PASSED",
            "Completed saved84 summary or audit changed")
    summary = read(summary_path)
    streams = selected_rows(summary)
    known = {row["path"]:row for row in summary["inputs"]}
    traces = [known[str(Path(row["video"]["path"]).with_name("rollout.npz"))] for row in streams]
    require(len({r["path"] for r in traces}) == 4 and all(identity(r["path"]) == r for r in traces),
            "Require four original distinct saved traces")
    controls = [identity(p) for p in (args.config,args.runtime_config,args.pause_receipt,args.code_manifest,
                                     summary_path,args.joined_root/"audit.json")]
    require({str(HERE/name) for name in ("event_profile.py","event_profile_worker.py")}
            <= {r["path"] for r in code}, "New profiler is absent from its published manifest")
    require(not any(Path(r["path"]).is_relative_to(args.output) for r in controls+traces+sources+code),
            "Profile output overlaps a protected input")
    with own_gpu() as owner:
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"authorization.json", dict(stage="SAVED32K_EVENT_PROFILE",
            evaluation_commit=args.evaluation_commit, saved_streams=streams, saved_trace_inputs=traces,
            evaluation_sources=code, training_source_inputs=sources, controller_inputs=controls,
            maximum_wall_seconds=1200, phase_wall_seconds=300, maximum_output_bytes=64*1024**2,
            calls_per_stream=FRAMES, maximum_new_policy_calls=4*FRAMES,
            physical_transitions=0, optimizer_updates=0, measured_end_to_end_latency_ns=None,
            user_accepted_checkpoint=None))
        result = dict(stage="SAVED32K_EVENT_PROFILE", status="RUNNING", controller=owner,
            phases=[], started_utc=now(), new_training_updates=0, new_physical_transitions=0)
        budget, requested, previous = ProfileBudget(config, receipt, args.output, owner), False, dict(os.environ)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            base = [runtime["primebot_python"], "-u", str(HERE/"event_profile.py")]
            for key in ("config","runtime_config","output","pause_receipt","audit_root","code_manifest",
                        "joined_root","evaluation_commit","manifest_sha256"):
                base.extend(("--"+key.replace("_","-"),str(getattr(args,key))))
            base.extend(("--child","--gpu-owner-pid",str(owner["pid"]),"--gpu-owner-ticks",owner["start_ticks"]))
            for event, method, horizon in CASES:
                validate_code_manifest(args.code_manifest,args.manifest_sha256,args.evaluation_commit)
                label = f"{event}_{method}_h{horizon}"
                result["phase"] = label
                save(args.output/"result.json",result)
                budget.phase_started = time.monotonic()
                value = guarded_run([*base,"--event-kind",event,"--tail-method",method,"--horizon",str(horizon)],
                    ROOT,budget,args.output/(label+".guard.json"),args.output/(label+".log"),
                    check_interval_s=.5,grace_s=5,stop_requested=lambda:requested)
                result["phases"].append(dict(name=label,guard=identity(args.output/(label+".guard.json"))))
                require(value["status"] == "WORKER_COMPLETED" and value["cleanup"]["worker_reaped"]
                        and value["cleanup"]["worker_return_code"] == 0,"Profile phase failed: "+label)
            values = [read(args.output/f"{e}_{m}_h{h}.summary.json") for e,m,h in CASES]
            require(all(value["status"] == "PROFILE_ACTION_PARITY_PASSED" and value["calls"] == FRAMES
                        for value in values),"Incomplete bounded profile parity")
            require(all(identity(r["path"]) == r for r in controls+traces+sources+code),
                    "Profile sources changed during inference")
            result.update(status="COMPLETED",source_integrity="PASSED",cases=values,
                new_policy_calls=4*FRAMES,user_accepted_checkpoint=None)
        except BaseException as error:
            result.update(status="FAILED",error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["ended_utc"] = now()
            save(args.output/"result.json",result)
            os.environ.clear()
            os.environ.update(previous)
            for signum,handler in handlers.items():signal.signal(signum,handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("config","runtime-config","output","pause-receipt","audit-root","code-manifest","joined-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--evaluation-commit",required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--child",action="store_true")
    parser.add_argument("--gpu-owner-pid",type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.add_argument("--event-kind",choices=("stop","late_burst"))
    parser.add_argument("--tail-method",choices=("masked","damped_cv"))
    parser.add_argument("--horizon",type=int,choices=(4,10))
    args=parser.parse_args()
    profile(args) if args.child else execute(args)
