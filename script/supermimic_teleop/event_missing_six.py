"""Run only six absent late-burst cases, preserving the failed original84 run."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from event_review import cases
from paused_preview import Budget, ITERATION, preflight
from paused_runtime import own_gpu
from preview_review import case_folder, environment, validate_code_manifest
from runtime_guard import allocated_tree, guarded_run

MISSING = tuple(("late_burst", "damped_cv", horizon, clip) for horizon in (5, 4) for clip in FIXED)
ORIGINAL_RESULT_SHA256 = "81f7f90c39b68a25fe5c15760e75902e6a9839f9a5ef3a44bd7a7ac3e38a7191"
PARTIAL_SUMMARY_SHA256 = "78985d701c1a00b0e6ee631302716e23b091e65c50a3bbd177496ee0dbc6bfde"


def partition(summary):
    rows = summary.get("rows", [])
    keys = [(r["event"], r["method"], r["horizon_frames"], r["clip"]) for r in rows]
    absent = [(r["event"], r["method"], r["horizon_frames"], r["clip"]) for r in summary.get("missing", [])]
    require(summary.get("stage") == "PARTIAL_SAVED_EVENT_ANALYSIS" and summary.get("source_campaign_status") == "FAILED"
            and summary.get("source_final_gate_passed") is False and summary.get("partial_source_reaudit") == "PASSED"
            and len(keys) == len(set(keys)) == 78 and len(absent) == len(set(absent)) == 6
            and set(absent) == set(MISSING) and not set(keys)&set(absent)
            and set(keys)|set(absent) == set(cases()), "Require the exact preserved78 and only the missing six")
    return tuple(keys), MISSING


class MissingBudget(Budget):
    def check(self):
        require(time.monotonic()-self.started < 1800, "Missing-six total1800s bound reached")
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 1024**3,
                "Missing-six output1GiB bound reached")
        record = super().check()
        record.update(maximum_wall_seconds=1800, phase_wall_seconds=1800, maximum_output_bytes=1024**3)
        save(self.output/"resources.json", record)
        return record


def parent_evidence(args, audit):
    original, partial = args.original_root, args.partial_root
    require(original.is_absolute() and original.resolve() == original and partial.is_absolute()
            and partial.resolve() == partial and original != args.output
            and not args.output.is_relative_to(original) and not args.output.is_relative_to(partial),
            "Preserve both original and partial-analysis roots")
    result_path, summary_path = original/"result.json", partial/"summary.json"
    require(identity(result_path)["sha256"] == ORIGINAL_RESULT_SHA256
            and identity(summary_path)["sha256"] == PARTIAL_SUMMARY_SHA256, "Preserved parent artifacts changed")
    result, summary, analysis = read(result_path), read(summary_path), read(partial/"audit.json")
    guard = read(original/(result["phase"]+".guard.json"))
    require(result.get("status") == guard.get("status") == "FAILED"
            and result.get("error") == guard.get("error")
                == "RuntimeError: Paused campaign whole/phase wall-time bound reached"
            and guard["cleanup"].get("worker_reaped") is True and guard["cleanup"].get("worker_return_code") == -15
            and all(not Path("/proc", str(pid)).exists() for pid in
                    (result["controller"]["pid"], guard["worker_pid"]))
            and analysis.get("status") == "PARTIAL_ANALYSIS_PASSED"
            and analysis.get("source_inputs_unchanged") is True, "Parent stop or partial analysis is not clean")
    reused, missing = partition(summary)
    auth, equality = read(original/"authorization.json"), read(original/"equivalence.json")
    require(read(original/"checkpoint_audit.json") == audit and equality.get("status") == "PASSED"
            and len(equality.get("cases", [])) == 21 and equality.get("atol") == equality.get("rtol") == 2e-5,
            "Checkpoint or existing action gate differs")
    records = [identity(p) for p in (result_path, original/(result["phase"]+".guard.json"),
        summary_path, partial/"audit.json", original/"authorization.json", original/"equivalence.json",
        original/"checkpoint_audit.json")]
    records += summary["inputs"]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]+auth["reused_nominal_evidence"]
    unique = {}
    for record in records:
        require(record["path"] not in unique or unique[record["path"]] == record,
                "Conflicting reused source identities")
        unique[record["path"]] = record
    require(all(identity(r["path"]) == r for r in unique.values()), "Saved78 or unchanged parent sources changed")
    return reused, missing, list(unique.values())


def execute(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require(str(Path(__file__).resolve()) in {r["path"] for r in code}, "Missing published follow-up source")
    reused, missing, parent = parent_evidence(args, audit)
    controls = [identity(path) for path in
                (args.config, args.runtime_config, args.pause_receipt, args.code_manifest)]
    require(not any(Path(r["path"]).is_relative_to(args.output) for r in parent+code+controls),
            "An input overlaps the new output")
    with own_gpu() as owner:
        require(owner is not None, "Cannot identify follow-up GPU owner")
        args.output.mkdir(exist_ok=False)
        shutil.copyfile(args.original_root/"checkpoint_audit.json", args.output/"checkpoint_audit.json")
        shutil.copyfile(args.original_root/"equivalence.json", args.output/"equivalence.json")
        save(args.output/"authorization.json", dict(stage="PAUSED32K_MISSING_SIX_EVENT_FOLLOWUP",
            evaluation_commit=args.evaluation_commit, iteration=ITERATION, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, reused_parent_artifacts=parent,
            original_root=str(args.original_root), original_result_status="FAILED", original_result_sha256=ORIGINAL_RESULT_SHA256,
            partial_analysis_root=str(args.partial_root), partial_summary_sha256=PARTIAL_SUMMARY_SHA256,
            reused_cases=reused, cases=missing, reused_physical_cases=78, new_physical_cases=6,
            action_gate="REUSED_UNCHANGED21_GROUPS", gpu_lock_owner=owner,
            maximum_wall_seconds=1800, phase_wall_seconds=1800, maximum_output_bytes=1024**3,
            new_training_updates=0, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(status="RUNNING", phases=[], controller=owner, started_utc=now(),
            stage="MISSING_SIX_EVENT_REPLAY", reused_physical_cases=78, scheduled_physical_cases=6,
            original_campaign_status="FAILED", combined84_comparison="PENDING_SEPARATE_READ_ONLY_ANALYSIS")
        budget, requested, previous = MissingBudget(config, receipt, args.output, owner), False, dict(os.environ)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            base = [runtime["primebot_python"], "-u", str(HERE/"event_review.py"),
                "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--evaluation-commit", args.evaluation_commit, "--gpu-owner-pid", str(owner["pid"]),
                "--gpu-owner-ticks", owner["start_ticks"]]
            for event, method, horizon, clip in missing:
                extra = ["--event-kind", event, "--tail-method", method, "--horizon", str(horizon), "--clip", clip]
                for phase in ("replay", "media"):
                    require(all(identity(r["path"]) == r for r in controls), "Follow-up controls changed")
                    validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                    label = f"{event}_{method}_h{horizon}_{clip}_{phase}"
                    result["phase"] = label
                    save(args.output/"result.json", result)
                    budget.phase_started = time.monotonic()
                    value = guarded_run([*base, "--child", phase, *extra], ROOT, budget,
                        args.output/(label+".guard.json"), args.output/(label+".log"), check_interval_s=.5,
                        grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=label, guard=identity(args.output/(label+".guard.json"))))
                    require(value["status"] == "WORKER_COMPLETED" and value["cleanup"]["worker_reaped"] is True
                            and value["cleanup"]["worker_return_code"] == 0, "Follow-up phase failed: "+label)
            case_records = []
            for event, method, horizon, clip in missing:
                path = case_folder(args.output/event, horizon, clip, method)/"review/result.json"
                saved = read(path)
                require(saved.get("status") == "PASSED" and len(saved.get("cases", [])) == 1,
                        "Missing one of the six completed physical/media records")
                row = saved["cases"][0]
                require((row["event"], row["tail_method"], row["horizon_frames"], row["clip"])
                        == (event, method, horizon, clip), "Follow-up case identity differs")
                case_records.append(identity(path))
            require(all(identity(r["path"]) == r for r in parent+sources+code+controls),
                    "Preserved parent or follow-up sources changed")
            result.update(status="COMPLETED", source_integrity="PASSED", case_records=case_records,
                new_physical_cases=6, new_training_updates=0, user_accepted_checkpoint=None)
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
    for name in ("config", "runtime-config", "output", "pause-receipt", "audit-root", "code-manifest", "original-root", "partial-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    execute(parser.parse_args())
