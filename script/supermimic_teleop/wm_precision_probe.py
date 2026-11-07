"""A separate bounded precision diagnostic; preserve the original failed screen."""

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
from paused_runtime import own_gpu
from preview_review import environment
from runtime_guard import guarded_run
from student_cost import reserve, settle
from wm_probe import ProbeBudget, probe_preflight

NEW_CODE = ("wm_precision_probe.py", "wm_precision_worker.py", "wm_precision_math.py")


def parent_evidence(root, settings):
    files = [root/name for name in ("result.json", "probe.guard.json", "probe_result.json", "authorization.json")]
    result, guard, probe, authorization = map(read, files)
    require(result.get("status") == probe.get("status") == "WM_PROBE_COMPLETED"
        and result.get("source_integrity") == probe.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and guard.get("status") == "WORKER_COMPLETED"
        and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists()
        and identity(root/"probe_result.json") == result["probe_result"] == settings["parent_result"]
        and probe.get("diagnostic_screen") == "FAILED" and probe.get("sampled_transition_count") == 256
        and probe.get("cuda_wm_step_forward_calls") == 368 and probe.get("cuda_backward_calls") == 4
        and probe.get("actual_optimizer_updates") == probe.get("new_physical_transitions") == 0
        and probe.get("test_sources_read") is False, "Use only the settled original native probe")
    protected = [row for name in ("evaluation_sources", "training_source_inputs", "controller_inputs")
        for row in authorization[name]]
    require(all(identity(r["path"]) == r for r in protected), "Original WM probe protected inputs changed")
    return [identity(p) for p in files]+protected


def precision_preflight(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = probe_preflight(args)
    settings = read(args.precision_plan)
    require(args.precision_plan == HERE/"wm_precision_probe_plan_20261107.json"
        and identity(args.precision_plan)["sha256"] == args.precision_plan_sha256
        and settings.get("schema") == "q1_frozen_wm_precision_probe_v1"
        and args.output == Path(settings["output"])
        and args.parent_root/"probe_result.json" == Path(settings["parent_result"]["path"])
        and settings.get("maximum_wall_seconds") == 300
        and settings.get("maximum_output_bytes") == 67108864
        and settings.get("expected_wm_step_calls") == 376 and settings.get("maximum_wm_step_calls") == 512
        and settings.get("maximum_backward_calls") == 8 and settings.get("same_selected_rows") == 256
        and settings.get("finite_difference_epsilons") == [.01, .001]
        and settings.get("gradient_relative_tolerance") == .05
        and settings.get("gradient_absolute_tolerance") == .002
        and settings.get("test_sources_read") is False
        and settings.get("new_physical_transitions") == settings.get("optimizer_updates") == 0,
        "Use the independently registered exact precision comparison")
    require({str(HERE/n) for n in NEW_CODE} <= {r["path"] for r in code}
        and not {str(HERE/n) for n in NEW_CODE} & {r["path"] for r in sources},
        "New precision sources are missing or overlap original frozen inputs")
    controls += parent_evidence(args.parent_root, settings)+[identity(args.precision_plan)]
    unique = {}
    for row in controls:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting precision input identities")
        unique[row["path"]] = row
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = precision_preflight(args)
    with own_gpu() as owner:
        owner = dict(owner, boot=config["expected_boot"], gpu=config["expected_gpu"])
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"reused_dataset.json", dataset)
        budget = ProbeBudget(config, receipt, args.output, owner)
        budget.check()
        started = time.monotonic_ns()
        reservation = reserve(args.cost_ledger, run_id=str(args.output), stage="r3", owner=owner,
            gpu=config["expected_gpu"], commit=args.evaluation_commit, started_ns=started)
        budget.maximum_wall = min(300, max(0, reservation["reserved_seconds"]-5))
        deadline = started+max(0, budget.maximum_wall-15)*1_000_000_000
        save(args.output/"authorization.json", dict(stage="R3_FROZEN_WM_PRECISION_COMPARISON",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, gpu_lock_owner=owner,
            cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            maximum_output_bytes=67108864, maximum_cuda_wm_forward_calls=512, maximum_cuda_backward_calls=8,
            cache_boundary_reused=identity(args.initial_r2_root/"cache_boundary.json"),
            new_physical_transitions=0, actual_optimizer_updates=0, test_sources_read=False,
            memory_reclaim_attempted=False, original_training_restarted=False,
            final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_FROZEN_WM_PRECISION_COMPARISON", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, actual_optimizer_updates=0,
            user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {n:signal.signal(n, stop) for n in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"precision.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["morph_python"], "-u", str(HERE/"wm_precision_worker.py")]
            values = {"runtime-config":args.runtime_config, "config":args.config,
                "plan":Path(config["output"])/"e1024_train_plan.json", "output":args.output,
                "training-root":args.training_root, "aggregate-root":args.aggregate_root,
                "probe-plan":args.probe_plan, "precision-plan":args.precision_plan,
                "precision-plan-sha256":args.precision_plan_sha256, "parent-root":args.parent_root,
                "expected-gpu":config["expected_gpu"], "gpu-owner-pid":owner["pid"],
                "gpu-owner-ticks":owner["start_ticks"], "deadline-ns":deadline}
            for key, value in values.items():
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"precision.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded precision worker did not complete")
            require(all(identity(r["path"]) == r for r in controls+sources+code), "Precision protected sources changed")
            probe = read(args.output/"precision_result.json")
            require(probe.get("status") == "WM_PRECISION_COMPLETED" and probe.get("source_integrity") == "PASSED"
                and probe.get("cuda_wm_step_forward_calls") == 376 and probe.get("cuda_backward_calls") == 8
                and probe.get("sampled_transition_count") == 256
                and probe.get("new_physical_transitions") == probe.get("actual_optimizer_updates") == 0,
                "No complete bounded precision comparison")
            result.update(status="WM_PRECISION_COMPLETED", diagnostic_screen=probe["diagnostic_screen"],
                source_integrity="PASSED", precision_result=identity(args.output/"precision_result.json"),
                natural_teleoperation_qualified=False)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            try:
                if guard_path.exists() and read(guard_path).get("cleanup", {}).get("worker_reaped") is True:
                    result["cost_settlement"] = settle(args.cost_ledger, run_id=str(args.output),
                        owner=owner, cleanup=read(guard_path)["cleanup"])
                result["ended_utc"] = now()
                save(args.output/"result.json", result)
            finally:
                os.environ.clear()
                os.environ.update(previous)
                for n, handler in handlers.items():
                    signal.signal(n, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output", "pause-receipt", "audit-root", "code-manifest",
            "training-root", "causal-plan", "source-split", "cost-ledger", "r1-analysis-root",
            "initial-r2-root", "aggregate-root", "probe-plan", "precision-plan", "parent-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256", "probe-plan-sha256", "precision-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
