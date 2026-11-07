"""One bounded, source-gated saved-transition WM diagnostic after the user pause."""

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
from runtime_guard import allocated_tree, guarded_run
from student_auxiliary_training import aggregate_evidence
from student_causal_review import trained_causal_evidence
from student_causal_training import CausalBudget, train_preflight
from student_cost import reserve, settle

NEW_CODE = ("wm_probe.py", "wm_probe_worker.py", "wm_probe_data.py", "wm_probe_math.py")


def probe_preflight(args):
    values = train_preflight(args)
    config, runtime, receipt, sources, audit, code, controls, dataset = values
    settings = read(args.probe_plan)
    root = ROOT/"data/supermimic_for_q1"
    require(args.initial_r2_root == root/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107"
        and args.aggregate_root == root/"Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_AGGREGATE01_20261107"
        and args.probe_plan == HERE/"wm_probe_plan_20261107.json"
        and identity(args.probe_plan)["sha256"] == args.probe_plan_sha256
        and settings.get("schema") == "q1_saved_transition_physical_wm_probe_v1"
        and settings.get("maximum_wall_seconds") == 300
        and settings.get("maximum_output_bytes") == 67108864
        and settings.get("maximum_sampled_transitions") == 256
        and settings.get("samples_per_group") == 64 and settings.get("maximum_forward_calls") == 512
        and settings.get("test_sources_read") is False
        and settings.get("optimizer_updates") == settings.get("new_physical_transitions") == 0,
        "Use the exact independently registered WM diagnostic")
    require({str(HERE/n) for n in NEW_CODE} <= {r["path"] for r in code}
        and not {str(HERE/n) for n in NEW_CODE} & {r["path"] for r in sources},
        "Missing independent probe sources or overlap with original training inputs")
    expected = {str(Path(runtime["adapted_source"])/"src/supermimic"/name)
        for name in ("simulator.py", "randomization.py")}
    expected.add(str(Path(runtime["morph_source"])/"src/morphmimic/env/observations.py"))
    require({r["path"] for r in settings["context_sources"]} == expected
        and all(identity(r["path"]) == r for r in settings["context_sources"]), "Nominal context source evidence differs")
    controls += trained_causal_evidence(args.initial_r2_root, args.training_root)
    controls += aggregate_evidence(args.aggregate_root)+[identity(args.probe_plan)]+settings["context_sources"]
    unique = {}
    for row in controls:
        require(row["path"] not in unique or unique[row["path"]] == row, "Conflicting WM probe source identities")
        unique[row["path"]] = row
    require(all(identity(r["path"]) == r for r in unique.values()), "WM probe admission source changed")
    return config, runtime, receipt, sources, audit, code, list(unique.values()), dataset


class ProbeBudget(CausalBudget):
    def __init__(self, *args):
        super().__init__(*args)
        self.maximum_wall = 300

    def check(self):
        require(allocated_tree(self.output)["unique_allocated_bytes"] < 67108864, "WM probe64MiB output bound reached")
        return super().check()


def execute(args):
    config, runtime, receipt, sources, audit, code, controls, dataset = probe_preflight(args)
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
        save(args.output/"authorization.json", dict(stage="R3_FROZEN_PHYSICAL_WM_PROBE",
            evaluation_commit=args.evaluation_commit, evaluation_sources=code,
            training_source_inputs=sources, controller_inputs=controls, gpu_lock_owner=owner,
            cost_reservation=reservation, maximum_wall_seconds=budget.maximum_wall,
            maximum_output_bytes=67108864, maximum_cuda_wm_forward_calls=512,
            new_physical_transitions=0, actual_optimizer_updates=0, test_sources_read=False,
            cache_boundary_reused=identity(args.initial_r2_root/"cache_boundary.json"),
            memory_reclaim_attempted=False, original_training_restarted=False,
            final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(stage="R3_FROZEN_PHYSICAL_WM_PROBE", status="RUNNING", controller=owner,
            started_utc=now(), new_physical_transitions=0, actual_optimizer_updates=0,
            user_accepted_checkpoint=None)
        requested, previous = False, dict(os.environ)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {n:signal.signal(n, stop) for n in (signal.SIGTERM, signal.SIGINT)}
        guard_path = args.output/"probe.guard.json"
        try:
            os.environ.update(environment(runtime), CUDA_VISIBLE_DEVICES=config["expected_gpu"])
            command = [runtime["morph_python"], "-u", str(HERE/"wm_probe_worker.py")]
            values = {"runtime-config":args.runtime_config, "config":args.config,
                "plan":Path(config["output"])/"e1024_train_plan.json", "output":args.output,
                "training-root":args.training_root, "aggregate-root":args.aggregate_root,
                "probe-plan":args.probe_plan, "probe-plan-sha256":args.probe_plan_sha256,
                "expected-gpu":config["expected_gpu"], "gpu-owner-pid":owner["pid"],
                "gpu-owner-ticks":owner["start_ticks"], "deadline-ns":deadline}
            for key, value in values.items():
                command.extend(("--"+key, str(value)))
            save(args.output/"result.json", result)
            guard = guarded_run(command, ROOT, budget, guard_path, args.output/"probe.log",
                check_interval_s=.5, grace_s=5, stop_requested=lambda:requested)
            require(guard.get("status") == "WORKER_COMPLETED"
                and guard.get("cleanup", {}).get("worker_reaped") is True
                and guard["cleanup"].get("worker_return_code") == 0, "Bounded WM probe worker did not complete")
            require(all(identity(r["path"]) == r for r in controls+sources+code), "WM probe protected sources changed")
            probe = read(args.output/"probe_result.json")
            require(probe.get("status") == "WM_PROBE_COMPLETED"
                and probe.get("source_integrity") == "PASSED"
                and probe.get("cuda_wm_step_forward_calls", 513) <= 512
                and probe.get("sampled_transition_count", 257) <= 256
                and probe.get("new_physical_transitions") == probe.get("actual_optimizer_updates") == 0,
                "No bounded complete saved-transition diagnostic")
            result.update(status="WM_PROBE_COMPLETED", diagnostic_screen=probe["diagnostic_screen"],
                source_integrity="PASSED", probe_result=identity(args.output/"probe_result.json"),
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
            "initial-r2-root", "aggregate-root", "probe-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("evaluation-commit", "manifest-sha256", "probe-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    execute(parser.parse_args())
