"""Review only the changed coverage-selected f0 encoder on seven fixed tasks."""

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
from paused_preview import preflight
from paused_runtime import own_gpu
from preview_review import environment, validate_code_manifest
from runtime_guard import guarded_run
from student_causal_review import (CausalReviewBudget, baseline_cases, child_command,
    folder, review_cases)
from student_causal_training import saved_analysis_gate
from student_review import training_evidence

STAGE = "COVERAGE01_F0_MATCHED_CLOSED_LOOP"


def selection_gate(fit, trace):
    require(fit.get("method") == "owned_source_coverage" and fit.get("actual_updates") == 3000
        and fit.get("actual_adam_steps") == [3000]*6
        and fit.get("selection_split") == "validation" and fit.get("test_used_for_selection") is False
        and fit.get("human_real_future_frames") == 0 and fit.get("normalization_refitted") is False,
        "Require completed coverage training with causal inputs and unchanged selection")
    require(trace and trace[0]["update"] == 0 and trace[-1]["update"] == fit["actual_updates"],
        "Missing actual coverage validation endpoints")
    initial = [trace[0]["fixed_anchor_r1_risk"], trace[0]["fixed_anchor_mixed_risk"]]
    eligible = [r for r in trace if r["fixed_anchor_r1_risk"] <= initial[0]
        and r["fixed_anchor_mixed_risk"] <= initial[1]]
    best = min(eligible, key=lambda r:r["validation_risk"])
    require(best["update"] == fit["selected_update"] > 0
        and best["validation_risk"] == fit["selected_validation_risk"] < trace[0]["validation_risk"]
        and fit["initial_fixed_anchor_risks"] == initial
        and fit["selected_fixed_anchor_risks"] == [best["fixed_anchor_r1_risk"], best["fixed_anchor_mixed_risk"]],
        "Only the actual eligible validation-selected encoder may enter new closed loop")
    return best


def coverage_evidence(root, plan):
    result, fit, guard, auth, trace = (read(root/n) for n in
        ("result.json", "fit_result.json", "fit.guard.json", "authorization.json", "training_trace.json"))
    selected = selection_gate(fit, trace)
    require(result.get("stage") == auth.get("stage") == "R3_OWNED_SOURCE_COVERAGE"
        and result.get("status") == fit.get("status") == "TRAINED_PENDING_CLOSED_LOOP"
        and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and result.get("fit_result") == identity(root/"fit_result.json")
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"].get("worker_return_code") == 0
        and not Path("/proc", str(result["controller"]["pid"])).exists(),
        "Wait for real coverage completion, source gate, settlement and cleanup")
    require(plan.get("schema") == "q1_coverage_selected_review_v1"
        and plan.get("training_root") == str(root)
        and plan.get("fit_result") == identity(root/"fit_result.json")
        and plan.get("selected_checkpoint") == identity(root/"selected.pt")
        and plan.get("selected_update") == selected["update"] == 500
        and plan.get("maximum_new_cases") == 7 and plan.get("maximum_wall_seconds") == 1800
        and plan.get("new_optimizer_updates") == 0,
        "Use the published exact coverage checkpoint and bounded review plan")
    collection = read(root/"collection_result.json")
    require(collection.get("status") == "PASSED" and collection.get("actual_physical_steps") == 9585
        and len(collection.get("clips", [])) == 30
        and read(root/"dataset_receipt.json").get("status") == "COVERAGE_TEACHER_LABELS_SAVED",
        "Missing actual coverage source collection")
    records = [identity(root/n) for n in ("result.json", "fit_result.json", "fit.guard.json", "authorization.json",
        "training_trace.json", "selected.pt", "current_statistics.json", "collection_result.json", "dataset_receipt.json")]
    records += auth["evaluation_sources"]+auth["training_source_inputs"]+auth["controller_inputs"]
    require(all(identity(r["path"]) == r for r in records), "Coverage training or protected source changed")
    return records


def changed_encoder(root, original):
    import torch
    from upper_mps_forecast import audit_cpu_tensors

    values = [torch.load(p/"selected.pt", map_location="cpu", weights_only=True) for p in (root, original)]
    for value in values:
        audit_cpu_tensors(torch, value)
    current, previous = values
    keys = ("schema", "interface", "current_statistics", "human_real_future_frames", "history_frames",
            "history_frame_dim", "decoder_sha256", "frozen_teacher_policy_sha256")
    require(all(current[k] == previous[k] for k in keys)
        and current["provenance"]["teacher"] == previous["provenance"]["teacher"]
        and current["selected_update"] == 500, "Coverage changed frozen metadata or selected encoder")
    require(current["encoder"].keys() == previous["encoder"].keys(), "Encoder structure changed")
    differences = {k:float((v-previous["encoder"][k]).abs().max()) for k,v in current["encoder"].items()}
    require(any(differences.values()) and differences["mean"] == differences["scale"] == 0
        and not torch.cuda.is_initialized(), "Require changed finite encoder and unchanged normalization on CPU")
    return dict(status="CHANGED_SELECTED_ENCODER", selected_update=500, tensor_maximum_abs_difference=differences,
        frozen_policy_metadata_equal=True, cuda_initialized=False,
        files=[identity(p/"selected.pt") for p in (root, original)])


def admission(args):
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    settings = read(args.review_plan)
    require(args.review_plan == HERE/"student_coverage_review_plan_20261107.json"
        and identity(args.review_plan)["sha256"] == args.review_plan_sha256
        and args.causal_training_root == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_20261107"
        and args.output == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_REVIEW_20261107",
        "Use the unique published coverage review paths")
    parent = training_evidence(args.training_root)
    protected = parent+saved_analysis_gate(args.r1_analysis_root, parent)
    protected += coverage_evidence(args.causal_training_root, settings)
    original = Path(settings["original_r2_root"])
    require(original == ROOT/"data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R2_20261107",
        "Use the original R2 encoder for changed-policy comparison")
    selected = changed_encoder(args.causal_training_root, original)
    protected += selected["files"]
    controls = [identity(p) for p in (args.config, args.runtime_config, args.pause_receipt,
        args.code_manifest, args.review_plan)]
    required = {str(HERE/n) for n in ("student_coverage_review.py", "student_causal_review.py", "student_causal_review_worker.py")}
    require(required <= {r["path"] for r in code} and not required & {r["path"] for r in sources},
        "Missing independent review sources or frozen input overlap")
    baselines = baseline_cases(read(args.r1_analysis_root/"summary.json"))
    return config, runtime, receipt, sources, audit, code, protected, controls, baselines, selected


def execute(args):
    config, runtime, receipt, sources, audit, code, protected, controls, baselines, selected = admission(args)
    with own_gpu() as owner:
        args.output.mkdir(exist_ok=False)
        save(args.output/"checkpoint_audit.json", audit)
        save(args.output/"baseline_cases.json", baselines)
        save(args.output/"selected_encoder_audit.json", selected)
        controls += [identity(args.output/n) for n in ("baseline_cases.json", "selected_encoder_audit.json")]
        save(args.output/"authorization.json", dict(stage=STAGE, evaluation_commit=args.evaluation_commit,
            evaluation_sources=code, training_source_inputs=sources, controller_inputs=controls,
            preserved_research_inputs=protected, cases=review_cases(), new_physical_cases=7,
            reused_baseline_cases=14, maximum_wall_seconds=1800, phase_wall_seconds=1800,
            maximum_output_bytes=3*1024**3, gpu_lock_owner=owner, new_optimizer_updates=0,
            human_real_future_frames=0, user_accepted_checkpoint=None))
        budget, previous, requested = CausalReviewBudget(config, receipt, args.output, owner), dict(os.environ), False
        result = dict(status="RUNNING", stage=STAGE, controller=owner, phases=[], cases=[], started_utc=now(),
            reused_baseline_cases=14, natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
        def stop(signum, frame):
            nonlocal requested
            requested = True
        handlers = {s:signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            for case in review_cases():
                args.event_kind, args.clip = case
                for phase in ("replay", "media"):
                    require(all(identity(r["path"]) == r for r in protected+controls), "Coverage review evidence changed")
                    validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                    name = f"{case[0]}_f0_student_h0_{case[1]}_{phase}"
                    result["phase"] = name
                    save(args.output/"result.json", result)
                    budget.phase_started = time.monotonic()
                    guard = guarded_run(child_command(args, runtime, owner, case, phase), ROOT, budget,
                        args.output/(name+".guard.json"), args.output/(name+".log"), check_interval_s=.5,
                        grace_s=5, stop_requested=lambda:requested)
                    result["phases"].append(dict(name=name, guard=identity(args.output/(name+".guard.json"))))
                    require(guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
                        and guard["cleanup"].get("worker_return_code") == 0, "Coverage review phase failed: "+name)
                result["cases"].append(identity(folder(args)/"review/result.json"))
            require(all(identity(r["path"]) == r for r in protected+controls+sources+code), "Coverage review source changed")
            result.update(status="COMPLETED", source_integrity="PASSED", new_physical_cases=7,
                total_comparison_cases=21, new_optimizer_updates=0, quality_conclusion="PENDING_SAVED_ANALYSIS")
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
    for name in ("config", "runtime-config", "output", "causal-training-root", "training-root", "r1-analysis-root",
            "pause-receipt", "audit-root", "code-manifest", "review-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("manifest-sha256", "evaluation-commit", "review-plan-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.set_defaults(horizon=0, tail_method="masked")
    execute(parser.parse_args())
