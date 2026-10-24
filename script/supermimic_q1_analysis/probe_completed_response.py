#!/usr/bin/env python3
"""Bind the preserved action-response experiment to completed R03 and V2 contacts."""

import argparse
import json
from pathlib import Path
import sys

from diagnose_completed_run import verified_evaluation
from render_audited_trace import checked_file, identity


def execute_bound(delegate, constructor, args):
    original = delegate.Q1Assets
    try:
        delegate.Q1Assets = constructor
        delegate.execute(args)
    finally:
        delegate.Q1Assets = original


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root, experiment, output = (p.resolve() for p in
                                (args.root, args.experiment, args.output))
    if not all(p.is_relative_to(root / "data/supermimic_for_q1")
               for p in (experiment, output)):
        raise ValueError("Use independent Q1 artifact locations")
    sys.path.insert(0, str(root / "script/supermimic_for_q1"))
    sys.path.insert(0, str(root / "script/supermimic_q1_extension"))
    import pipeline
    original_plan = json.loads((experiment / "plan.json").read_text())
    pipeline.verify_plan(original_plan)
    if json.loads((experiment / "receipt.json").read_text())["status"] != "CONTINUATION_PIPELINE_COMPLETED":
        raise ValueError("Continuation has not completed")
    summary = json.loads((experiment / "summary.json").read_text())
    if summary["status"] != "CONTINUATION_PIPELINE_COMPLETED":
        raise ValueError("Continuation summary has not completed")
    final = summary["evaluations"]["final"]
    evaluation = verified_evaluation(final["receipt"], final["audit"])
    admission = json.loads((Path(evaluation["checkpoint"]["path"]).parent / "admission.json").read_text())
    import numpy as np
    import probe_action_response as delegate
    from contact_protocol import ContactCorrectedAssets
    assets = ContactCorrectedAssets(root)
    if assets.manifest() != admission["assets"]:
        raise ValueError("V2 contact assets differ from the final admission")
    backend_record = admission["backend_receipt"]
    backend = json.loads(checked_file(backend_record).read_text())
    for record in backend["source_files"]:
        checked_file(record)
    baseline_oom = original_plan["baseline_health"]["oom_kill"]

    def health():
        sample = pipeline.runner.health(root, baseline_oom)
        if (sample["oom_kill"] != baseline_oom
                or sample["cgroup_memory_bytes"] >= 30 * 1024**3 - 896 * 1024**2
                or sample["gpu_temperature_c"] >= 86
                or sample["gpu_free_mib"] < 2048):
            raise RuntimeError("Independent response probe resource guard refused")
        return sample

    if args.mode == "prepare":
        if output.exists():
            raise FileExistsError(output)
        rows = [row for row in evaluation["rows"] if row["split"] == "training"]
        if len(rows) != 2:
            raise ValueError("Expected the two registered training clips")
        rng = np.random.default_rng(20261023)
        samples = []
        for row in rows:
            if row["executed_steps"] < delegate.HORIZON + 2:
                raise ValueError("Saved clip is too short for the fixed sampling plan")
            for fraction in (.25, .65):
                directions = rng.normal(size=(4, assets.model.nu))
                directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
                samples.append({"clip": row["clip"], "fraction": fraction,
                    "start": int(np.floor(fraction * (row["executed_steps"] - delegate.HORIZON))),
                    "reference": row["reference"], "trace": row["trace"],
                    "directions": directions.tolist()})
        budget = len(samples) * len(delegate.VARIANTS) * delegate.HORIZON
        plan = {"status": "PREPARED_NOT_EXECUTED", "source": identity(__file__),
            "delegate_source": identity(delegate.__file__), "root": str(root),
            "binding_sources": [identity(Path(__file__).with_name(name)) for name in
                                ("diagnose_completed_run.py", "render_audited_trace.py")],
            "checkpoint": evaluation["checkpoint"], "evaluation": final["receipt"],
            "evaluation_audit": final["audit"], "original_plan": identity(experiment / "plan.json"),
            "backend": backend_record, "sources": backend["source_files"],
            "assets": assets.manifest(), "samples": samples, "horizon": delegate.HORIZON,
            "readouts": delegate.READOUTS, "max_control_steps": budget,
            "max_physics_substeps": budget * 20, "initial_health": health(),
            "protocol": "R03 final; Q1_FOOT_PRIORITY_V2; effective foot friction1, zero encoder/COM offsets. Four original-state baselines must replay before perturbing only the first action. Original directions, magnitudes and H1/8/32 readouts; no policy inference or optimizer.",
            "limits": "Local visited-state open-loop action sensitivity, not full actor-gradient accuracy or usable walking. A failed or admitted output is never reused."}
        output.mkdir(parents=True, exist_ok=False)
        delegate.write_json(output / "plan.json", plan)
        print(json.dumps({"status": plan["status"], "plan": identity(output / "plan.json"),
                          "max_control_steps": budget}), flush=True)
        return

    plan = json.loads((output / "plan.json").read_text())
    for record in (plan["source"], plan["delegate_source"], plan["evaluation_audit"],
                   plan["original_plan"], *plan["binding_sources"]):
        checked_file(record)
    if (plan["source"] != identity(__file__)
            or plan["delegate_source"] != identity(delegate.__file__)
            or plan["checkpoint"] != evaluation["checkpoint"]
            or plan["evaluation"] != final["receipt"] or plan["assets"] != assets.manifest()):
        raise ValueError("Prepared R03 binding differs from the current inputs")
    before = health()
    delegated_args = argparse.Namespace(root=root, output=output, device=args.device)
    execute_bound(delegate, ContactCorrectedAssets, delegated_args)
    import audit_action_response
    old_argv = sys.argv
    try:
        sys.argv = [str(audit_action_response.__file__), "--input",
                    str(output / "receipt.json"), "--output", str(output / "summary_audit.json")]
        audit_action_response.main()
    finally:
        sys.argv = old_argv
    pipeline.verify_plan(original_plan)
    result = {"status": "R03_V2_ACTION_RESPONSE_AND_AUDIT_COMPLETED",
        "plan": identity(output / "plan.json"), "receipt": identity(output / "receipt.json"),
        "audit": identity(output / "summary_audit.json"), "source": identity(__file__),
        "delegate_source": identity(delegate.__file__), "initial_health": before,
        "final_health": health(), "frozen_verified_before_and_after": True,
        "new_policy_inference": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY"}
    delegate.write_json(output / "completion.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
