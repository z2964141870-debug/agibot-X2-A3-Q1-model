"""Continue from an audited final state after the complete Q1 batch comparison."""

import hashlib
import json
from pathlib import Path


CONTRACT_KEYS = ("assets", "sim", "world_model", "policy", "critic", "wm_trainer",
                 "policy_trainer", "train_clips", "heldout_clips", "backend_receipt",
                 "reference_receipt", "motion_manifest")
MODEL_KEYS = ("wm", "policy", "critic")
OPTIMIZER_KEYS = ("wm_opt", "policy_opt", "critic_opt")


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check_identity(record):
    if identity(record["path"]) != record:
        raise ValueError(f"Frozen input changed: {record['path']}")


def validate_contract(source, target):
    for key in CONTRACT_KEYS:
        if json.dumps(source[key], sort_keys=True) != json.dumps(target[key], sort_keys=True):
            raise ValueError(f"Continuation contract differs: {key}")
    if source["assets"].get("contact_protocol", {}).get("version") != "Q1_FOOT_PRIORITY_V2":
        raise ValueError("Continuation requires the corrected Q1 contact protocol")


def validated_parent(pair_directory, parent_run):
    pair, run = Path(pair_directory).resolve(), Path(parent_run).resolve()
    comparison_path, plan_path = pair / "comparison.json", pair / "plan.json"
    terminal = json.loads((pair / "receipt.json").read_text())
    comparison = json.loads(comparison_path.read_text())
    plan = json.loads(plan_path.read_text())
    if terminal["status"] != "PAIR_COMPLETED" or terminal["comparison"] != identity(comparison_path):
        raise ValueError("Batch comparison has not completed successfully")
    if (comparison["status"] != "MATCHED_BATCH_COMPARISON_COMPLETED"
            or comparison["sole_setting_change_verified"] is not True
            or comparison["plan"] != identity(plan_path)):
        raise ValueError("Batch comparison identity or matched setting check failed")
    arms = [arm for arm in plan["arms"] if Path(arm["run"]).resolve() == run]
    if len(arms) != 1:
        raise ValueError("Parent is not a registered arm of this completed pair")
    arm = arms[0]
    final = identity(run / "final.pt")
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    audit_path = pair / (arm["name"] + "_training_audit.json")
    audit = json.loads(audit_path.read_text())
    if (receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION"
            or final not in receipt["artifacts"]
            or audit["status"] != "TRAINING_SAVE_AUDIT_PASSED"
            or not audit["gates"] or not all(audit["gates"].values())
            or audit["final"] != final or audit["training_receipt"] != identity(receipt_path)):
        raise ValueError("Parent final state lacks a matching successful save audit")
    evaluation_path = Path(arm["evaluations"]["final"]["receipt"])
    evaluation = json.loads(evaluation_path.read_text())
    trace_audit_path = pair / (arm["name"] + "_eval_final_audit.json")
    trace_audit = json.loads(trace_audit_path.read_text())
    if (evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED"
            or evaluation["checkpoint"] != final
            or trace_audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
            or trace_audit["verified"] is not True
            or trace_audit["receipt"] != identity(evaluation_path)):
        raise ValueError("Parent lacks matching independently audited physics evaluation")
    for row in evaluation["rows"]:
        check_identity(row["reference"])
        check_identity(row["trace"])
    # All pair outputs were bound by the completed controller's successful events.
    events = [json.loads(line) for line in (pair / "events.jsonl").read_text().splitlines()]
    for path in (audit_path, evaluation_path, trace_audit_path):
        expected = identity(path)
        if not any(event.get("result") == expected and event.get("exit_code") == 0 for event in events):
            raise ValueError(f"Completed pair event does not bind {path}")
    return {"checkpoint": final, "receipt": identity(receipt_path), "save_audit": identity(audit_path),
            "admission": identity(run / "admission.json"),
            "evaluation": identity(evaluation_path), "evaluation_audit": identity(trace_audit_path),
            "pair_receipt": identity(pair / "receipt.json"), "comparison": identity(comparison_path),
            "parent_counts": {key: receipt[key] for key in ("step", "wm_updates", "policy_updates")},
            "usability_gates": evaluation["usability_gates"]}


def load_parent(evidence, target, modules, optimizers, sim):
    import torch

    for key in ("checkpoint", "receipt", "save_audit", "admission", "evaluation", "evaluation_audit", "pair_receipt", "comparison"):
        check_identity(evidence[key])
    source = torch.load(evidence["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    admission = json.loads(Path(evidence["admission"]["path"]).read_text())
    if json.dumps(source["manifest"], sort_keys=True) != json.dumps(admission, sort_keys=True):
        raise ValueError("Parent admission differs from its audited checkpoint manifest")
    if any(source[key] != value for key, value in evidence["parent_counts"].items()):
        raise ValueError("Parent checkpoint counters do not match its final receipt")
    validate_contract(source["manifest"], target)
    for item in source["manifest"]["source_files"] + source["manifest"]["train_clips"]:
        check_identity(item)
    for key in MODEL_KEYS + OPTIMIZER_KEYS:
        values = source[key].values() if key in MODEL_KEYS else (
            value for state in source[key]["state"].values() for value in state.values())
        if any(not bool(torch.isfinite(value).all()) for value in values if torch.is_tensor(value)):
            raise ValueError(f"Nonfinite parent {key}")
    initial_steps = {}
    for key, module in zip(MODEL_KEYS, modules):
        module.load_state_dict(source[key], strict=True)
    for key, optimizer in zip(OPTIMIZER_KEYS, optimizers):
        optimizer.load_state_dict(source[key])
        counts = sorted({int(state["step"]) for state in optimizer.state.values()})
        if not counts or len(counts) != 1:
            raise ValueError(f"Missing or inconsistent parent optimizer counters: {key}")
        initial_steps[key] = counts
    for key in ("bin_return", "bin_memory"):
        if not bool(torch.isfinite(source[key]).all()):
            raise ValueError("Nonfinite parent return sampler")
        getattr(sim, key).copy_(source[key])
    return {"type": "WARM_START_AFTER_COMPLETED_MATCHED_PAIR", "evidence": evidence,
            "initial_optimizer_steps": initial_steps, "source_counters": evidence["parent_counts"],
            "learning_rate_step_offset": source["manifest"]["learning_rate_step_offset"] + source["step"],
            "sampler": "Parent return bins copied; fresh episode reset after step0 save.",
            "replay_and_rng": "Fresh replay and explicit new seed; not an exact process resume.",
            "initialization_entry": identity(__file__)}
