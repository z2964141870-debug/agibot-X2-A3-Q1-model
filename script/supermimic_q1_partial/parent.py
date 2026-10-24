"""Admit an audited R02 partial checkpoint without changing its failed state."""

import importlib.util
import json
from pathlib import Path


EXTENSION = Path(__file__).resolve().parents[1] / "supermimic_q1_extension"
_spec = importlib.util.spec_from_file_location("q1_completed_parent_contract", EXTENSION / "initialization.py")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
CONTRACT_KEYS = _legacy.CONTRACT_KEYS
MODEL_KEYS = _legacy.MODEL_KEYS
OPTIMIZER_KEYS = _legacy.OPTIMIZER_KEYS
identity = _legacy.identity
check_identity = _legacy.check_identity
validate_contract = _legacy.validate_contract


def validated_parent(pair_directory, parent_run):
    pair, run = Path(pair_directory).resolve(), Path(parent_run).resolve()
    comparison_path, plan_path = pair / "comparison.json", pair / "plan.json"
    plan, comparison = json.loads(plan_path.read_text()), json.loads(comparison_path.read_text())
    state = json.loads((pair / "status.json").read_text())
    audit_path = Path(plan["checkpoint_audit"]["path"])
    check_identity(plan["checkpoint_audit"])
    check_identity(plan["spec"])
    spec_path = Path(plan["spec"]["path"])
    spec = json.loads(spec_path.read_text())
    audit = json.loads(audit_path.read_text())
    check_identity(audit["source"])
    check_identity(plan["source"])
    failure_path = run / "receipt.json"
    failure = json.loads(failure_path.read_text())
    controller_path = run.parent / "controller_status.json"
    controller = json.loads(controller_path.read_text())
    if (spec_path.parent != run.parent or controller["status"] != "FAILED" or controller["phase"] != "train"
            or failure["status"] != "FAILED" or failure["step"] != 5300
            or failure["error"] != "RuntimeError: Container memory protection: 33833570304 bytes"):
        raise ValueError("Expected the unchanged R02 memory-protection failure, not a completed parent")
    if (audit["status"] != "PARTIAL_TRAINING_CHECKPOINT_AUDIT_PASSED" or audit["checkpoint_step"] != 5000
            or not audit["gates"] or not all(audit["gates"].values()) or audit["final_budget_completed"]
            or audit["frozen_changes"] or audit["collection_transitions"] != 320000
            or audit["wm_updates"] != 1949 or audit["policy_updates"] != 1899):
        raise ValueError("The independently audited step5000 partial save is required")
    checkpoint = identity(run / "step_005000.pt")
    if checkpoint != audit["files"][1] or checkpoint not in failure["artifacts"]:
        raise ValueError("Partial checkpoint does not match both independent audit and original failure receipt")
    names = spec["split_clips"]["development"]
    if (state["status"] != "COMPLETED" or state["comparison"] != identity(comparison_path)
            or comparison["status"] != "DEVELOPMENT_PARTIAL_CHECKPOINT_MATCHED_PHYSICS_COMPLETED"
            or comparison["plan"] != identity(plan_path) or comparison["training_complete"]
            or comparison["test_evaluated"] or plan["clips"] != names or plan["test_clips"]
            or sorted(row["clip"] for row in comparison["pairs"]) != sorted(names)
            or any(row["split"] != "development" for row in comparison["pairs"])):
        raise ValueError("Require the genuine, completed development-only partial-save comparison")
    for row in plan["checkpoints"] + spec["frozen_files"] + comparison["evidence"]:
        check_identity(row)
    evaluation_path = pair / "evaluation_step5000/receipt.json"
    trace_audit_path = pair / "evaluation_audit_step5000.json"
    evaluation = json.loads(evaluation_path.read_text())
    trace_audit = json.loads(trace_audit_path.read_text())
    if (evaluation["checkpoint"] != checkpoint or evaluation["training_step"] != 5000
            or trace_audit["status"] != "EVALUATION_TRACE_AUDIT_PASSED"
            or not trace_audit["verified"] or trace_audit["receipt"] != identity(evaluation_path)
            or sorted(row["clip"] for row in evaluation["rows"]) != sorted(names)):
        raise ValueError("Partial parent evaluation is not bound to its independent trajectory audit")
    for row in evaluation["rows"]:
        check_identity(row["reference"])
        check_identity(row["trace"])
    return {"checkpoint": checkpoint, "receipt": identity(failure_path), "save_audit": identity(audit_path),
            "admission": identity(run / "admission.json"), "evaluation": identity(evaluation_path),
            "evaluation_audit": identity(trace_audit_path), "development_plan": identity(plan_path),
            "development_status": identity(pair / "status.json"), "comparison": identity(comparison_path),
            "failed_controller": identity(controller_path), "original_spec": identity(spec_path),
            "parent_counts": {"step": 5000, "wm_updates": 1949, "policy_updates": 1899},
            "usability_gates": evaluation["usability_gates"]}


def load_parent(evidence, target, modules, optimizers, sim):
    import torch

    for record in evidence.values():
        if isinstance(record, dict) and {"path", "bytes", "sha256"} <= record.keys():
            check_identity(record)
    source = torch.load(evidence["checkpoint"]["path"], map_location="cpu", mmap=True, weights_only=False)
    admission = json.loads(Path(evidence["admission"]["path"]).read_text())
    if (json.dumps(source["manifest"], sort_keys=True) != json.dumps(admission, sort_keys=True)
            or any(source[key] != value for key, value in evidence["parent_counts"].items())):
        raise ValueError("Partial checkpoint manifest or saved counters differ")
    validate_contract(source["manifest"], target)
    for record in source["manifest"]["source_files"] + source["manifest"]["train_clips"]:
        check_identity(record)
    for key in MODEL_KEYS + OPTIMIZER_KEYS:
        values = source[key].values() if key in MODEL_KEYS else (
            value for state in source[key]["state"].values() for value in state.values())
        if any(not bool(torch.isfinite(value).all()) for value in values if torch.is_tensor(value)):
            raise ValueError(f"Nonfinite parent state: {key}")
    counters = {}
    expected = {"wm_opt": 19490, "policy_opt": 1899, "critic_opt": 37980}
    for key, module in zip(MODEL_KEYS, modules):
        module.load_state_dict(source[key], strict=True)
    for key, optimizer in zip(OPTIMIZER_KEYS, optimizers):
        optimizer.load_state_dict(source[key])
        counts = sorted({int(state["step"]) for state in optimizer.state.values()})
        if counts != [expected[key]]:
            raise ValueError(f"Partial parent Adam counter mismatch: {key}")
        counters[key] = counts
    for key in ("bin_return", "bin_memory"):
        if not bool(torch.isfinite(source[key]).all()):
            raise ValueError("Nonfinite parent reference sampler")
        getattr(sim, key).copy_(source[key])
    return {"type": "WARM_START_FROM_AUDITED_PARTIAL_AFTER_MEMORY_PROTECTION", "evidence": evidence,
            "initial_optimizer_steps": counters, "source_counters": evidence["parent_counts"],
            "learning_rate_step_offset": source["manifest"]["learning_rate_step_offset"] + source["step"],
            "sampler": "Parent return bins copied, then fresh episodes after step0 save.",
            "replay_and_rng": "Fresh replay and new explicit seed; not an exact process resume.",
            "parent_terminal_state": "FAILED_UNCHANGED_NOT_COMPLETED", "initialization_entry": identity(__file__)}
