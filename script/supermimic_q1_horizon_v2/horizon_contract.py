"""Admit completed R03 as the common parent of a bounded horizon comparison."""

import copy
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT / "supermimic_for_q1"))
sys.path.insert(0, str(SCRIPT / "supermimic_q1_extension"))
sys.path.insert(0, str(SCRIPT / "supermimic_q1_analysis"))

import initialization
import pipeline as continuation
from diagnose_completed_run import verified_evaluation

identity = initialization.identity
check_identity = initialization.check_identity
CONTRACT_KEYS = initialization.CONTRACT_KEYS


def validated_parent(experiment, run):
    experiment, run = Path(experiment).resolve(), Path(run).resolve()
    if run != experiment / "run":
        raise ValueError("Expected the registered completed continuation run")
    plan = json.loads((experiment / "plan.json").read_text())
    continuation.verify_plan(plan)
    terminal = json.loads((experiment / "receipt.json").read_text())
    summary_path = experiment / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (terminal["status"] != "CONTINUATION_PIPELINE_COMPLETED"
            or terminal["summary"] != identity(summary_path)
            or summary != continuation.summarize(plan)):
        raise ValueError("Parent continuation lacks its completed, bound summary")
    evaluation = verified_evaluation(summary["evaluations"]["final"]["receipt"],
                                     summary["evaluations"]["final"]["audit"])
    receipt_path = run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    final = identity(run / "final.pt")
    saved = json.loads((experiment / "training_audit.json").read_text())
    if (receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION"
            or final not in receipt["artifacts"] or evaluation["checkpoint"] != final
            or saved["final"] != final or saved["training_receipt"] != identity(receipt_path)):
        raise ValueError("Parent save and evaluation identities differ")
    return {"checkpoint": final, "receipt": identity(receipt_path),
            "save_audit": identity(experiment / "training_audit.json"),
            "admission": identity(run / "admission.json"),
            "evaluation": summary["evaluations"]["final"]["receipt"],
            "evaluation_audit": summary["evaluations"]["final"]["audit"],
            "pair_receipt": identity(experiment / "receipt.json"),
            "comparison": identity(summary_path),
            "parent_protocol": "COMPLETED_CONTINUATION_R03",
            "compatibility_aliases": "pair_receipt=continuation terminal; comparison=continuation summary",
            "parent_counts": {key: receipt[key] for key in ("step", "wm_updates", "policy_updates")},
            "usability_gates": evaluation["usability_gates"]}


def validate_contract(source, target):
    _validate_with(initialization.validate_contract, source, target)


def load_parent(evidence, target, modules, optimizers, sim, binding):
    target["horizon_experiment"] = binding
    original = initialization.validate_contract
    try:
        # The retained loader checks all tensors and Adam state; only H is admitted here.
        initialization.validate_contract = lambda source, destination: _validate_with(
            original, source, destination)
        result = initialization.load_parent(evidence, target, modules, optimizers, sim)
    finally:
        initialization.validate_contract = original
    result["type"] = "WARM_START_AFTER_COMPLETED_R03_FOR_HORIZON_PAIR"
    result["horizon_contract_entry"] = identity(__file__)
    target["source_files"] += binding["sources"]
    target["paper_deviations"] += [
        "Matched fresh-replay continuation from completed Q1 R03; H32 control versus H8 imagination."
    ]
    return result


def _validate_with(original, source, target):
    binding = target["horizon_experiment"]
    if (source["policy_trainer"]["horizon"] != 32 or binding["horizon"] not in (8, 32)
            or target["policy_trainer"]["horizon"] != binding["horizon"]):
        raise ValueError("Unregistered horizon change")
    normalized = copy.deepcopy(target)
    normalized["policy_trainer"]["horizon"] = 32
    original(source, normalized)


def matched_settings(manifest):
    settings = {key: copy.deepcopy(manifest[key]) for key in CONTRACT_KEYS}
    settings["policy_trainer"].pop("horizon")
    settings["args"] = {key: value for key, value in manifest["args"].items() if key != "output"}
    for key in ("warmup_steps", "model_warmup_updates", "learning_rate_step_offset", "code_commit",
                "source_files", "initialization"):
        settings[key] = manifest[key]
    settings["binding"] = {key: value for key, value in manifest["horizon_experiment"].items()
                           if key not in ("arm", "horizon")}
    return settings
