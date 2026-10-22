"""Initialize an independent run from a verified completed training run."""

import json
from pathlib import Path

import torch

from supermimic.a3_assets import identity


def load_initialization(checkpoint_path, calibration_audit, contract, wm, policy, critic, wt, pt):
    checkpoint_path = Path(checkpoint_path)
    source_id = identity(checkpoint_path)
    receipt_path = checkpoint_path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "TRAINING_FINISHED_PENDING_EVALUATION" or source_id not in receipt["artifacts"]:
        raise ValueError("Initialization must belong to a completed training run")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(source[key] != receipt[key] for key in ("step", "wm_updates", "policy_updates")):
        raise ValueError("Initialization must be the completed run's final state")
    source_contract = source["manifest"]
    for name in ("assets", "sim", "world_model", "policy", "critic", "wm_trainer", "policy_trainer",
                 "train_clips", "heldout_clips", "backend_receipt"):
        if source_contract[name] != contract[name]:
            raise ValueError(f"Initialization contract differs: {name}")
    model_source = source_id
    audit_id = None
    wm_state, wm_opt = source["wm"], source["wm_opt"]
    if calibration_audit is not None:
        calibration_audit = Path(calibration_audit)
        audit = json.loads(calibration_audit.read_text())
        if audit["status"] != "WORLD_MODEL_CALIBRATION_INDEPENDENTLY_VERIFIED" or audit["passed_for_candidate"] is not True:
            raise ValueError("Calibrated model did not pass independent candidate verification")
        if audit["source_checkpoint"] != source_id:
            raise ValueError("Calibrated model belongs to a different source policy")
        if identity(audit["receipt"]["path"]) != audit["receipt"]:
            raise ValueError("Calibrated model receipt changed")
        models = [row for row in audit["artifacts"] if Path(row["path"]).name == "calibrated_world_model.pt"]
        if len(models) != 1 or identity(models[0]["path"]) != models[0]:
            raise ValueError("Calibrated model identity differs")
        model_source = models[0]
        calibrated = torch.load(model_source["path"], map_location="cpu", weights_only=False)
        if calibrated["source_checkpoint"] != source_id or calibrated["wm_config"] != contract["world_model"]:
            raise ValueError("Calibrated model contract differs")
        wm_state, wm_opt = calibrated["wm"], calibrated["wm_opt"]
        audit_id = identity(calibration_audit)
    wm.load_state_dict(wm_state, strict=True)
    policy.load_state_dict(source["policy"], strict=True)
    critic.load_state_dict(source["critic"], strict=True)
    wt.opt.load_state_dict(wm_opt)
    pt.opt.load_state_dict(source["policy_opt"])
    pt.critic_opt.load_state_dict(source["critic_opt"])
    initial_steps = {}
    for name, optimizer in (("wm_opt", wt.opt), ("policy_opt", pt.opt), ("critic_opt", pt.critic_opt)):
        initial_steps[name] = sorted({int(state["step"]) for state in optimizer.state.values()})
        if not initial_steps[name]:
            raise ValueError("Initialization optimizer has no training state")
        if not all(torch.isfinite(value).all() for state in optimizer.state.values()
                   for value in state.values() if torch.is_tensor(value)):
            raise ValueError("Nonfinite initialization optimizer state")
    if not all(torch.isfinite(value).all() for module in (wm, policy, critic) for value in module.state_dict().values()):
        raise ValueError("Nonfinite initialization network state")
    return {"type": "warm_start_in_new_run", "source_checkpoint": source_id,
            "initialization_entry": identity(__file__),
            "source_receipt": identity(receipt_path), "model_source": model_source,
            "calibration_audit": audit_id, "initial_optimizer_steps": initial_steps,
            "source_counters": {name: source[name] for name in ("step", "wm_updates", "policy_updates")},
            "learning_rate_step_offset": source_contract.get("learning_rate_step_offset", 0) + source["step"],
            "replay_and_random_seed": "Fresh replay, adaptive sampler, simulator episodes and explicit run seed; no exact process resume."}
