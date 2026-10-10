"""Independent CPU receipt for the zero-update gradient diagnostic."""

import json
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
from script.a3.fullchain_support import ROOT


def finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite captured rollout")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def main():
    torch.set_num_threads(1)
    output = ROOT / "data/experiments/a3_gradient_20261010_E08"
    result = json.loads((output / "gradient_diagnostic.json").read_text())
    state = json.loads((output / "state.json").read_text())
    if state["status"] != "complete" or result["optimizer_steps"] != 0 or result["global_step"] != 0:
        raise ValueError("Not a successful zero-update diagnostic")
    if result["parameters_before_sha256"] != result["parameters_after_sha256"]:
        raise ValueError("Parameters changed during probing")
    capture = output / "rollout.pt"
    if sha256(capture) != result["rollout_sha256"] or capture.stat().st_size != result["rollout_bytes"]:
        raise ValueError("Rollout capture changed")
    rollout = torch.load(capture, map_location="cpu", weights_only=False)
    finite(rollout)
    checkpoint = output / "initial_rollout/model_step_000000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if validate_payload(payload) != 0 or payload["optimizer_state_dict"]["state"]:
        raise ValueError("Initial checkpoint has optimizer updates")
    logs = ROOT / "logs/a3_gradient_20261010_E08"
    steps = logs / "actual_optimizer_steps.jsonl"
    if steps.exists() and steps.read_text().strip():
        raise ValueError("Actual optimizer update was recorded")
    rows = [dict(batch=batch["batch"], losses=batch["losses"],
                 geometry=batch["gradient_geometry"], ratio=batch["ratio"],
                 initial_value_explained_variance=batch["initial_value_explained_variance"])
            for batch in result["batches"]]
    names = ["script/a3/gradient_probe.py", "script/a3/run_gradient_probe.py",
             "script/a3/test_gradient_probe.py", "script/a3/gradient_report.py",
             "data/experiments/a3_gradient_20261010_E07/job.json",
             "data/experiments/a3_gradient_20261010_E07/state.json",
             "logs/a3_gradient_20261010_E07/console.log"]
    names.extend(str(path.relative_to(ROOT)) for parent in (output, logs)
                 for path in parent.rglob("*") if path.is_file())
    receipt = dict(experiment="E07_FAILED_E08_COMPLETE", initial_code_commit="d28561b966309401a57432b53e73fbe7e50b1b33",
        cpu_rollout_reload_finite=True, checkpoint_global_step=0, optimizer_steps=0,
        initial_checkpoint_networks_optimizer_finite=True, gradient_parameters_unchanged=True,
        rollout_changed_parameters=result["rollout_changed_parameters"],
        batches=rows, artifacts=[dict(path=name, bytes=(ROOT/name).stat().st_size,
                       sha256=sha256(ROOT/name)) for name in sorted(set(names))],
        conclusions=dict(aux_dominates_actor_gradient=False, initial_ppo_ratio_consistency_verified=False,
                         positive_policy_improvement_verified=False), backup_status="LOCAL_ONLY")
    atomic_json(ROOT / "data/manifests/a3_gradient_20261010.json", receipt)
    print(json.dumps(dict(checks="PASS", artifact_count=len(receipt["artifacts"]), optimizer_steps=0)))


if __name__ == "__main__":
    main()
