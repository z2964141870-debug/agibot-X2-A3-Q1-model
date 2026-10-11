"""Verify registered adapter sources and latest saved states before recovery."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import ROOT


def verify(experiment):
    output = ROOT / f"data/experiments/a3_controller_adapter_20261010_{experiment}"
    job = json.loads((output / "job.json").read_text())
    if job["experiment"] != experiment:
        raise ValueError("Experiment identity mismatch")
    for name, expected in job["code_sha256"].items():
        if sha256(ROOT / "script/a3" / name) != expected:
            raise ValueError(f"Registered source changed: {name}")
    metadata = json.loads((output / "training_windows.json").read_text())
    dataset = output / "training_windows.npz"
    if dataset.stat().st_size != metadata["bytes"] or sha256(dataset) != metadata["sha256"]:
        raise ValueError("Dataset identity mismatch")
    with np.load(dataset, allow_pickle=False) as stored:
        if not stored["training_allowed"].item():
            raise ValueError("Dataset is not allowed for training")
        for name in stored.files:
            value = stored[name]
            if value.dtype.kind in "fiu" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite dataset field: {name}")
    rows = []
    for objective in job["objectives"]:
        sidecars = sorted((output / objective).glob("adapter_step_*.json"))
        if not sidecars:
            raise ValueError(f"No saved state for {objective}")
        path = sidecars[-1].with_suffix(".pt")
        sidecar = json.loads(sidecars[-1].read_text())
        if path.stat().st_size != sidecar["bytes"] or sha256(path) != sidecar["sha256"]:
            raise ValueError("Checkpoint identity mismatch")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        step = int(saved["step"])
        if (step != sidecar["step"] or not 0 <= step <= job["steps"]
                or saved["objective"] != objective
                or saved["dataset_sha256"] != metadata["sha256"]
                or saved["base_policy_tensor_sha256"] != metadata["teacher_policy_tensor_sha256"]):
            raise ValueError("Checkpoint registration mismatch")
        tensors = list(saved["adapter_state_dict"].values())
        tensors += [value for state in saved["optimizer_state_dict"]["state"].values()
                    for value in state.values() if isinstance(value, torch.Tensor)]
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("Nonfinite model or optimizer")
        if len(saved["losses"]) != step or [r["step"] for r in saved["losses"]] != list(range(1, step + 1)):
            raise ValueError("Update ledger mismatch")
        if any(not np.isfinite([r["loss"], r["gradient_norm"]]).all() for r in saved["losses"]):
            raise ValueError("Nonfinite loss history")
        generator = torch.Generator()
        generator.set_state(saved["generator_state"])
        exported = path.with_suffix(".npz")
        if sha256(exported) != sidecar["export_sha256"]:
            raise ValueError("Export identity mismatch")
        with np.load(exported, allow_pickle=False) as stored:
            if int(stored["step"]) != step:
                raise ValueError("Export step mismatch")
            np.testing.assert_array_equal(stored["weights"], saved["adapter_state_dict"]["weight"].numpy().T)
            np.testing.assert_array_equal(stored["bias"], saved["adapter_state_dict"]["bias"].numpy())
            if any(not np.isfinite(stored[k]).all() for k in ("weights", "bias", "mean", "std")):
                raise ValueError("Nonfinite export")
        rows.append(dict(objective=objective, step=step, sha256=sidecar["sha256"],
                         bytes=path.stat().st_size, cpu_reload_model_optimizer_finite=True,
                         export_matches_checkpoint=True, generator_state_reload=True))
    receipt = dict(status="passed", experiment=experiment,
                   verified_at=datetime.now(timezone.utc).isoformat(),
                   boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                   registered_sources_verified=True, latest_checkpoints=rows,
                   dataset_sha256=metadata["sha256"], backup_status="LOCAL_ONLY",
                   scope="RECOVERY_INTEGRITY_ONLY_NOT_EFFECT_ACCEPTANCE")
    atomic_json(output / "restart_verification_20261011.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E20", "E21"), default="E20")
    verify(parser.parse_args().experiment)
