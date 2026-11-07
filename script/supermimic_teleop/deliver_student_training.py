"""Render six training panels from saved real R1 receipts, without inference."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def deliver(source, output):
    names = ("result.json", "fit_result.json", "training_trace.json", "collection_result.json", "dataset_receipt.json")
    values = {name:read(source/name) for name in names}
    campaign, fit = values["result.json"], values["fit_result.json"]
    trace, collection, dataset = (values[name] for name in names[2:])
    if (campaign.get("status") != "TRAINED_PENDING_CLOSED_LOOP" or campaign.get("source_integrity") != "PASSED"
            or fit.get("status") != "TRAINED_PENDING_CLOSED_LOOP" or fit.get("actual_updates", 0) <= 0
            or Path(fit.get("provenance", {}).get("teacher", {}).get("path", "")).name != "ckpt_32000.pt"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or collection.get("status") != "PASSED" or dataset.get("status") != "PACKING_ACTION_PARITY_PASSED"):
        raise ValueError("Require completed real32k R1 provenance, packing parity and settled cost")
    if not trace or trace[0]["update"] != 0 or trace[-1]["update"] != fit["actual_updates"]:
        raise ValueError("Training trace does not include baseline and final actual update")
    if output.exists():
        raise FileExistsError("Use one independent fresh figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True)
    updates = [row["update"] for row in trace]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    selected = fit["selected_update"]
    ax = axes[0, 0]
    ax.plot(updates, [row["validation_risk"] for row in trace], color="#16816a")
    ax.axvline(selected, color="#cc5a4b", linestyle="--", label=f"Selected update {selected}")
    ax.set(title="Validation action MSE (selection only)", xlabel="Actual Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for i, horizon in enumerate((10, 5, 4, 2, 1)):
        ax.plot(updates, [row["validation"][i]["action_mse"] for row in trace], label=f"H{horizon}")
    ax.set(title="Validation by requested preview", xlabel="Actual Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    observed = [row for row in trace if row["loss"] is not None]
    ax = axes[0, 2]
    ax.plot([row["update"] for row in observed], [row["loss"] for row in observed], color="#4074a7")
    ax.set(title="Recorded training batch loss", xlabel="Actual Adam updates", ylabel="MSE")
    ax = axes[1, 0]
    ax.plot([row["update"] for row in observed], [row["gradient_norm"] for row in observed], color="#9b624a")
    ax.axhline(1., color="#555555", linestyle=":", label="Clipping limit")
    ax.set(title="Encoder gradient norm before clipping", xlabel="Actual Adam updates", ylabel="L2 norm")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    counts = fit["actual_visible_prefix_counts"]
    ax.bar(np.arange(len(counts)), counts, color="#16816a")
    ax.set(title="Actual visible-prefix exposures", xlabel="Available future slots (not requested H)", ylabel="Training records")
    ax.set_xticks(range(11))
    ax = axes[1, 2]
    ax.plot(updates, [row["elapsed_seconds"] for row in trace], color="#826194")
    cost = campaign["cost_settlement"]["charged_seconds"]
    ax.set(title=f"Fit wall time; GPU-owning run charged {cost}s", xlabel="Actual Adam updates", ylabel="Elapsed fit seconds")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Q1 R1 encoder distillation | 32k research teacher | closed-loop quality pending", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    source_records = []
    for name in names:
        raw = (source/name).read_bytes()
        source_records.append(dict(path=str(source/name), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(stage="R1_SAVED_TRAINING_FIGURES", source_inputs=source_records,
        selected_update=selected, actual_updates=fit["actual_updates"], charged_gpu_seconds=cost,
        new_inference_calls=0, new_physical_steps=0, new_optimizer_updates=0,
        closed_loop_acceptance="PENDING", natural_teleoperation_qualified=False, hardware_latency_ms=None)
    (output/"audit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output), allow_nan=False))
