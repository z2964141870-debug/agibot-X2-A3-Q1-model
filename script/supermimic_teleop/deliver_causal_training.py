"""Plot six R2 training panels from real completed receipts, without inference."""

import argparse
import hashlib
import json
from pathlib import Path


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def deliver(source, output):
    names = ("result.json", "fit_result.json", "training_trace.json", "reused_dataset.json", "current_statistics.json")
    before = [identity(source/name) for name in names]
    campaign, fit, trace, dataset, statistics = [json.loads((source/name).read_text()) for name in names]
    if (campaign.get("stage") != "R2_F0_HISTORY_ENCODER_DISTILLATION"
            or campaign.get("status") != fit.get("status") or fit.get("status") != "TRAINED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED" or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or fit.get("actual_updates", 0) <= 0 or fit.get("human_real_future_frames") != 0
            or fit.get("schema") != "q1_low_latency_f0_history_encoder_v1"
            or fit.get("current_statistics") != statistics or statistics.get("fitted_split") != "train"
            or dataset.get("actual_states_and_labels_reused") is not True or dataset.get("new_physical_transitions") != 0
            or Path(fit.get("provenance", {}).get("teacher", {}).get("path", "")).name != "ckpt_32000.pt"):
        raise ValueError("Require real R2 provenance, frozen train statistics, reused states and settled cost")
    if not trace or trace[0]["update"] != 0 or trace[-1]["update"] != fit["actual_updates"]:
        raise ValueError("Keep baseline and all actual fit updates")
    if output.exists() or not output.is_absolute():
        raise FileExistsError("Use a fresh absolute figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    updates = [row["update"] for row in trace]
    observed = [row for row in trace if row["loss"] is not None]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(updates, [row["validation_risk"] for row in trace], color="#16816a")
    ax.axvline(fit["selected_update"], color="#cc5a4b", linestyle="--", label="Selected by validation")
    ax.set(title="Validation action MSE (equal source clips)", xlabel="Actual Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for i, source_row in enumerate(trace[0]["validation"]):
        ax.plot(updates, [row["validation"][i]["action_mse"] for row in trace],
                label="Source "+str(source_row["source_clip_id"]))
    ax.set(title="Validation by original source clip", xlabel="Actual Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    ax.plot([row["update"] for row in observed], [row["loss"] for row in observed], color="#4074a7")
    ax.set(title="Recorded training batch loss", xlabel="Actual Adam updates", ylabel="MSE")
    ax = axes[1, 0]
    ax.plot([row["update"] for row in observed], [row["gradient_norm"] for row in observed], color="#9b624a")
    ax.axhline(1., color="#555555", linestyle=":")
    ax.set(title="Encoder gradient norm before clipping", xlabel="Actual Adam updates", ylabel="L2 norm")
    ax = axes[1, 1]
    samples = fit["source_sample_counts"]
    ax.bar(list(samples), list(samples.values()), color="#16816a")
    ax.set(title="Training source exposures; future slots0", xlabel="Original source clip ID", ylabel="Sample count")
    ax = axes[1, 2]
    ax.plot(updates, [row["elapsed_seconds"] for row in trace], color="#826194")
    ax.set(title="Fit wall time; run charged "+str(campaign["cost_settlement"]["charged_seconds"])+"s",
           xlabel="Actual Adam updates", ylabel="Elapsed fit seconds")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Q1 R2 f0/history encoder | reused real32k teacher states | closed-loop quality pending", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    if any(identity(Path(row["path"])) != row for row in before):
        raise ValueError("Saved R2 inputs changed while plotting")
    audit = dict(stage="R2_SAVED_TRAINING_FIGURES", source_inputs=before,
        actual_updates=fit["actual_updates"], selected_update=fit["selected_update"],
        charged_gpu_seconds=campaign["cost_settlement"]["charged_seconds"], human_real_future_frames=0,
        new_inference_calls=0, new_physical_steps=0, new_optimizer_updates=0,
        closed_loop_acceptance="PENDING", natural_teleoperation_qualified=False, hardware_latency_ms=None)
    (output/"audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output), allow_nan=False))
