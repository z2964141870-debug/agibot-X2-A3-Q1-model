"""Read the actual aggregation receipts and draw one training six-panel figure."""

import argparse
import hashlib
import json
from pathlib import Path


def deliver(source, output):
    names = ("result.json", "fit_result.json", "training_trace.json", "collection_result.json")
    before = {name:(source/name).read_bytes() for name in names}
    campaign, fit, trace, collection = [json.loads(before[name]) for name in names]
    if (campaign.get("method") != fit.get("method") or fit.get("method") != "one_round_student_state_aggregation"
            or campaign.get("status") != fit.get("status") or fit.get("status") != "TRAINED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED" or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or collection.get("status") != "PASSED" or collection.get("teacher_behavior_fraction") != .5
            or collection.get("collection_behavior_is_causal") is not False or collection.get("test_sources_read") is not False
            or campaign.get("new_physical_transitions") != collection.get("actual_physical_steps")
            or fit.get("actual_updates", 0) <= 0 or fit.get("human_real_future_frames") != 0
            or trace[0]["update"] != 0 or trace[-1]["update"] != fit["actual_updates"]):
        raise ValueError("Require actual source-gated aggregation, settled cost and preserved privileged collection scope")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Use one new absolute figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    updates = [row["update"] for row in trace]
    measured = [row for row in trace if row["loss"] is not None]
    axes[0, 0].plot(updates, [row["validation_risk"] for row in trace], color="#16816a")
    axes[0, 0].axvline(fit["selected_update"], linestyle="--", color="#cc5a4b")
    axes[0, 0].set(title="Equal-source and equal-origin validation MSE", xlabel="Actual Adam updates")
    for origin, label in ((0, "Original teacher states"), (1, "Mixed student states")):
        axes[0, 1].plot(updates, [row["validation_origin_risks"][origin] for row in trace], label=label)
    axes[0, 1].set(title="Validation risk by state origin", xlabel="Actual Adam updates")
    axes[0, 1].legend(fontsize=8)
    for ax, field, title, color in ((axes[0, 2], "loss", "Recorded batch loss", "#4074a7"),
            (axes[1, 0], "gradient_norm", "Encoder gradient norm before clipping", "#9b624a")):
        ax.plot([row["update"] for row in measured], [row[field] for row in measured], color=color)
        ax.set(title=title, xlabel="Actual Adam updates")
    samples = fit["source_sample_counts"]
    axes[1, 1].bar(list(samples), list(samples.values()), color="#16816a")
    axes[1, 1].set(title="Train exposures (origin: original source ID)", ylabel="Sample count")
    axes[1, 1].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[1, 2].plot(updates, [row["elapsed_seconds"] for row in trace], color="#826194")
    axes[1, 2].set(title="Fit wall time; charged collection + fit "+str(campaign["cost_settlement"]["charged_seconds"])+"s",
                    xlabel="Actual Adam updates", ylabel="Seconds")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("Q1 causal student-state aggregation | half-teacher collection is training only | quality pending", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    if any((source/name).read_bytes() != value for name, value in before.items()):
        raise ValueError("Actual training inputs changed while plotting")
    audit = dict(stage="ACTUAL_AGGREGATION_TRAINING_FIGURES", new_inference=0, new_physics=0, new_training=0,
        actual_updates=fit["actual_updates"], selected_update=fit["selected_update"],
        charged_gpu_seconds=campaign["cost_settlement"]["charged_seconds"],
        sources={name:dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in before.items()},
        closed_loop_acceptance="PENDING", hardware_latency_ms=None)
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output)))
