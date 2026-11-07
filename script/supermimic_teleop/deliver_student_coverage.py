"""Plot completed coverage fitting with expanded and fixed validation kept separate."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def deliver(source, output):
    names = ("result.json", "fit_result.json", "training_trace.json", "collection_result.json", "dataset_receipt.json")
    records, values = [], {}
    for name in names:
        raw = (source/name).read_bytes()
        records.append(dict(path=str(source/name), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        values[name] = json.loads(raw)
    campaign, fit, trace, collection, dataset = (values[n] for n in names)
    if (campaign.get("status") != fit.get("status") or fit.get("status") != "TRAINED_PENDING_CLOSED_LOOP"
            or fit.get("method") != "owned_source_coverage" or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or collection.get("status") != "PASSED" or dataset.get("status") != "COVERAGE_TEACHER_LABELS_SAVED"
            or not trace or trace[0]["update"] != 0 or trace[-1]["update"] != fit["actual_updates"]):
        raise ValueError("Use only real completed, source-gated and settled coverage training")
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["update"] for r in trace]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, [r["validation_risk"] for r in trace], color="#15836c", label="Expanded validation")
    ax.axvline(fit["selected_update"], color="#bd5142", linestyle="--", label="Selected encoder")
    ax.set(title="Expanded validation: equal groups / clips / origins", ylabel="Action MSE")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, label in (("fixed_anchor_r1_risk", "Fixed R1 validation"), ("fixed_anchor_mixed_risk", "Fixed mixed validation")):
        ax.plot(x, [r[key] for r in trace], label=label)
    ax.set(title="Original validation anchors (unchanged datasets)", ylabel="Action MSE")
    ax.legend(fontsize=8)
    sampled = [r for r in trace if r["loss"] is not None]
    axes[0, 2].plot([r["update"] for r in sampled], [r["loss"] for r in sampled], color="#367ab6")
    axes[0, 2].set(title="Recorded training batches", ylabel="Action MSE")
    axes[1, 0].plot([r["update"] for r in sampled], [r["gradient_norm"] for r in sampled], color="#ab7655")
    axes[1, 0].axhline(1., color="#555555", linestyle=":")
    axes[1, 0].set(title="Encoder gradient norm before clipping", ylabel="L2 norm")
    exposures = fit["source_sample_counts"]
    totals = [sum(v for k, v in exposures.items() if int(k.split(":")[1]) == i) for i in range(3)]
    axes[1, 1].bar(["Old teacher", "Old mixed", "New teacher"], totals, color=["#367ab6", "#ab7655", "#15836c"])
    axes[1, 1].set(title="Actual training state-origin exposures", ylabel="Sample count")
    axes[1, 2].plot(x, [r["elapsed_seconds"] for r in trace], color="#8b649c")
    cost = campaign["cost_settlement"]["charged_seconds"]
    axes[1, 2].set(title=f"Fit time; collection + training charged {cost}s", ylabel="Seconds")
    for i, ax in enumerate(axes.flat):
        if i != 4:
            ax.set_xlabel("Actual Adam updates")
        ax.grid(alpha=.2)
    fig.suptitle("Q1 f0/history coverage experiment | frozen 32k teacher and R2 statistics | quality pending", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    columns = ["update", "validation_risk", "fixed_anchor_r1_risk", "fixed_anchor_mixed_risk",
               "fixed_anchor_non_regression", "loss", "gradient_norm", "elapsed_seconds"]
    with (output/"training_points.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key:r[key] for key in columns} for r in trace)
    summary = dict(stage="COVERAGE_REAL_TRAINING_FIGURES", source_inputs=records,
        selected_update=fit["selected_update"], actual_updates=fit["actual_updates"],
        charged_gpu_seconds=cost, coverage=fit["source_coverage"], new_inference_calls=0,
        new_physical_transitions=0, new_training_updates=0, missing_rl_metrics="not recorded by this encoder-only fit",
        new_500_update_running_mean_not_fabricated=True, closed_loop_quality="PENDING", user_accepted_checkpoint=None)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output), allow_nan=False))
