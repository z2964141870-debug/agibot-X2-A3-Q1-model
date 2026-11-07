"""Make six-panel figures only from the completed paired training receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def deliver(source, output):
    names = ["result.json", "paired_fit_result.json", "transition_label_audit.json"]
    names += [f"{arm}/{name}" for arm in ("action_only", "next_state")
              for name in ("fit_result.json", "training_trace.json")]
    before = {name:(source/name).read_bytes() for name in names}
    records = {name:json.loads(value) for name, value in before.items()}
    campaign, pair = records["result.json"], records["paired_fit_result.json"]
    if (campaign.get("status") != "PAIRED_FIT_PASSED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or pair.get("matched_source_exposures") is not True
            or records["transition_label_audit.json"].get("status") != "PASSED"):
        raise ValueError("Need actual paired training, source/label gates and settled cost")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Use a new absolute figure output")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    for arm in ("action_only", "next_state"):
        fit, trace = (records[f"{arm}/{name}"] for name in ("fit_result.json", "training_trace.json"))
        if trace[-1]["update"] != fit["actual_updates"] or fit["actual_updates"] <= 0:
            raise ValueError("Training trace has missing actual updates")
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        x = [row["update"] for row in trace]
        measured = [row for row in trace if row["loss"] is not None]
        axes[0, 0].plot(x, [row["validation_risk"] for row in trace])
        axes[0, 0].axvline(fit["selected_update"], color="#b94046", linestyle="--")
        axes[0, 0].set(title="Validation action MSE; selected update", xlabel="Actual Adam updates")
        for i, label in ((0, "Original states"), (1, "Mixed behavior states")):
            axes[0, 1].plot(x, [r["validation_origin_risks"][i] for r in trace], label=label)
        axes[0, 1].set(title="Validation by state origin", xlabel="Actual Adam updates")
        axes[0, 1].legend(fontsize=8)
        for ax, key, title in ((axes[0, 2], "action_loss", "Recorded batch action MSE"),
                (axes[1, 0], "auxiliary_state_loss", "Recorded batch next-state MSE"),
                (axes[1, 1], "gradient_norm", "Encoder gradient before clipping")):
            ax.plot([r["update"] for r in measured], [r[key] for r in measured])
            ax.set(title=title, xlabel="Actual Adam updates")
        counts = fit["source_sample_counts"]
        axes[1, 2].bar(list(counts), list(counts.values()), color="#16816a")
        axes[1, 2].set(title="Matched source exposure (origin: source ID)")
        axes[1, 2].tick_params(axis="x", labelrotation=90, labelsize=7)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.suptitle(f"Q1 {arm} | same f0 deployment encoder | closed-loop quality pending", fontsize=12)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(arm+"_training_six_panel."+suffix), dpi=160)
        plt.close(fig)
    if any((source/name).read_bytes() != raw for name, raw in before.items()):
        raise ValueError("Actual training records changed during plotting")
    audit = dict(stage="PAIRED_AUXILIARY_TRAINING_FIGURES", new_training=0, new_physics=0,
        sources={name:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in before.items()},
        actual_optimizer_updates=pair["actual_optimizer_updates"], hardware_latency_ms=None,
        user_accepted_checkpoint=None)
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output)))
