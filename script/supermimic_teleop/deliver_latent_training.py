"""Six panels from actual latent training and the unchanged historical control."""

import argparse
import hashlib
import json
from pathlib import Path


def deliver(source, control, output):
    paths = [source/name for name in ("result.json", "fit_result.json", "training_trace.json", "teacher_latent_audit.json")]
    paths += [control/name for name in ("fit_result.json", "training_trace.json")]
    before = {str(path):path.read_bytes() for path in paths}
    campaign, fit, trace, labels, prior, prior_trace = [json.loads(before[str(path)]) for path in paths]
    if (campaign.get("status") != "TRAINED_PENDING_CLOSED_LOOP" or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED" or labels.get("status") != "PASSED"
            or fit.get("control_retrained") is not False
            or fit["batch_identity_sha256"] != prior["batch_identity_sha256"]
            or fit["actual_updates"] != trace[-1]["update"] or fit["actual_updates"] != prior["actual_updates"]):
        raise ValueError("Need the actual settled/source-gated fit and matched historical control")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Use a fresh absolute figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    x = [row["update"] for row in trace]
    axes[0, 0].plot(x, [row["validation_risk"] for row in trace], label="Latent supervision")
    axes[0, 0].plot([r["update"] for r in prior_trace], [r["validation_risk"] for r in prior_trace], label="Preserved action control")
    axes[0, 0].axvline(fit["selected_update"], color="#b94046", linestyle="--")
    axes[0, 0].set(title="Validation action MSE; selected update")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x, [row["validation_latent_mse"] for row in trace])
    axes[0, 1].set(title="Validation teacher-latent MSE (offline labels)")
    for i, label in ((0, "Original states"), (1, "Mixed states")):
        axes[0, 2].plot(x, [r["validation_origin_risks"][i] for r in trace], label=label)
    axes[0, 2].set(title="Validation action MSE by state origin")
    axes[0, 2].legend(fontsize=8)
    measured = [r for r in trace if r["loss"] is not None]
    for key, label in (("action_loss", "Action MSE"), ("latent_loss", "Latent MSE")):
        axes[1, 0].plot([r["update"] for r in measured], [r[key] for r in measured], label=label)
    axes[1, 0].set(title="Recorded training batch losses")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([r["update"] for r in measured], [r["gradient_norm"] for r in measured])
    axes[1, 1].set(title="Encoder gradient before clipping")
    counts = fit["source_sample_counts"]
    axes[1, 2].bar(list(counts), list(counts.values()), color="#16816a")
    axes[1, 2].tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[1, 2].set(title="Exact matched source exposure")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_xlabel("Actual Adam updates" if ax is not axes[1, 2] else "Origin: source ID")
    fig.suptitle("Q1 f0/history latent supervision | historical control reused | closed-loop pending", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("latent_training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    if any(Path(path).read_bytes() != raw for path, raw in before.items()):
        raise ValueError("Actual records changed during plotting")
    audit = dict(stage="ACTUAL_LATENT_TRAINING_SIX_PANEL", new_training=0, new_physics=0,
        actual_optimizer_updates=fit["actual_updates"], reused_control_updates=prior["actual_updates"],
        control_retrained=False, sources={path:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for path, raw in before.items()}, hardware_latency_ms=None, user_accepted_checkpoint=None)
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "control", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.control, args.output)))
